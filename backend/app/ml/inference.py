import hashlib
import threading
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

from backend.app.core.config import (
    CLASS_DISPLAY_NAMES,
    CLASS_NAMES,
    MAX_IMAGE_PIXELS,
    MEDICAL_MODEL_DEVICE,
    MEDICAL_MODEL_INPUT_SIZE,
    MEDICAL_MODEL_PATH,
)

ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)


class InferenceError(RuntimeError):
    """Base exception for image inference failures."""


class ModelNotConfiguredError(InferenceError):
    """Raised when no trained model artifact is available."""


class ModelLoadError(InferenceError):
    """Raised when a configured model cannot be loaded."""


class InvalidImageError(InferenceError):
    """Raised when uploaded bytes are not a supported image."""


class ModelOutputError(InferenceError):
    """Raised when a model does not satisfy the classifier output contract."""


class MedicalImageClassifier:
    """Lazy-loading PyTorch classifier for HAM10000-compatible models."""

    def __init__(
        self,
        model_path: Path,
        class_names: Sequence[str],
        input_size: int = 224,
        device: str = "auto",
        max_image_pixels: int = 40_000_000,
    ) -> None:
        if not class_names:
            raise ValueError("class_names must contain at least one class")
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if max_image_pixels <= 0:
            raise ValueError("max_image_pixels must be positive")

        self.model_path = Path(model_path).expanduser().resolve()
        self.class_names = list(class_names)
        self.input_size = input_size
        self.max_image_pixels = max_image_pixels
        self.device = self._resolve_device(device)
        self._model: torch.nn.Module | None = None
        self._model_sha256: str | None = None
        self._load_lock = threading.Lock()
        self._transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
            ]
        )

    @staticmethod
    def _resolve_device(requested_device: str) -> torch.device:
        requested_device = requested_device.lower()
        if requested_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise ModelLoadError("CUDA was requested but is not available")
        try:
            return torch.device(requested_device)
        except RuntimeError as exc:
            raise ModelLoadError(f"Unsupported inference device: {requested_device}") from exc

    def status(self) -> dict[str, Any]:
        return {
            "model_id": "skin-lesion-pytorch",
            "configured": self.model_path.is_file(),
            "loaded": self._model is not None,
            "artifact": str(self.model_path),
            "artifact_sha256": self._model_sha256,
            "device": str(self.device),
            "input_size": self.input_size,
            "class_names": self.class_names,
            "preprocessing": {
                "color_mode": "RGB",
                "resize": [self.input_size, self.input_size],
                "normalization_mean": list(NORMALIZATION_MEAN),
                "normalization_std": list(NORMALIZATION_STD),
            },
        }

    def _load_model(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model
        if not self.model_path.is_file():
            raise ModelNotConfiguredError(
                f"No trained PyTorch model found at {self.model_path}"
            )

        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                if self.model_path.suffix == ".pt2":
                    exported_program = torch.export.load(self.model_path)
                    model = exported_program.module().to(self.device)
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        model = torch.jit.load(
                            str(self.model_path),
                            map_location=self.device,
                        )
                    model.eval()
            except Exception as exc:
                raise ModelLoadError(
                    "The configured PyTorch model could not be loaded"
                ) from exc

            self._model_sha256 = self._sha256(self.model_path)
            self._model = model
            return model

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _decode_image(self, image_bytes: bytes) -> tuple[Image.Image, dict[str, Any]]:
        if not image_bytes:
            raise InvalidImageError("The uploaded image is empty")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(image_bytes)) as source_image:
                    image_format = source_image.format
                    width, height = source_image.size
                    if image_format not in ALLOWED_IMAGE_FORMATS:
                        raise InvalidImageError(
                            "Only JPEG, PNG, and WebP images are supported"
                        )
                    if width <= 0 or height <= 0 or width * height > self.max_image_pixels:
                        raise InvalidImageError("Image dimensions are invalid or too large")
                    source_image.load()
                    image = ImageOps.exif_transpose(source_image).convert("RGB")
        except InvalidImageError:
            raise
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise InvalidImageError("The uploaded file is not a valid image") from exc
        except Image.DecompressionBombWarning as exc:
            raise InvalidImageError("The uploaded image dimensions are too large") from exc

        return image, {
            "format": image_format,
            "width": width,
            "height": height,
        }

    def _extract_logits(self, output: Any) -> torch.Tensor:
        if isinstance(output, dict):
            output = output.get("logits")
        elif isinstance(output, (tuple, list)):
            output = output[0] if output else None

        if not isinstance(output, torch.Tensor):
            raise ModelOutputError("Model output must contain a logits tensor")
        if output.ndim == 1:
            output = output.unsqueeze(0)
        expected_shape = (1, len(self.class_names))
        if tuple(output.shape) != expected_shape:
            raise ModelOutputError(
                f"Expected model output shape {expected_shape}, got {tuple(output.shape)}"
            )
        if not torch.isfinite(output).all().item():
            raise ModelOutputError("Model output contains non-finite values")
        return output

    def predict(self, image_bytes: bytes, filename: str | None = None) -> dict[str, Any]:
        image, image_metadata = self._decode_image(image_bytes)
        model = self._load_model()
        input_tensor = self._transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self._extract_logits(model(input_tensor))
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()

        top_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        probability_rows = [
            {
                "class_id": class_name,
                "display_name": CLASS_DISPLAY_NAMES.get(class_name, class_name),
                "probability": float(probabilities[index]),
            }
            for index, class_name in enumerate(self.class_names)
        ]
        probability_rows.sort(key=lambda row: row["probability"], reverse=True)

        return {
            "model_id": "skin-lesion-pytorch",
            "model_sha256": self._model_sha256,
            "filename": filename,
            "image": image_metadata,
            "prediction": {
                "class_id": self.class_names[top_index],
                "display_name": CLASS_DISPLAY_NAMES.get(
                    self.class_names[top_index], self.class_names[top_index]
                ),
                "confidence": float(probabilities[top_index]),
            },
            "probabilities": probability_rows,
            "disclaimer": (
                "Research use only. This output is not a diagnosis and must not be used "
                "for clinical decision-making."
            ),
        }


classifier = MedicalImageClassifier(
    model_path=MEDICAL_MODEL_PATH,
    class_names=CLASS_NAMES,
    input_size=MEDICAL_MODEL_INPUT_SIZE,
    device=MEDICAL_MODEL_DEVICE,
    max_image_pixels=MAX_IMAGE_PIXELS,
)


def get_classifier() -> MedicalImageClassifier:
    return classifier


async def get_classifier_dependency() -> MedicalImageClassifier:
    return classifier
