from io import BytesIO

import pytest
import torch
from PIL import Image

from backend.app.ml.inference import (
    InvalidImageError,
    MedicalImageClassifier,
    ModelNotConfiguredError,
    ModelOutputError,
)

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


class SevenClassFixture(torch.nn.Module):
    def forward(self, image):
        logits = torch.arange(7, dtype=image.dtype, device=image.device)
        return logits.unsqueeze(0)


class InvalidOutputFixture(torch.nn.Module):
    def forward(self, image):
        return torch.zeros((1, 2), dtype=image.dtype, device=image.device)


def save_exported_model(path, model):
    example = torch.zeros((1, 3, 16, 16))
    exported_model = torch.export.export(model, (example,))
    torch.export.save(exported_model, path)


def make_png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (24, 12), color=(80, 40, 20)).save(output, format="PNG")
    return output.getvalue()


def make_classifier(model_path):
    return MedicalImageClassifier(
        model_path=model_path,
        class_names=CLASS_NAMES,
        input_size=16,
        device="cpu",
    )


def test_exported_model_image_inference(tmp_path):
    model_path = tmp_path / "classifier.pt2"
    save_exported_model(model_path, SevenClassFixture())
    classifier = make_classifier(model_path)

    assert classifier.status()["loaded"] is False
    result = classifier.predict(make_png(), filename="lesion.png")

    assert result["prediction"]["class_id"] == "vasc"
    assert result["filename"] == "lesion.png"
    assert result["image"] == {"format": "PNG", "width": 24, "height": 12}
    assert len(result["probabilities"]) == 7
    assert sum(row["probability"] for row in result["probabilities"]) == pytest.approx(1)
    assert len(result["model_sha256"]) == 64
    assert classifier.status()["loaded"] is True


def test_inference_requires_a_configured_model(tmp_path):
    classifier = make_classifier(tmp_path / "missing.ts")

    with pytest.raises(ModelNotConfiguredError, match="No trained PyTorch model"):
        classifier.predict(make_png())


def test_inference_rejects_invalid_images(tmp_path):
    classifier = make_classifier(tmp_path / "missing.ts")

    with pytest.raises(InvalidImageError, match="not a valid image"):
        classifier.predict(b"not an image")


def test_inference_validates_model_output_shape(tmp_path):
    model_path = tmp_path / "invalid-output.pt2"
    save_exported_model(model_path, InvalidOutputFixture())
    classifier = make_classifier(model_path)

    with pytest.raises(ModelOutputError, match="Expected model output shape"):
        classifier.predict(make_png())
