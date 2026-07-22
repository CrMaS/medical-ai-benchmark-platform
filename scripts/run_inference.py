import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import (
    CLASS_NAMES,
    MAX_IMAGE_PIXELS,
    MEDICAL_MODEL_DEVICE,
    MEDICAL_MODEL_INPUT_SIZE,
    MEDICAL_MODEL_PATH,
)
from backend.app.ml.inference import InferenceError, MedicalImageClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a trained PyTorch skin-lesion classifier on one image."
    )
    parser.add_argument("image", type=Path, help="Path to a JPEG, PNG, or WebP image.")
    parser.add_argument(
        "--model",
        type=Path,
        default=MEDICAL_MODEL_PATH,
        help="Path to the trained PyTorch .pt2 or TorchScript model artifact.",
    )
    parser.add_argument(
        "--device",
        default=MEDICAL_MODEL_DEVICE,
        help="PyTorch device such as auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=MEDICAL_MODEL_INPUT_SIZE,
        help="Square model input size in pixels.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Image file not found: {args.image}")

    classifier = MedicalImageClassifier(
        model_path=args.model,
        class_names=CLASS_NAMES,
        input_size=args.input_size,
        device=args.device,
        max_image_pixels=MAX_IMAGE_PIXELS,
    )
    try:
        result = classifier.predict(args.image.read_bytes(), args.image.name)
    except InferenceError as exc:
        raise SystemExit(f"Inference failed: {exc}") from exc

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
