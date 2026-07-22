import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = Path(
    os.environ.get("MEDICAL_AI_RUNS_DIR", PROJECT_ROOT / "runs")
).expanduser().resolve()
MODELS_DIR = PROJECT_ROOT / "models"
MEDICAL_MODEL_PATH = Path(
    os.environ.get(
        "MEDICAL_AI_MODEL_PATH",
        MODELS_DIR / "skin_lesion_classifier.pt2",
    )
).expanduser().resolve()
MEDICAL_MODEL_DEVICE = os.environ.get("MEDICAL_AI_DEVICE", "auto").lower()
MEDICAL_MODEL_INPUT_SIZE = int(os.environ.get("MEDICAL_AI_INPUT_SIZE", "224"))
MAX_IMAGE_BYTES = int(
    os.environ.get("MEDICAL_AI_MAX_IMAGE_BYTES", str(10 * 1024 * 1024))
)
MAX_IMAGE_PIXELS = int(os.environ.get("MEDICAL_AI_MAX_IMAGE_PIXELS", "40000000"))

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]

CLASS_DISPLAY_NAMES = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

DATASETS = [
    {
        "id": "ham10000-sample",
        "name": "HAM10000-style sample dataset",
        "classes": CLASS_NAMES,
        "status": "demo",
        "description": "Demo dataset configuration for skin-lesion classification benchmarking.",
    }
]

MODELS = [
    {
        "id": "random-baseline",
        "name": "Random Baseline",
        "framework": "Python",
        "status": "available",
        "description": "A deterministic random baseline used to test the benchmark pipeline.",
    },
    {
        "id": "skin-lesion-pytorch",
        "name": "Skin Lesion PyTorch Classifier",
        "framework": "PyTorch",
        "status": "requires_configuration",
        "description": "A configurable seven-class HAM10000-compatible image classifier.",
    },
]
