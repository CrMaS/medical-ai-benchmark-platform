from pathlib import Path

RUNS_DIR = Path("runs")

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]

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
    }
]
