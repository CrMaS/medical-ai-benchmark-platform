import random
import uuid
from collections import Counter
from datetime import datetime, timezone

from backend.app.core.config import CLASS_NAMES
from backend.app.ml.evaluation import compute_classification_metrics


def _make_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"run_{timestamp}_{suffix}"


def run_random_baseline(num_samples: int = 200, seed: int = 42) -> dict:
    if not 10 <= num_samples <= 10_000:
        raise ValueError("num_samples must be between 10 and 10000")

    rng = random.Random(seed)
    num_classes = len(CLASS_NAMES)

    y_true = [rng.randint(0, num_classes - 1) for _ in range(num_samples)]
    y_pred = [rng.randint(0, num_classes - 1) for _ in range(num_samples)]

    metrics = compute_classification_metrics(y_true, y_pred, CLASS_NAMES)

    label_distribution = Counter(y_true)

    return {
        "run_id": _make_run_id(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "id": "random-baseline",
            "name": "Random Baseline",
        },
        "dataset": {
            "id": "ham10000-sample",
            "name": "HAM10000-style sample dataset",
        },
        "task": "multi-class skin lesion classification",
        "num_samples": num_samples,
        "seed": seed,
        "class_names": CLASS_NAMES,
        "label_distribution": {
            class_name: label_distribution[class_idx]
            for class_idx, class_name in enumerate(CLASS_NAMES)
        },
        "metrics": metrics,
    }
