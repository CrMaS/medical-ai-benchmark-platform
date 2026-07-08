# scripts/run_benchmark.py

import json
import random
from pathlib import Path

from backend.app.ml.evaluation import compute_classification_metrics


CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def main():
    n = 200
    y_true = [random.randint(0, len(CLASS_NAMES) - 1) for _ in range(n)]
    y_pred = [random.randint(0, len(CLASS_NAMES) - 1) for _ in range(n)]

    metrics = compute_classification_metrics(y_true, y_pred, CLASS_NAMES)

    result = {
        "model": "random-baseline",
        "dataset": "ham10000-sample",
        "num_samples": n,
        "metrics": metrics,
    }

    Path("runs").mkdir(exist_ok=True)
    with open("runs/random_baseline.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()