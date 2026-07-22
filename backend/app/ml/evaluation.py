from numbers import Integral
from typing import Sequence

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> dict:
    """Compute metrics over the complete configured label space."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    class_names = list(class_names)

    if not class_names:
        raise ValueError("class_names must contain at least one class")
    if len(set(class_names)) != len(class_names):
        raise ValueError("class_names must be unique")
    if not y_true:
        raise ValueError("y_true and y_pred must contain at least one sample")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    labels = list(range(len(class_names)))
    invalid_labels = [
        value
        for value in (*y_true, *y_pred)
        if not isinstance(value, Integral) or value not in labels
    ]
    if invalid_labels:
        raise ValueError(
            f"labels must be integers between 0 and {len(class_names) - 1}"
        )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # Averaging the per-class recalls keeps the configured class space stable,
        # including when a small evaluation split does not contain every class.
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }

    per_class = {}
    for idx, class_name in enumerate(class_names):
        per_class[class_name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    metrics["per_class"] = per_class
    metrics["confusion_matrix"] = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    ).tolist()

    return metrics
