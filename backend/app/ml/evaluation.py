from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_classification_metrics(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

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
