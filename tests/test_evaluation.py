import pytest

from backend.app.ml.evaluation import compute_classification_metrics


def test_metrics_use_complete_class_space_without_warnings():
    metrics = compute_classification_metrics(
        y_true=[0, 0, 1],
        y_pred=[0, 2, 1],
        class_names=["a", "b", "c"],
    )

    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 1 + 0) / 3)
    assert metrics["confusion_matrix"] == [[1, 0, 1], [0, 1, 0], [0, 0, 0]]
    assert metrics["per_class"]["c"]["support"] == 0


@pytest.mark.parametrize(
    ("y_true", "y_pred", "class_names", "message"),
    [
        ([], [], ["a"], "at least one sample"),
        ([0], [], ["a"], "same length"),
        ([1], [0], ["a"], "between 0 and 0"),
        ([0], [0], [], "at least one class"),
        ([0], [0], ["a", "a"], "unique"),
    ],
)
def test_metrics_reject_invalid_inputs(y_true, y_pred, class_names, message):
    with pytest.raises(ValueError, match=message):
        compute_classification_metrics(y_true, y_pred, class_names)
