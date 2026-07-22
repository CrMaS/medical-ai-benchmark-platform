import pytest

from backend.app.core.config import CLASS_NAMES
from backend.app.services.benchmark_service import run_random_baseline


def test_random_baseline_is_reproducible():
    first = run_random_baseline(num_samples=25, seed=7)
    second = run_random_baseline(num_samples=25, seed=7)

    assert first["metrics"] == second["metrics"]
    assert first["label_distribution"] == second["label_distribution"]
    assert list(first["label_distribution"]) == CLASS_NAMES
    assert sum(first["label_distribution"].values()) == 25
    assert first["run_id"] != second["run_id"]


@pytest.mark.parametrize("count", [9, 10_001])
def test_random_baseline_rejects_invalid_sample_count(count):
    with pytest.raises(ValueError, match="between 10 and 10000"):
        run_random_baseline(num_samples=count)
