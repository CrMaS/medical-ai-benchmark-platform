import json
from pathlib import Path

from backend.app.core.config import RUNS_DIR


def ensure_runs_dir() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def save_run(run: dict) -> Path:
    ensure_runs_dir()
    output_path = RUNS_DIR / f"{run['run_id']}.json"

    with output_path.open("w") as f:
        json.dump(run, f, indent=2)

    return output_path


def list_runs() -> list[dict]:
    ensure_runs_dir()
    runs = []

    for path in sorted(RUNS_DIR.glob("*.json"), reverse=True):
        with path.open() as f:
            runs.append(json.load(f))

    return runs


def get_run(run_id: str) -> dict | None:
    ensure_runs_dir()
    path = RUNS_DIR / f"{run_id}.json"

    if not path.exists():
        return None

    with path.open() as f:
        return json.load(f)
