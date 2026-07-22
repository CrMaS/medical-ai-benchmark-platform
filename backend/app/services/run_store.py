import json
import os
import re
import tempfile
from pathlib import Path

from backend.app.core.config import RUNS_DIR

RUN_ID_PATTERN = re.compile(r"^run_\d{8}_\d{6}_[0-9a-f]{8}$")


def ensure_runs_dir() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def save_run(run: dict) -> Path:
    run_id = run.get("run_id")
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run contains an invalid run_id")

    ensure_runs_dir()
    output_path = RUNS_DIR / f"{run_id}.json"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=RUNS_DIR,
            prefix=f".{run_id}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(run, temporary_file, indent=2)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return output_path


def list_runs() -> list[dict]:
    ensure_runs_dir()
    runs = []

    for path in RUNS_DIR.glob("*.json"):
        try:
            with path.open(encoding="utf-8") as f:
                run = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(run, dict):
            runs.append(run)

    return sorted(
        runs,
        key=lambda run: str(run.get("created_at", "")),
        reverse=True,
    )


def get_run(run_id: str) -> dict | None:
    ensure_runs_dir()
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        return None

    path = RUNS_DIR / f"{run_id}.json"

    if not path.exists():
        return None

    try:
        with path.open(encoding="utf-8") as f:
            run = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    return run if isinstance(run, dict) else None
