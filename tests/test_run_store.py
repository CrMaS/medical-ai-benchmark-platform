import json

import pytest

from backend.app.services import run_store


@pytest.fixture
def isolated_runs_dir(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    monkeypatch.setattr(run_store, "RUNS_DIR", runs_dir)
    return runs_dir


def make_run(
    run_id="run_20260722_120000_0123abcd",
    created_at="2026-07-22T12:00:00+00:00",
):
    return {"run_id": run_id, "created_at": created_at, "metrics": {"accuracy": 0.5}}


def test_save_list_and_get_run(isolated_runs_dir):
    older = make_run(created_at="2026-07-21T12:00:00+00:00")
    newer = make_run(
        run_id="run_20260722_130000_deadbeef",
        created_at="2026-07-22T13:00:00+00:00",
    )

    output_path = run_store.save_run(older)
    run_store.save_run(newer)

    assert output_path == isolated_runs_dir / f"{older['run_id']}.json"
    assert run_store.get_run(older["run_id"]) == older
    assert run_store.list_runs() == [newer, older]


def test_store_rejects_unsafe_id_and_ignores_corrupt_files(isolated_runs_dir):
    with pytest.raises(ValueError, match="invalid run_id"):
        run_store.save_run(make_run(run_id="../outside"))

    isolated_runs_dir.mkdir()
    (isolated_runs_dir / "broken.json").write_text("{", encoding="utf-8")
    (isolated_runs_dir / "list.json").write_text(
        json.dumps([1, 2]),
        encoding="utf-8",
    )

    assert run_store.get_run("../outside") is None
    assert run_store.list_runs() == []
