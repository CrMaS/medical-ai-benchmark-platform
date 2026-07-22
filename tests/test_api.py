import httpx
import pytest

from backend.app.main import app
from backend.app.services import run_store


@pytest.mark.anyio
async def test_api_workflow(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "RUNS_DIR", tmp_path / "runs")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        assert (await client.get("/health")).json() == {"status": "healthy"}
        assert (await client.get("/datasets")).status_code == 200
        assert (await client.get("/models")).status_code == 200

        response = await client.post(
            "/benchmarks/random-baseline",
            json={"num_samples": 10, "seed": 3},
        )
        assert response.status_code == 200
        created_run = response.json()

        listed_runs = await client.get("/runs")
        assert listed_runs.status_code == 200
        assert listed_runs.json() == [created_run]

        fetched_run = await client.get(f"/runs/{created_run['run_id']}")
        assert fetched_run.status_code == 200
        assert fetched_run.json() == created_run


@pytest.mark.anyio
async def test_api_validation_and_missing_run(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "RUNS_DIR", tmp_path / "runs")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        invalid_response = await client.post(
            "/benchmarks/random-baseline",
            json={"num_samples": 9},
        )
        assert invalid_response.status_code == 422
        assert (await client.get("/runs/not-a-run-id")).status_code == 404
