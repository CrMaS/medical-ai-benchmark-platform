from fastapi import APIRouter, HTTPException

from backend.app.services.run_store import get_run, list_runs

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("")
async def get_runs():
    return list_runs()


@router.get("/{run_id}")
async def get_single_run(run_id: str):
    run = get_run(run_id)

    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    return run
