from fastapi import APIRouter

from backend.app.schemas.benchmark import RandomBenchmarkRequest
from backend.app.services.benchmark_service import run_random_baseline
from backend.app.services.run_store import save_run

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


@router.post("/random-baseline")
async def create_random_baseline_benchmark(request: RandomBenchmarkRequest):
    run = run_random_baseline(
        num_samples=request.num_samples,
        seed=request.seed,
    )
    save_run(run)
    return run
