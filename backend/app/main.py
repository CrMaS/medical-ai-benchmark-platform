from fastapi import FastAPI

from backend.app.api.routes_benchmarks import router as benchmarks_router
from backend.app.api.routes_datasets import router as datasets_router
from backend.app.api.routes_inference import router as inference_router
from backend.app.api.routes_models import router as models_router
from backend.app.api.routes_runs import router as runs_router

app = FastAPI(
    title="Medical AI Benchmark Platform",
    description="Benchmarking API for skin-lesion classification models.",
    version="0.2.0",
)

app.include_router(benchmarks_router)
app.include_router(datasets_router)
app.include_router(inference_router)
app.include_router(models_router)
app.include_router(runs_router)


@app.get("/")
async def root():
    return {
        "name": "Medical AI Benchmark Platform",
        "status": "running",
        "version": "0.2.0",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
