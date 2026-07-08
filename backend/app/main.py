# backend/app/main.py

from fastapi import FastAPI
import json
from pathlib import Path

app = FastAPI(title="Medical AI Benchmark Platform")


@app.get("/")
def root():
    return {
        "name": "Medical AI Benchmark Platform",
        "status": "running",
    }


@app.get("/runs")
def list_runs():
    run_files = Path("runs").glob("*.json")
    runs = []

    for file in run_files:
        with open(file) as f:
            runs.append(json.load(f))

    return runs