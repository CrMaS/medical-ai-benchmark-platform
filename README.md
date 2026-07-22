# Medical AI Benchmark Platform

Medical AI Benchmark Platform is a Dockerized FastAPI application for evaluating
skin-lesion classification models on HAM10000-style datasets.

It demonstrates:

- medical image classification evaluation
- PyTorch exported-model medical-image inference
- validated JPEG, PNG, and WebP uploads
- FastAPI backend development
- Dockerized deployment
- reproducible benchmark runs
- per-class performance reporting
- MLOps-oriented experiment tracking

This project is for research and educational benchmarking only. It is not a
medical device and must not be used for diagnosis or clinical decision-making.

## Run locally

```bash
python -m venv .venv
./.venv/bin/pip install -r backend/requirements.txt
./.venv/bin/uvicorn backend.app.main:app --reload
```

Open `http://localhost:8000/docs` for the interactive API documentation. The
main endpoints list datasets, models, and saved runs, and execute the
deterministic random baseline:

```bash
curl -X POST http://localhost:8000/benchmarks/random-baseline \
  -H 'content-type: application/json' \
  -d '{"num_samples": 200, "seed": 42}'
```

Benchmark runs are stored under `runs/`. Set `MEDICAL_AI_RUNS_DIR` to use a
different location.

## Configure medical-image inference

The inference service deliberately does not ship pretend medical weights. Place
a trained and independently validated HAM10000-compatible PyTorch artifact at
`models/skin_lesion_classifier.pt2`. Its input, preprocessing, output, and
class-order contract is documented in `models/README.md`.

You can choose another artifact or execution device with environment variables:

```bash
export MEDICAL_AI_MODEL_PATH=/path/to/skin_lesion_classifier.pt2
export MEDICAL_AI_DEVICE=cpu  # or cuda
./.venv/bin/uvicorn backend.app.main:app --reload
```

Check whether the model is configured and loaded:

```bash
curl http://localhost:8000/inference/status
```

Classify a dermoscopic image:

```bash
curl -X POST http://localhost:8000/inference/skin-lesion \
  -F 'image=@/path/to/lesion.jpg;type=image/jpeg'
```

The image is decoded in memory and is not persisted by the inference endpoint.
The response contains the predicted class, all seven class probabilities, image
metadata, and the SHA-256 identity of the loaded model. Restart the service after
replacing a loaded model artifact.

The same inference pipeline is available from the command line:

```bash
./.venv/bin/python scripts/run_inference.py /path/to/lesion.jpg \
  --model models/skin_lesion_classifier.pt2
```

The same benchmark can be run from the command line, including from outside the
repository directory:

```bash
./.venv/bin/python scripts/run_benchmark.py --num-samples 200 --seed 42 --save
```

## Docker and tests

```bash
docker compose up --build
./.venv/bin/python -m pytest
```

The random baseline and sample dataset configuration are synthetic pipeline
fixtures. Inference becomes medically meaningful only when you supply a properly
trained, externally validated, and documented model artifact. Outputs are never
a diagnosis and must not be used for clinical decision-making.
