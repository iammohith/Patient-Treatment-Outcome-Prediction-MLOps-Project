# Patient Treatment Outcome Prediction MLOps Pipeline

This repository contains an end-to-end MLOps pipeline for predicting patient improvement scores based on drug treatment data.

## Project Structure

```text
├── data/                # Data storage
│   ├── raw/             # Raw CSV files
│   └── processed/       # Encoded/Scaled data (generated)
├── src/                 # Source code
│   ├── data/            # Data processing scripts
│   ├── train/           # Training scripts
│   └── api/             # FastAPI application
├── models/              # Serialized models and artifacts
├── docker/              # Dockerfiles
├── k8s/                 # Kubernetes manifests
├── web/                 # Frontend application
├── dvc.yaml             # DVC pipeline configuration
└── .github/workflows/   # CI/CD pipelines
```

## Setup & Usage

### 1. Prerequisites

- Python 3.9+
- Docker
- Kubernetes (Minikube or similar)
- DVC

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Data Pipeline (DVC)

Run the full data ingestion and training pipeline:

```bash
dvc repro
```

This will:

1. Preprocess `data/raw/real_drug_dataset.csv`
2. Train the XGBoost model
3. Save metrics to `metrics.json`

### 4. Run API Locally

```bash
uvicorn src.api.main:app --reload
```

Visit `http://localhost:8000/docs` for Swagger UI.

### 5. Run Frontend Locally

Open `web/index.html` in your browser. Ensure the API is running on port 8000.

### 6. Docker

Build images:

```bash
docker build -f docker/api/Dockerfile -t drug-prediction-api:latest .
docker build -f docker/web/Dockerfile -t drug-prediction-web:latest .
```

### 7. Kubernetes Deployment

Deploy to cluster:

```bash
kubectl apply -f k8s/api.yaml
kubectl apply -f k8s/web.yaml
```

## Monitoring

The API provides Prometheus metrics at `/metrics`.
Key metrics:

- `request_count`: Total requests by endpoint
- `request_latency_seconds`: Latency distribution
