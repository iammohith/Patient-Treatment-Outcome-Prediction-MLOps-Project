from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import time
import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("drug_prediction_api")

# Environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
ENCODERS_PATH = os.getenv("ENCODERS_PATH", "data/processed/encoders.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "data/processed/scaler.pkl")
API_KEY = os.getenv("API_KEY", "secret-token")  # Default for dev, change in prod!

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=403,
        detail="Could not validate credentials"
    )

app = FastAPI(
    title="Drug Treatment Outcome Prediction API",
    description="Production-ready MLOps inference service for patient outcome prediction.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics
REQUEST_COUNT = Counter('request_count', 'Total number of requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency', ['endpoint'])

# Global artifacts
model = None
encoders = None
scaler = None

class PredictionRequest(BaseModel):
    Age: int
    Gender: str
    Condition: str
    Drug_Name: str
    Dosage_mg: float
    Treatment_Duration_days: int
    Side_Effects: str

class PredictionResponse(BaseModel):
    Improvement_Score: float
    Model_Version: str = "v1"

@app.on_event("startup")
def load_artifacts():
    global model, encoders, scaler
    try:
        logger.info("Loading model artifacts...")
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        scaler = joblib.load(SCALER_PATH)
        logger.info("Artifacts loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        # In production, we might want to crash if artifacts fail
        # raise e

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "service": "drug-prediction-api"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(get_api_key)])
def predict(request: PredictionRequest):
    start_time = time.time()
    
    if not model or not encoders or not scaler:
        logger.error("Model not initialized")
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='503').inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Received prediction request for Patient Age: {request.Age}, Drug: {request.Drug_Name}")
        data = request.dict()
        df = pd.DataFrame([data])
        
        # Preprocessing
        for col, le in encoders.items():
            if col in df.columns:
                try:
                    df[col] = le.transform(df[col])
                except ValueError:
                    logger.warning(f"Unknown category encountered for {col}: {data[col]}")
                    raise HTTPException(status_code=400, detail=f"Unknown category for {col}: {data[col]}")

        # Scaling
        num_cols = ['Age', 'Dosage_mg', 'Treatment_Duration_days']
        df[num_cols] = scaler.transform(df[num_cols])
        
        # Ensure column order matches training
        expected_cols = ['Age', 'Gender', 'Condition', 'Drug_Name', 'Dosage_mg', 'Treatment_Duration_days', 'Side_Effects']
        df = df[expected_cols]
        
        prediction = model.predict(df)
        score = float(prediction[0])
        
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint='/predict').observe(latency)
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='200').inc()
        
        logger.info(f"Prediction success. Score: {score}")
        return {"Improvement_Score": score}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='500').inc()
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
