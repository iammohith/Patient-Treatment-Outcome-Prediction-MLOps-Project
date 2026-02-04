from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import time

# Environment variables for paths
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
ENCODERS_PATH = os.getenv("ENCODERS_PATH", "data/processed/encoders.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "data/processed/scaler.pkl")

app = FastAPI(title="Drug Treatment Outcome Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics
REQUEST_COUNT = Counter('request_count', 'Total number of requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency', ['endpoint'])

# Global variables for artifacts
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

@app.on_event("startup")
def load_artifacts():
    global model, encoders, scaler
    try:
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Artifacts loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        # We might want to crash here in prod, but for now we'll just log it
        # raise e

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    start_time = time.time()
    REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc()
    
    if not model or not encoders or not scaler:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        data = request.dict()
        df = pd.DataFrame([data])
        
        # Preprocessing
        # 1. Label Encoding
        for col, le in encoders.items():
            if col in df.columns:
                # Handle unseen labels carefully or just assume valid input for now
                try:
                    df[col] = le.transform(df[col])
                except ValueError:
                    # Fallback for unknown categories if needed, or error
                     raise HTTPException(status_code=400, detail=f"Unknown category for {col}: {data[col]}")

        # 2. Scaling
        num_cols = ['Age', 'Dosage_mg', 'Treatment_Duration_days']
        df[num_cols] = scaler.transform(df[num_cols])
        
        # Ensure column order matches training
        # We can get feature names from the model if available, or rely on consistency
        # XGBoost handles numpy arrays, so column order matters if passing simple array
        # Ideally, we pass DMatrix or ensuring DataFrame columns are correct
        # For this scaffold, let's assume the column order is consistent with training dataframe structure
        # (Age, Gender, Condition, Drug_Name, Dosage_mg, Treatment_Duration_days, Side_Effects)
        # We might need to reorder df to match
        
        expected_cols = ['Age', 'Gender', 'Condition', 'Drug_Name', 'Dosage_mg', 'Treatment_Duration_days', 'Side_Effects']
        df = df[expected_cols]
        
        prediction = model.predict(df)
        score = float(prediction[0])
        
        REQUEST_LATENCY.labels(endpoint='/predict').observe(time.time() - start_time)
        return {"Improvement_Score": score}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
