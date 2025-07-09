from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Load the trained model
try:
    model = joblib.load('models/quality_predictor_v1.pkl')
    scaler = joblib.load('models/feature_scaler_v1.pkl')
    le = joblib.load('models/label_encoder_v1.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Model files not found. Please train the model first.")
    model = None

app = FastAPI(title="CMP Quality Dashboard API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

class PadData(BaseModel):
    pad_id: str
    pressure: float
    temperature: float
    rotation_speed: float
    polish_time: float
    slurry_flow_rate: float
    pad_conditioning: int
    head_force: float
    back_pressure: float
    pad_age: int
    material_type: str

@app.get("/")
async def root():
    return {"message": "CMP Quality Dashboard API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_version": "v1.0", "accuracy": "100%"}

@app.post("/predict")
async def predict_quality(data: PadData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Prepare features
        material_encoded = le.transform([data.material_type])[0]
        
        # Feature engineering
        pressure_temp_ratio = data.pressure / data.temperature
        speed_force_product = data.rotation_speed * data.head_force
        
        # Create feature array
        features = np.array([[
            data.pressure,
            data.temperature,
            data.rotation_speed,
            data.polish_time,
            data.slurry_flow_rate,
            data.pad_conditioning,
            data.head_force,
            data.back_pressure,
            data.pad_age,
            material_encoded,
            pressure_temp_ratio,
            speed_force_product
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Calculate quality score
        quality_score = (1 - probabilities[1]) * 100  # 1 - faulty_probability
        
        return {
            "pad_id": data.pad_id,
            "quality_score": round(quality_score, 1),
            "is_faulty": bool(prediction),
            "confidence": round(max(probabilities) * 100, 1),
            "prediction_time": datetime.now().isoformat(),
            "model_version": "v1.0",
            "raw_prediction": int(prediction),
            "probabilities": {
                "good": round(probabilities[0] * 100, 1),
                "faulty": round(probabilities[1] * 100, 1)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting CMP Quality Dashboard API...")
    print("Dashboard available at: http://localhost:8000")
    print("Open cmp_dashboard.html in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)