import os
import json
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

# Monitoring
from src.monitoring.monitoring import log_prediction, generate_metrics

app = FastAPI(title="Fraud Detection API")

# ============================
# Paths
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "last_metrics.json")

USE_MLFLOW = os.getenv("USE_MLFLOW", "false").lower() == "true"


# ============================
# Load model
# ============================
def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"⚠ Aucun modèle trouvé : {MODEL_PATH}")
        return None

    try:
        model = joblib.load(MODEL_PATH)
        print(f"✔ Modèle chargé : {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        return None


# Hook appelé par FastAPI (vient du remote, on le garde)
@app.on_event("startup")
def startup_event():
    global model
    print("📦 Loading model…")
    model = load_model()
    print("✅ Model ready!")


# ============================
# Load metrics
# ============================
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return {"message": "No metrics available yet."}

    try:
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"error": "Unable to read metrics file."}


# ============================
# Pydantic schema
# ============================
class Transaction(BaseModel):
    features: list


# ============================
# Health check
# ============================
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


# ============================
# Version
# ============================
@app.get("/version")
def model_version():
    if os.path.exists(MODEL_PATH):
        timestamp = os.path.getmtime(MODEL_PATH)
        version = str(datetime.fromtimestamp(timestamp))
    else:
        version = "unknown"

    return {
        "model_version": version,
        "last_metrics": load_metrics()
    }


# ============================
# Prediction endpoint
# ============================
@app.post("/predict")
def predict(transaction: Transaction):

    if model is None:
        return {"error": "Model not loaded"}

    try:
        features = np.array(transaction.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]

    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

    # Monitoring logging
    try:
        log_prediction(transaction.features, int(prediction), float(proba))
    except Exception as e:
        print(f"[Warning] Monitoring log failed: {e}")

    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": float(proba)
    }


# ============================
# Metrics
# ============================
@app.get("/metrics")
def metrics():
    try:
        return generate_metrics()
    except Exception as e:
        return {"error": f"Failed to load metrics: {e}"}


# ============================
# Root
# ============================
@app.get("/")
def root():
    return {
        "message": "Fraud Detection API running 🚀",
        "endpoints": ["/predict", "/health", "/version", "/metrics"]
    }
