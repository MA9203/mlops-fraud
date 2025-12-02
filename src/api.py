import os
import mlflow
from joblib import load
from fastapi import FastAPI

app = FastAPI()

USE_MLFLOW = os.getenv("USE_MLFLOW", "false").lower() == "true"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PKL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

MODEL_URI = "runs:/c83550f503c24fdcaefdd67707e7ac89/model"

model = None


def load_model_prod():
    """Chargement du modèle .pkl pour Render."""
    if not os.path.exists(MODEL_PKL_PATH):
        raise FileNotFoundError(f"Model file missing: {MODEL_PKL_PATH}")
    return load(MODEL_PKL_PATH)


def load_model_dev():
    """Chargement du modèle MLflow quand disponible."""
    return mlflow.sklearn.load_model(MODEL_URI)


@app.on_event("startup")
def startup_event():
    global model

    if USE_MLFLOW:
        print("🔥 Loading MLflow model…")
        model = load_model_dev()
    else:
        print("📦 Loading production model (.pkl)…")
        model = load_model_prod()

    print("✅ Model loaded successfully!")
