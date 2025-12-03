import os
from joblib import load
from fastapi import FastAPI

app = FastAPI()

# ⚙️ Mode MLflow optionnel (local seulement)
USE_MLFLOW = os.getenv("USE_MLFLOW", "false").lower() == "true"

# 📦 Chemin du modèle .pkl pour Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PKL_PATH = os.path.join(BASE_DIR, "model.pkl")

model = None


def load_model_prod():
    """Chargement du modèle .pkl pour Render."""
    if not os.path.exists(MODEL_PKL_PATH):
        raise FileNotFoundError(f"❌ Model file missing: {MODEL_PKL_PATH}")
    return load(MODEL_PKL_PATH)


@app.on_event("startup")
def startup_event():
    global model

    print("📦 Loading production model (.pkl)…")
    model = load_model_prod()

    print("✅ Model loaded successfully!")
