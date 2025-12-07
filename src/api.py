import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Fraud Detection API")

# 🔥 Le modèle doit être défini ici pour être global
model = None
feature_names = None


class Transaction(BaseModel):
    features: list


# 🔥 Fonction universelle pour charger le modèle (test + prod)
def load_model():
    global model, feature_names
    model_path = "src/model/model.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = joblib.load(model_path)

    # Important pour l’ordre des features
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    else:
        raise ValueError("Model does not contain feature_names_in_ attribute.")


@app.on_event("startup")
def startup_event():
    """Chargement du modèle au démarrage de l’API"""
    try:
        load_model()
        print("Model loaded successfully at startup.")
    except Exception as e:
        print(f"Error while loading model: {e}")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "n_features_expected": len(feature_names) if feature_names else None
    }


@app.post("/predict")
def predict(transaction: Transaction):
    global model, feature_names

    if model is None:
        # 🔥 Solution pour les tests : charger automatiquement si nécessaire
        load_model()

    X = transaction.features

    if len(X) != len(feature_names):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid number of features: got {len(X)}, expected {len(feature_names)}"
        )

    try:
        import numpy as np
        X = np.array(X).reshape(1, -1)

        pred = model.predict(X)[0]
        proba = (
            float(model.predict_proba(X)[0][1])
            if hasattr(model, "predict_proba")
            else float(pred)
        )

        return {
            "prediction": int(pred),
            "probability": proba
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
