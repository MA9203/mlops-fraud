import os
import pickle
import pandas as pd


def load_model(model_dir="src/model"):
    """Load model.pkl from the model directory."""
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def predict_single(model, data: dict):
    """Run prediction on a single transaction."""

    # Convert dict -> DataFrame
    df = pd.DataFrame([data])

    # Respecter l’ordre EXACT des features
    expected = model.feature_names_in_.tolist()
    df = df[expected]

    # Prediction
    pred = model.predict(df)[0]

    # Probability
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(df)[0][1])
    else:
        proba = float(pred)

    return {
        "prediction": int(pred),
        "probability": proba
    }
