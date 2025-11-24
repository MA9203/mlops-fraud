import mlflow
import pandas as pd

def load_model(model_uri: str):
    return mlflow.sklearn.load_model(model_uri)

def predict_single(model, data: dict):
    df = pd.DataFrame([data])

    # 🔥 1. Récupérer l’ordre utilisé par le modèle
    expected_features = model.feature_names_in_.tolist()

    # 🔥 2. Réordonner exactement dans cet ordre
    df = df[expected_features]

    # 🔥 3. Faire la prédiction
    prediction = model.predict(df)[0]

    # Cas logistic regression → predict_proba OK
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0][1]
    else:
        # Cas XGBoost modèle binaire → sortie directe
        proba = float(model.predict_proba(df)[0][1]) if hasattr(model, "predict_proba") else float(prediction)

    return {
        "prediction": int(prediction),
        "probability": float(proba)
    }
