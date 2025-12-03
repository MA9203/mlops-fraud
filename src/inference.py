import pandas as pd

def predict_single(model, data: dict):
    df = pd.DataFrame([data])

    # 🔥 1. Récupérer l’ordre utilisé par le modèle
    expected_features = model.feature_names_in_.tolist()

    # 🔥 2. Réordonner exactement dans cet ordre
    df = df[expected_features]

    # 🔥 3. Faire la prédiction
    prediction = model.predict(df)[0]

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(df)[0][1])
    else:
        proba = float(prediction)

    return {
        "prediction": int(prediction),
        "probability": proba
    }
