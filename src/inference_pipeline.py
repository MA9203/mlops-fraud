import json
import joblib
import pandas as pd
import os

MODEL_PATH = "src/model/model.pkl"
FEATS_PATH = "src/model/feature_names.json"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found.")
    return joblib.load(MODEL_PATH)

def load_feature_names():
    if not os.path.exists(FEATS_PATH):
        raise FileNotFoundError("Feature names file not found.")
    with open(FEATS_PATH, "r") as f:
        return json.load(f)

def run_inference(input_dict):
    model = load_model()
    expected = load_feature_names()

    df = pd.DataFrame([input_dict])
    df = df[expected]

    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    return {
        "prediction": int(pred),
        "probability": float(proba)
    }


if __name__ == "__main__":
    sample = {k: 0 for k in load_feature_names()}
    print(run_inference(sample))
