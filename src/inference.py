import pickle
import pandas as pd


def load_model(model_path="src/model/model.pkl"):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_single(model, data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return prediction, probability
