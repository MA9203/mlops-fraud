from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd

# 🔥 Import corrigé
from .inference import load_model, predict_single

app = FastAPI(title="Fraud Detection API")

MODEL_URI = "runs:/c83550f503c24fdcaefdd67707e7ac89/model"



model = None

class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

@app.on_event("startup")
def load_model_on_startup():
    global model
    model = load_model(MODEL_URI)

@app.post("/predict")
def predict(transaction: Transaction):
    data = transaction.dict()
    result = predict_single(model, data)
    return result
