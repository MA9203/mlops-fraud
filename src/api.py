import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json

# Import only basic monitoring functions (no Evidently dependencies)
from src.monitoring.monitoring import log_prediction, generate_metrics

app = FastAPI(title="Fraud Detection API")

# 🔥 Le modèle doit être défini ici pour être global
model = None
feature_names = None


class Transaction(BaseModel):
    features: list


class DriftReportResponse(BaseModel):
    timestamp: str
    reference_data_shape: list
    current_data_shape: list
    drift_detected: bool
    number_of_drifted_features: int
    total_features: int


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
    if model is None:
        # Load model automatically if needed
        load_model()

    X = transaction.features

    if len(X) != len(feature_names):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid number of features: got {len(X)}, "
                   f"expected {len(feature_names)}"
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

        # Log the prediction for monitoring
        log_prediction(transaction.features, int(pred), float(proba))

        return {
            "prediction": int(pred),
            "probability": proba
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/metrics")
def get_metrics():
    """Get model performance metrics"""
    try:
        metrics = generate_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Metrics generation failed: {str(e)}"
        )


@app.post("/monitoring/drift/check")
def check_data_drift():
    """Run data drift detection and return results"""
    try:
        # Import here to avoid loading Evidently at startup
        from src.monitoring.data_drift import run_data_drift_monitoring
        drift_summary = run_data_drift_monitoring()
        return drift_summary
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Data drift detection failed: {str(e)}"
        )


@app.get("/monitoring/drift/report")
def get_drift_report():
    """Get the latest data drift report"""
    try:
        report_path = "reports/data_drift_report.json"
        if not os.path.exists(report_path):
            raise HTTPException(
                status_code=404,
                detail="No drift report found. Run drift detection first."
            )
        with open(report_path, 'r') as f:
            report = json.load(f)
        return report
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve drift report: {str(e)}"
        )


@app.post("/monitoring/model/check")
def check_model_drift():
    """Run model drift detection and return results"""
    try:
        # Import here to avoid loading Evidently at startup
        from src.monitoring.model_drift import run_model_drift_monitoring
        model_drift_report = run_model_drift_monitoring()
        return model_drift_report
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model drift detection failed: {str(e)}"
        )


@app.get("/monitoring/model/report")
def get_model_drift_report():
    """Get the latest model drift report"""
    try:
        report_path = "reports/model_drift_report.json"
        if not os.path.exists(report_path):
            raise HTTPException(
                status_code=404,
                detail="No model drift report found. "
                       "Run model drift detection first."
            )
        with open(report_path, 'r') as f:
            report = json.load(f)
        return report
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve model drift report: {str(e)}"
        )
