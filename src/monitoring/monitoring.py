import os
import json
import datetime

LOG_FILE = "monitoring/predictions_log.json"


# =========================================
# 1. Logging des prédictions
# =========================================
def log_prediction(features, prediction, probability):

    os.makedirs("monitoring", exist_ok=True)

    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "features": features,
        "prediction": prediction,
        "probability": probability
    }

    # Append dans un fichier JSON (liste)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            json.dump([log_entry], f, indent=4)
    else:
        with open(LOG_FILE, "r") as f:
            data = json.load(f)

        data.append(log_entry)

        with open(LOG_FILE, "w") as f:
            json.dump(data, f, indent=4)


# =========================================
# 2. Génération de métriques globales
# =========================================
def generate_metrics():
    if not os.path.exists(LOG_FILE):
        return {"error": "No prediction logs found"}

    with open(LOG_FILE, "r") as f:
        data = json.load(f)

    total = len(data)
    frauds = sum(1 for entry in data if entry["prediction"] == 1)
    avg_proba = sum(entry["probability"] for entry in data) / total

    return {
        "total_predictions": total,
        "fraud_predictions": frauds,
        "fraud_rate": frauds / total if total > 0 else 0,
        "average_probability": avg_proba,
    }
