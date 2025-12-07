import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

DATA_PROCESSED = "data/processed/creditcard_processed.csv"
MODEL_PATH = "src/model/model.pkl"
OUT_METRICS = "metrics.json"


def evaluate():

    print("Chargement dataset…")
    df = pd.read_csv(DATA_PROCESSED)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Chargement modèle…")
    model = joblib.load(MODEL_PATH)
    preds = model.predict_proba(X_test)[:, 1]

    print("Calcul métriques…")
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, preds))
    }

    print("Écriture metrics.json…")
    with open(OUT_METRICS, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    evaluate()
