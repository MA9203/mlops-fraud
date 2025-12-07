import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import mlflow
import mlflow.sklearn

DATA_PROCESSED = "data/processed/creditcard_processed.csv"
MODEL_OUT = "src/model/model.pkl"

def train():

    print("Chargement du dataset preprocessé…")
    df = pd.read_csv(DATA_PROCESSED)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Initialisation de MLflow…")
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("fraud_detection_pipeline")

    with mlflow.start_run(run_name="baseline_logreg"):
        print("Entraînement du modèle LogisticRegression…")
        model = LogisticRegression(max_iter=3000)
        model.fit(X_train, y_train)

        preds = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, preds)
        pr = average_precision_score(y_test, preds)

        mlflow.log_metric("ROC_AUC", roc)
        mlflow.log_metric("PR_AUC", pr)

        mlflow.sklearn.log_model(model, artifact_path="model")

        print("Export du modèle pour l’API…")
        os.makedirs("src/model", exist_ok=True)
        joblib.dump(model, MODEL_OUT)

        print("Modèle enregistré dans :", MODEL_OUT)

if __name__ == "__main__":
    train()
