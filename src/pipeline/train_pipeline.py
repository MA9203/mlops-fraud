import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import pickle
import json
import mlflow
import mlflow.sklearn
import os


# ===========================
# 📥 Load dataset
# ===========================
def load_data():
    DATA_PATH = "data/raw/creditcard.csv"
    print("📥 Chargement du dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"✔ Données chargées : {df.shape}")
    return df


# ===========================
# ⚙️ Preprocessing
# ===========================
def preprocess(df):
    print("⚙️ Préprocessing...")

    # Target = Class
    df = df.rename(columns={"Class": "is_fraud"})
    
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("✔ Split terminé :")
    print("  - Train :", X_train.shape)
    print("  - Test :", X_test.shape)

    return X_train, X_test, y_train, y_test


# ===========================
# 🧠 Train
# ===========================
def train(X_train, y_train):
    print("🧠 Entraînement du modèle...")

    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)

    print("✔ Modèle entraîné.")
    return model


# ===========================
# 📊 Evaluate
# ===========================
def evaluate(model, X_test, y_test):
    print("📊 Évaluation...")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("""
===== 📈 METRIQUES =====
Accuracy : {:.4f}
Precision : {:.4f}
Recall : {:.4f}
F1-score : {:.4f}
ROC-AUC : {:.4f}

Matrice de confusion :
{}
========================
""".format(acc, prec, rec, f1, auc, cm))

    return acc, prec, rec, f1, auc


# ===========================
# 💾 Save model
# ===========================
def save_model(model):
    os.makedirs("models", exist_ok=True)
    model_path = "models/model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"💾 Modèle enregistré dans {model_path}")


# ===========================
# 📝 Save metrics for API
# ===========================
def save_metrics_json(acc, prec, rec, f1, auc):
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc
    }

    with open("models/last_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("📄 Metrics saved → models/last_metrics.json")


# ===========================
# 🧪 MLflow Logging
# ===========================
def log_mlflow(model, acc, prec, rec, f1, auc):
    print("📝 Logging dans MLflow...")

    # 🚀 IMPORTANT : en Docker, on utilise le service "mlflow_ui"
    mlflow.set_tracking_uri("http://mlflow_ui:5000")

    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run():
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        mlflow.sklearn.log_model(model, artifact_path="model")

    print("✔ MLflow logging terminé.")


# ===========================
# 🚀 MAIN PIPELINE
# ===========================
if __name__ == "__main__":

    print("🚀 Pipeline d'entraînement lancé...")

    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)

    model = train(X_train, y_train)

    acc, prec, rec, f1, auc = evaluate(model, X_test, y_test)

    save_model(model)
    save_metrics_json(acc, prec, rec, f1, auc)
    log_mlflow(model, acc, prec, rec, f1, auc)

    print("🎉 Pipeline complet — modèle mis à jour & métriques sauvegardées.")
