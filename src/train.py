import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import mlflow
import mlflow.sklearn

df = pd.read_csv("data/raw/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("fraud_detection_pipeline")

with mlflow.start_run(run_name="baseline_logreg"):

    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, preds)
    pr = average_precision_score(y_test, preds)

    mlflow.log_metric("ROC_AUC", roc)
    mlflow.log_metric("PR_AUC", pr)

    # LOG OBLIGATOIRE DU MODÈLE !!!
    mlflow.sklearn.log_model(model, artifact_path="model")

print("Model trained and logged successfully.")
