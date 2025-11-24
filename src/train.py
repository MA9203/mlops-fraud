import yaml
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split

from utils.loaders import load_dataset
from utils.metrics import evaluate
from models.baseline import build_logistic
from models.xgboost_model import build_xgb

def main():

    with open("src/config.yaml") as f:
        config = yaml.safe_load(f)

    X, y = load_dataset(config["data"]["processed_path"])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=y
    )

    mlflow.set_experiment("fraud_detection_pipeline")

    # Logistic Regression
    with mlflow.start_run(run_name="logistic_regression"):
        model = build_logistic(config)
        model.fit(X_train, y_train)
        roc, pr = evaluate(model, X_val, y_val)

        mlflow.log_metric("ROC_AUC", roc)
        mlflow.log_metric("PR_AUC", pr)
        mlflow.sklearn.log_model(model, "model")

    # XGBoost
    with mlflow.start_run(run_name="xgboost_model"):
        model = build_xgb(config)
        model.fit(X_train, y_train)
        roc, pr = evaluate(model, X_val, y_val)

        mlflow.log_metric("ROC_AUC", roc)
        mlflow.log_metric("PR_AUC", pr)
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()



