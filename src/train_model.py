import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

DATA = "data/processed/creditcard_processed.csv"
MODEL_OUT = "src/model/model.pkl"


def train():
    df = pd.read_csv(DATA)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)

    os.makedirs("src/model", exist_ok=True)
    joblib.dump(model, MODEL_OUT)


if __name__ == "__main__":
    train()
