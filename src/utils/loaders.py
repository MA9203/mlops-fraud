import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y
