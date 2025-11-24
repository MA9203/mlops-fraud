from utils.loaders import load_dataset

def test_load_dataset():
    X, y = load_dataset("data/processed/creditcard_processed.csv")
    assert len(X) == len(y)
    assert "Class" not in X.columns
