from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_predict_wrong_dim():
    body = {"features": [0] * 10}  # mauvaise dimension

    response = client.post("/predict", json=body)

    assert response.status_code == 400
    assert "Invalid number of features" in response.json()["detail"]
