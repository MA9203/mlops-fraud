from src.api import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_wrong_dim():
    # On envoie volontairement un mauvais nombre de features
    body = {"features": [0] * 10}

    response = client.post("/predict", json=body)

    # L’API doit retourner une erreur 400
    assert response.status_code == 400
    assert "Invalid number of features" in response.json()["detail"]


def test_predict_endpoint():
    sample = {
        "features": [
            39069.0,  2.0, -6.160209, 8.288441, -0.360762, 2.771247, 3.823486,
            -3.564795, 2.532856, -2.049711, 1.475739, -1.371376, 1.051136,
            -0.436560, 0.373717, -0.075482, -1.146653, 0.399141, -0.238253,
            -0.025162, -0.226487, -0.308950, -0.071583, -0.215537, -0.193889,
            -0.465211, -0.058132, 0.020083, -0.082360, -0.467923
        ]
    }

    response = client.post("/predict", json=sample)

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()
