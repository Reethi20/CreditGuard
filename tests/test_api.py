
from fastapi.testclient import TestClient

from backend.api.main import app


client = TestClient(app)


def create_valid_transaction():
    """Create a synthetic transaction with all 30 features."""

    features = {
        "Time": 1000,
        "Amount": 100,
    }

    for i in range(1, 29):
        features[f"V{i}"] = 0

    return {"features": features}


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_valid_prediction():
    response = client.post(
        "/predict",
        json=create_valid_transaction()
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in ["FRAUD", "LEGITIMATE"]
    assert 0 <= result["fraud_probability"] <= 1
    assert 0 <= result["fraud_probability_percentage"] <= 100
    assert result["risk_category"] in ["HIGH", "LOW"]
    assert result["threshold"] == 0.5


def test_missing_feature_rejected():
    transaction = create_valid_transaction()

    del transaction["features"]["V14"]

    response = client.post(
        "/predict",
        json=transaction
    )

    assert response.status_code == 422
