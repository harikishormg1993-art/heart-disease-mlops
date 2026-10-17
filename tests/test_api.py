from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root_and_health():
    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 200


def test_invalid_payload_is_rejected():
    response = client.post("/predict", json={"age": -5})
    assert response.status_code == 422


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text
