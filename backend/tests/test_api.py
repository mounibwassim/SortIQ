import pytest  # pyre-ignore
# Verified Web-Only Import Sync
from fastapi.testclient import TestClient  # pyre-ignore
from main import app  # pyre-ignore

client = TestClient(app)

def test_health_check_endpoint():
    # Because tests run without lifespan events by default in some setups,
    # or the model mockup isn't loaded, we are just testing routing.
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "db_connected" in data
    assert "model_loaded" in data

def test_stats_endpoint():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_scans" in data
    assert "class_distribution" in data

def test_history_endpoint():
    response = client.get("/history?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
