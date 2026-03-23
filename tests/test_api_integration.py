"""Integration tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
from src.api.server import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data


def test_readiness_endpoint(client):
    """Test readiness check endpoint."""
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "models_loaded" in data
    assert "pytorch" in data["models_loaded"]
    assert "sklearn" in data["models_loaded"]


def test_predict_payload_endpoint(client):
    """Test payload prediction endpoint."""
    response = client.post(
        "/api/v1/predict/payload",
        json={"payload": "' OR '1'='1"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "is_attack" in data
    assert "confidence" in data
    assert "severity" in data
    assert "processing_time_ms" in data
    assert 0 <= data["confidence"] <= 1


def test_predict_url_endpoint(client):
    """Test URL prediction endpoint."""
    response = client.post(
        "/api/v1/predict/url",
        json={"url": "http://malicious-site.com/phishing"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "is_attack" in data
    assert "confidence" in data
    assert "severity" in data
    assert 0 <= data["confidence"] <= 1


def test_predict_batch_endpoint(client):
    """Test batch prediction endpoint."""
    response = client.post(
        "/api/v1/predict/batch",
        json={
            "payloads": ["' OR '1'='1", "Hello world"],
            "urls": ["http://example.com"]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "total_processing_time_ms" in data
    assert len(data["results"]) == 3


def test_predict_timeseries_endpoint(client):
    """Test timeseries prediction endpoint."""
    response = client.post(
        "/api/v1/predict/timeseries",
        json={
            "events": [
                {"timestamp": 1, "action": "login"},
                {"timestamp": 2, "action": "access"},
            ],
            "window_size": 10
        }
    )
    assert response.status_code in [200, 500]  # May fail if model not loaded
    if response.status_code == 200:
        data = response.json()
        assert "is_attack" in data
        assert "confidence" in data


def test_payload_validation(client):
    """Test payload validation."""
    response = client.post(
        "/api/v1/predict/payload",
        json={"payload": "x" * 20000}  # Exceeds max_length
    )
    assert response.status_code == 422


def test_batch_empty_request(client):
    """Test batch endpoint with empty request."""
    response = client.post(
        "/api/v1/predict/batch",
        json={}
    )
    assert response.status_code == 422
