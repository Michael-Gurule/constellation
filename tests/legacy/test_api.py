"""
Unit tests for API endpoints.
"""

import pytest
from api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "models_loaded" in data


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "degraded", "unavailable"]


def test_models_status():
    """Test models status endpoint."""
    response = client.get("/api/v1/models/status")
    assert response.status_code == 200

    data = response.json()
    assert "models" in data
    assert "isolation_forest" in data["models"]


def test_invalid_endpoint():
    """Test invalid endpoint returns 404."""
    response = client.get("/api/v1/invalid")
    assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
