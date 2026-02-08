import pytest
from fastapi.testclient import TestClient

from app.backend import api as api_module


@pytest.fixture()
def client():
    previous_read = api_module.runtime_config.api_read_token
    previous_admin = api_module.runtime_config.api_admin_token
    previous_max_fps = api_module.runtime_config.max_fps

    api_module.runtime_config.api_read_token = "read-token"
    api_module.runtime_config.api_admin_token = "admin-token"

    with TestClient(api_module.app) as test_client:
        yield test_client

    api_module.runtime_config.api_read_token = previous_read
    api_module.runtime_config.api_admin_token = previous_admin
    api_module.runtime_config.max_fps = previous_max_fps


def test_health_endpoint_is_public(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_config_requires_read_permission(client):
    unauthorized = client.get("/api/v1/config")
    assert unauthorized.status_code == 401

    authorized = client.get("/api/v1/config", headers={"Authorization": "Bearer read-token"})
    assert authorized.status_code == 200
    assert "config" in authorized.json()


def test_patch_config_requires_admin_permission(client):
    response_read = client.patch(
        "/api/v1/config",
        headers={"Authorization": "Bearer read-token"},
        json={"max_fps": 24},
    )
    assert response_read.status_code == 401

    response_admin = client.patch(
        "/api/v1/config",
        headers={"Authorization": "Bearer admin-token"},
        json={"max_fps": 24},
    )
    assert response_admin.status_code == 200
    payload = response_admin.json()
    assert "max_fps" in payload["updated_fields"]
    assert payload["config"]["max_fps"] == 24


def test_metrics_requires_read_permission(client):
    unauthorized = client.get("/api/v1/metrics")
    assert unauthorized.status_code == 401

    authorized = client.get("/api/v1/metrics", headers={"Authorization": "Bearer read-token"})
    assert authorized.status_code == 200
    data = authorized.json()
    assert "frames_processed" in data
