import pytest
from starlette.websockets import WebSocketDisconnect

from app.backend.analysis import BrowserAnalysisService
from app.backend.api import create_app
from app.backend.errors import ApiError
from app.backend.security import Permission, authorize
from tests.conftest import READ, fake_pipeline_factory, make_config


def test_read_only_installation_does_not_grant_admin():
    config = make_config(api_admin_token=None)
    with pytest.raises(ApiError) as denied:
        authorize(config, "read-token", Permission.ADMIN)
    assert denied.value.status_code == 403


def test_admin_only_installation_opens_reading_but_not_changes():
    config = make_config(api_read_token=None)
    assert authorize(config, None, Permission.READ).value == "anonymous"
    with pytest.raises(ApiError) as denied:
        authorize(config, None, Permission.ADMIN)
    assert denied.value.status_code == 401
    with pytest.raises(ApiError) as invalid:
        authorize(config, "wrong", Permission.READ)
    assert invalid.value.status_code == 401


def test_read_token_keeps_anonymous_readers_out():
    with pytest.raises(ApiError) as denied:
        authorize(make_config(), None, Permission.READ)
    assert denied.value.status_code == 401


def test_admin_only_installation_accepts_anonymous_camera_session():
    from fastapi.testclient import TestClient

    config = make_config(api_read_token=None)
    app = create_app(config, analysis_service=BrowserAnalysisService(config, fake_pipeline_factory(config)))
    with TestClient(app) as public_client:
        with public_client.websocket_connect("/api/v1/ws/analyze") as ws:
            ws.send_json({"type": "auth", "token": ""})  # What the dashboard sends with an empty token field.
            assert ws.receive_json()["type"] == "ready"
        assert public_client.patch("/api/v1/config", json={"emotion_interval": 1.0}).status_code == 401


def test_websocket_first_message_authentication(client):
    with client.websocket_connect("/api/v1/ws/analyze") as ws:
        ws.send_json({"type": "auth", "token": "read-token"})
        assert ws.receive_json()["type"] == "ready"


@pytest.mark.parametrize(
    "payload", [{"type": "auth", "token": "wrong"}, [], {"type": "ping"}, {"type": "auth", "token": 42}]
)
def test_websocket_rejects_invalid_first_message(client, payload):
    with pytest.raises(WebSocketDisconnect) as denied:
        with client.websocket_connect("/api/v1/ws/analyze") as ws:
            ws.send_json(payload)
            ws.receive_json()
    assert denied.value.code == 4401


def test_websocket_rejects_query_credentials(client):
    with pytest.raises(WebSocketDisconnect) as denied:
        with client.websocket_connect("/api/v1/ws/analyze?token=read-token") as ws:
            ws.receive_json()
    assert denied.value.code == 4401


def test_foreign_browser_origin_is_denied_before_model_start(client):
    with pytest.raises(WebSocketDisconnect) as denied:
        with client.websocket_connect(
            "/api/v1/ws/analyze", headers={**READ, "Origin": "https://foreign.example"}
        ) as ws:
            ws.receive_json()
    assert denied.value.code == 4403


def test_same_origin_browser_can_authenticate(client):
    with client.websocket_connect(
        "/api/v1/ws/analyze", headers={**READ, "Origin": "http://testserver"}
    ) as ws:
        assert ws.receive_json()["type"] == "ready"


def test_foreign_http_origin_cannot_mutate_open_demo(open_client):
    response = open_client.patch(
        "/api/v1/config", headers={"Origin": "https://foreign.example"}, json={"max_fps": 10}
    )
    assert response.status_code == 403


@pytest.mark.parametrize("value", ["24", True, 24.5])
def test_config_rejects_implicit_numeric_coercion(client, value):
    assert (
        client.patch(
            "/api/v1/config", headers={"Authorization": "Bearer admin-token"}, json={"max_fps": value}
        ).status_code
        == 422
    )


def test_runtime_status_is_separate_from_liveness(client):
    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/api/v1/status").status_code == 401
    assert client.get("/api/v1/status", headers=READ).json() == {
        "state": "idle",
        "error_code": None,
        "clients": 0,
    }
