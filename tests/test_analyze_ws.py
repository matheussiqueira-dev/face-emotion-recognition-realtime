import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.backend.analysis import BrowserAnalysisService
from app.backend.api import create_app
from tests.conftest import fake_pipeline_factory, make_config

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def _jpeg() -> bytes:
    ok, buffer = cv2.imencode(".jpg", np.zeros((120, 160, 3), dtype=np.uint8))
    assert ok
    return buffer.tobytes()


def test_analyze_requires_token(client):
    with pytest.raises(WebSocketDisconnect) as closed:
        with client.websocket_connect("/api/v1/ws/analyze") as ws:
            ws.receive_json()
    assert closed.value.code == 4401


def test_analyze_returns_tracks_for_uploaded_frame(client):
    with client.websocket_connect("/api/v1/ws/analyze", headers={"Authorization": "Bearer read-token"}) as ws:
        assert ws.receive_json() == {"type": "ready", "analysis_enabled": True}
        ws.send_bytes(_jpeg())
        result = ws.receive_json()

    assert result["type"] == "analysis"
    assert result["latency_ms"] >= 0
    [track] = result["tracks"]
    assert track["box"] == {"x": 10, "y": 10, "w": 40, "h": 40}
    assert track["emotion"]["dominant"] == "happy"


def test_analyze_rejects_undecodable_and_oversized_frames(client):
    with client.websocket_connect("/api/v1/ws/analyze", headers={"Authorization": "Bearer read-token"}) as ws:
        ws.receive_json()
        ws.send_bytes(b"not a jpeg")
        assert ws.receive_json()["code"] == "frame_rejected"

        ws.send_bytes(b"\xff" * 2_000_001)
        assert ws.receive_json()["code"] == "frame_rejected"

        # The session survives rejected frames.
        ws.send_bytes(_jpeg())
        assert ws.receive_json()["type"] == "analysis"


def test_analyze_rejects_resolution_above_limit_before_decoding():
    # 160x120 = 19200 px declared in the header; the limit is checked without decoding pixels.
    config = make_config(max_frame_pixels=10_000)
    app = create_app(config, analysis_service=BrowserAnalysisService(config, fake_pipeline_factory(config)))
    with TestClient(app) as client:
        with client.websocket_connect(
            "/api/v1/ws/analyze", headers={"Authorization": "Bearer read-token"}
        ) as ws:
            ws.receive_json()
            ws.send_bytes(_jpeg())
            rejected = ws.receive_json()
    assert rejected["code"] == "frame_rejected"
    assert "160x120" in rejected["message"]


def test_analyze_accepts_only_jpeg(client):
    ok, png = cv2.imencode(".png", np.zeros((120, 160, 3), dtype=np.uint8))
    assert ok
    with client.websocket_connect("/api/v1/ws/analyze", headers={"Authorization": "Bearer read-token"}) as ws:
        ws.receive_json()
        ws.send_bytes(png.tobytes())
        rejected = ws.receive_json()
    assert rejected == {"type": "error", "code": "frame_rejected", "message": "Frame must be a JPEG image."}


def test_analyze_answers_ping(client):
    with client.websocket_connect("/api/v1/ws/analyze", headers={"Authorization": "Bearer read-token"}) as ws:
        ws.receive_json()
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}


def test_analyze_enforces_session_limit():
    config = make_config(max_analyze_sessions=1)
    app = create_app(config, analysis_service=BrowserAnalysisService(config, fake_pipeline_factory(config)))
    with TestClient(app) as client:
        with client.websocket_connect(
            "/api/v1/ws/analyze", headers={"Authorization": "Bearer read-token"}
        ) as first:
            assert first.receive_json() == {"type": "ready", "analysis_enabled": True}
            with pytest.raises(WebSocketDisconnect) as closed:
                with client.websocket_connect(
                    "/api/v1/ws/analyze", headers={"Authorization": "Bearer read-token"}
                ) as second:
                    second.receive_json()
            assert closed.value.code == 1013
