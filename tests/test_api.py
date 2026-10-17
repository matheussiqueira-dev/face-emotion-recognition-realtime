from tests.conftest import ADMIN, READ


def test_health_endpoint_is_public(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_responses_carry_request_id_and_security_headers(client):
    response = client.get("/api/v1/health", headers={"X-Request-ID": "trace-123"})
    assert response.headers["x-request-id"] == "trace-123"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_malformed_request_id_is_replaced(client):
    response = client.get("/api/v1/health", headers={"X-Request-ID": "bad id\twith spaces"})
    assert response.headers["x-request-id"] != "bad id\twith spaces"
    assert len(response.headers["x-request-id"]) == 32


def test_docs_page_skips_csp_but_keeps_other_headers(client):
    response = client.get("/api/v1/docs")
    assert response.status_code == 200
    assert "content-security-policy" not in response.headers
    assert response.headers["x-content-type-options"] == "nosniff"


def test_errors_use_uniform_envelope(client):
    response = client.get("/api/v1/config", headers={"X-Request-ID": "req-1"})
    assert response.status_code == 401
    assert response.json() == {
        "error": {"code": "unauthorized", "message": "Missing or invalid API token.", "request_id": "req-1"}
    }


def test_config_requires_read_permission(client):
    assert client.get("/api/v1/config").status_code == 401
    assert client.get("/api/v1/config", headers={"Authorization": "Bearer wrong"}).status_code == 401

    authorized = client.get("/api/v1/config", headers=READ)
    assert authorized.status_code == 200
    assert "config" in authorized.json()


def test_config_snapshot_masks_tokens(client):
    config = client.get("/api/v1/config", headers=READ).json()["config"]
    assert config["api_read_token"] == "***"
    assert config["api_admin_token"] == "***"


def test_patch_config_with_read_token_is_forbidden_not_unauthorized(client):
    response = client.patch("/api/v1/config", headers=READ, json={"max_fps": 24})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"


def test_patch_config_as_admin(client):
    response = client.patch("/api/v1/config", headers=ADMIN, json={"max_fps": 24})
    assert response.status_code == 200
    payload = response.json()
    assert "max_fps" in payload["updated_fields"]
    assert payload["config"]["max_fps"] == 24


def test_patch_config_rejects_unknown_fields(client):
    response = client.patch("/api/v1/config", headers=ADMIN, json={"max_fsp": 24})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"][0]["field"] == "max_fsp"


def test_invalid_patch_leaves_config_untouched(client, config):
    response = client.patch("/api/v1/config", headers=ADMIN, json={"max_fps": 24, "jpeg_quality": 0})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_request"
    assert config.max_fps == 30
    assert config.jpeg_quality == 80


def test_video_source_change_requires_configured_auth(open_client):
    response = open_client.patch("/api/v1/config", json={"video_source": "/etc/passwd"})
    assert response.status_code == 403

    # Other settings stay editable in open (demo) mode.
    assert open_client.patch("/api/v1/config", json={"max_fps": 20}).status_code == 200


def test_video_source_change_allowed_for_admin_when_auth_configured(client):
    response = client.patch("/api/v1/config", headers=ADMIN, json={"video_source": "other.mp4"})
    assert response.status_code == 200
    assert response.json()["config"]["video_source"] == "other.mp4"


def test_metrics_requires_read_permission(client):
    assert client.get("/api/v1/metrics").status_code == 401
    authorized = client.get("/api/v1/metrics", headers=READ)
    assert authorized.status_code == 200
    assert "frames_processed" in authorized.json()


def test_prometheus_metrics(client):
    response = client.get("/api/v1/metrics/prometheus", headers=READ)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "# TYPE emotionai_frames_processed_total counter" in response.text
    assert "emotionai_frames_processed_total 0" in response.text


def test_cors_does_not_allow_credentials(client):
    response = client.options(
        "/api/v1/health",
        headers={"Origin": "https://example.com", "Access-Control-Request-Method": "GET"},
    )
    assert "access-control-allow-credentials" not in response.headers
