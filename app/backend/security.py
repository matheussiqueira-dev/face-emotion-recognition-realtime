from __future__ import annotations

from fastapi import HTTPException
from fastapi import status
from fastapi import WebSocket

from app.core.config import AppConfig


def extract_token(authorization_header: str | None, x_api_key: str | None) -> str | None:
    if x_api_key:
        return x_api_key.strip()

    if not authorization_header:
        return None

    value = authorization_header.strip()
    if value.lower().startswith("bearer "):
        return value[7:].strip() or None
    return value or None


def _is_auth_enabled(config: AppConfig) -> bool:
    return bool(config.api_read_token or config.api_admin_token)


def _has_permission(config: AppConfig, token: str | None, permission: str) -> bool:
    if not _is_auth_enabled(config):
        return True

    if not token:
        return False

    admin_token = config.api_admin_token or config.api_read_token
    read_token = config.api_read_token or admin_token
    allowed_read = {item for item in (read_token, admin_token) if item}

    if permission == "admin":
        return bool(admin_token and token == admin_token)
    return token in allowed_read


def ensure_http_permission(
    config: AppConfig,
    permission: str,
    authorization_header: str | None,
    x_api_key: str | None,
) -> None:
    token = extract_token(authorization_header, x_api_key)
    if _has_permission(config, token, permission):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized request.",
    )


async def ensure_websocket_permission(
    websocket: WebSocket,
    config: AppConfig,
    permission: str = "read",
) -> bool:
    token = extract_token(
        websocket.headers.get("authorization"),
        websocket.headers.get("x-api-key"),
    ) or websocket.query_params.get("token")

    if _has_permission(config, token, permission):
        return True

    await websocket.close(code=4401, reason="Unauthorized")
    return False
