from __future__ import annotations

import asyncio
import hmac
import json
from enum import Enum
from urllib.parse import urlsplit

from fastapi import WebSocket

from app.backend.errors import ApiError
from app.core.config import AppConfig


class Permission(str, Enum):
    READ = "read"
    ADMIN = "admin"


class Role(str, Enum):
    ANONYMOUS = "anonymous"
    READER = "reader"
    ADMIN = "admin"


WS_CLOSE_UNAUTHORIZED = 4401
WS_CLOSE_FORBIDDEN = 4403


def extract_token(authorization_header: str | None, x_api_key: str | None) -> str | None:
    if x_api_key:
        return x_api_key.strip() or None

    if not authorization_header:
        return None

    value = authorization_header.strip()
    if value.lower().startswith("bearer "):
        return value[7:].strip() or None
    return value or None


def _matches(candidate: str, expected: str | None) -> bool:
    # Constant-time comparison so response timing does not reveal how much of a token matched.
    return bool(expected) and hmac.compare_digest(candidate.encode(), expected.encode())


def resolve_role(config: AppConfig, token: str | None) -> Role | None:
    """Map a token to a role. Returns None for a presented-but-invalid token."""
    if not config.auth_enabled:
        return Role.ADMIN  # Auth disabled (local/demo mode): everything is allowed.
    if not token:
        return Role.ANONYMOUS

    if _matches(token, config.api_admin_token):
        return Role.ADMIN
    if _matches(token, config.api_read_token):
        return Role.READER
    return None


def authorize(config: AppConfig, token: str | None, permission: Permission) -> Role:
    """Raise 401 when the caller is not authenticated, 403 when authenticated without the permission."""
    role = resolve_role(config, token)
    if role is None or role is Role.ANONYMOUS:
        raise ApiError(401, "Missing or invalid API token.")
    if permission is Permission.ADMIN and role is not Role.ADMIN:
        raise ApiError(403, "This operation requires the admin token.")
    return role


async def authorize_websocket(websocket: WebSocket, config: AppConfig, permission: Permission) -> bool:
    """Authenticate before allocating a model. Browser credentials travel in the first message."""
    if not origin_allowed(
        websocket.headers.get("origin"), websocket.headers.get("host", ""), websocket.url.scheme, config
    ):
        await websocket.close(code=WS_CLOSE_FORBIDDEN, reason="Origin is not allowed.")
        return False
    if "token" in websocket.query_params:
        await websocket.close(
            code=WS_CLOSE_UNAUTHORIZED, reason="Use headers or the first authentication message."
        )
        return False
    token = extract_token(
        websocket.headers.get("authorization"),
        websocket.headers.get("x-api-key"),
    )

    # Accept the transport only; no service/model is created until authorization succeeds.
    await websocket.accept()
    if config.auth_enabled and not token:
        try:
            message = await asyncio.wait_for(websocket.receive(), timeout=5.0)
            text = message.get("text", "")
            if not isinstance(text, str) or len(text) > 4096:
                raise ValueError("Invalid authentication message")
            payload = json.loads(text)
            if not isinstance(payload, dict) or payload.get("type") != "auth":
                raise ValueError("Invalid authentication message")
            token = payload.get("token")
            if not isinstance(token, str):
                raise ValueError("Invalid token")
        except (TimeoutError, ValueError):
            await websocket.close(
                code=WS_CLOSE_UNAUTHORIZED, reason="Missing or invalid authentication message."
            )
            return False

    try:
        authorize(config, token, permission)
    except ApiError as error:
        code = WS_CLOSE_FORBIDDEN if error.status_code == 403 else WS_CLOSE_UNAUTHORIZED
        await websocket.close(code=code, reason=error.message)
        return False
    return True


def origin_allowed(origin: str | None, host: str, scheme: str, config: AppConfig) -> bool:
    """No Origin is valid for non-browser clients; browsers need same origin or an explicit allowlist."""
    if origin is None:
        return True
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    expected_scheme = "https" if scheme in {"https", "wss"} else "http"
    same_origin = parsed.scheme == expected_scheme and parsed.netloc.lower() == host.lower()
    return same_origin or origin in [item for item in config.cors_origins if item != "*"]
