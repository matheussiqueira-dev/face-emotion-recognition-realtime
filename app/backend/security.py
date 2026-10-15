from __future__ import annotations

import hmac
from enum import Enum

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

    admin_token = config.api_admin_token or config.api_read_token
    if _matches(token, admin_token):
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
    """Browsers cannot set headers on websockets, so the token may also come from the ?token= query param."""
    token = extract_token(
        websocket.headers.get("authorization"),
        websocket.headers.get("x-api-key"),
    ) or websocket.query_params.get("token")

    try:
        authorize(config, token, permission)
    except ApiError as error:
        code = WS_CLOSE_FORBIDDEN if error.status_code == 403 else WS_CLOSE_UNAUTHORIZED
        await websocket.close(code=code, reason=error.message)
        return False
    return True
