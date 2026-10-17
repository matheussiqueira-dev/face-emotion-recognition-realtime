"""FastAPI dependencies resolving collaborators from app.state, so tests and deployments inject their own."""

from __future__ import annotations

from fastapi import Header, Request

from app.backend.analysis import BrowserAnalysisService
from app.backend.errors import ApiError
from app.backend.security import Permission, authorize, extract_token, origin_allowed
from app.backend.services import StreamService
from app.core.config import AppConfig


def get_config(request: Request) -> AppConfig:
    return request.app.state.config


def get_stream_service(request: Request) -> StreamService:
    return request.app.state.stream_service


def get_analysis_service(request: Request) -> BrowserAnalysisService:
    return request.app.state.analysis_service


def _authorize_request(
    request: Request, permission: Permission, authorization: str | None, x_api_key: str | None
) -> None:
    if not origin_allowed(
        request.headers.get("origin"),
        request.headers.get("host", ""),
        request.url.scheme,
        get_config(request),
    ):
        raise ApiError(403, "Origin is not allowed.")
    authorize(get_config(request), extract_token(authorization, x_api_key), permission)


def require_read(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    _authorize_request(request, Permission.READ, authorization, x_api_key)


def require_admin(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    _authorize_request(request, Permission.ADMIN, authorization, x_api_key)
