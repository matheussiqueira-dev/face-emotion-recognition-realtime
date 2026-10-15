"""Uniform error contract: every failure is returned as {"error": {code, message, request_id, details?}}."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


LOGGER = logging.getLogger(__name__)

_STATUS_CODES: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    422: "validation_error",
    429: "too_many_requests",
    500: "internal_error",
    503: "service_unavailable",
}


class ApiError(Exception):
    """Expected, client-facing failure raised by routes and dependencies."""

    def __init__(self, status_code: int, message: str, code: str | None = None, details: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code or _STATUS_CODES.get(status_code, "error")
        self.message = message
        self.details = details


def error_body(code: str, message: str, request_id: str | None, details: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message, "request_id": request_id}
    if details is not None:
        error["details"] = details
    return {"error": error}


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


async def _handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
    body = error_body(exc.code, exc.message, _request_id(request), exc.details)
    return JSONResponse(status_code=exc.status_code, content=body)


async def _handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = _STATUS_CODES.get(exc.status_code, "error")
    message = exc.detail if isinstance(exc.detail, str) else code.replace("_", " ")
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(code, message, _request_id(request)),
        headers=getattr(exc, "headers", None),
    )


async def _handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    details = [
        {"field": ".".join(str(part) for part in item.get("loc", ()) if part != "body"), "message": item.get("msg")}
        for item in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content=error_body("validation_error", "Request validation failed.", _request_id(request), details),
    )


async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    request_id = _request_id(request)
    # Log the full traceback server-side; never leak internals to the client.
    LOGGER.exception("Unhandled error (request_id=%s)", request_id)
    return JSONResponse(
        status_code=500,
        content=error_body("internal_error", "Unexpected server error.", request_id),
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApiError, _handle_api_error)
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(Exception, _handle_unexpected_error)
