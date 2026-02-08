from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from datetime import timezone
from pathlib import Path

from fastapi import Depends
from fastapi import FastAPI
from fastapi import Header
from fastapi import HTTPException
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.backend.schemas import ConfigResponseSchema
from app.backend.schemas import ConfigUpdateRequestSchema
from app.backend.schemas import HealthResponseSchema
from app.backend.schemas import MetricsResponseSchema
from app.backend.schemas import StreamEnvelopeSchema
from app.backend.schemas import UpdateResultSchema
from app.backend.security import ensure_http_permission
from app.backend.security import ensure_websocket_permission
from app.backend.services import StreamService
from app.core.config import AppConfig


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

frontend_path = Path(__file__).resolve().parent.parent / "frontend"
runtime_config = AppConfig()
stream_service = StreamService(runtime_config)


def _model_dump(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def _require_read_permission(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    ensure_http_permission(runtime_config, "read", authorization, x_api_key)


def _require_admin_permission(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    ensure_http_permission(runtime_config, "admin", authorization, x_api_key)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await stream_service.shutdown()


app = FastAPI(
    title="EmotionAI API",
    version="2.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=runtime_config.cors_origins,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_index() -> HTMLResponse:
    with open(frontend_path / "index.html", "r", encoding="utf-8") as file:
        return HTMLResponse(content=file.read())


@app.get("/api/v1/health", response_model=HealthResponseSchema)
async def health_check() -> HealthResponseSchema:
    return HealthResponseSchema(timestamp_utc=datetime.now(timezone.utc).isoformat())


@app.get(
    "/api/v1/config",
    response_model=ConfigResponseSchema,
    dependencies=[Depends(_require_read_permission)],
)
async def get_config() -> ConfigResponseSchema:
    return ConfigResponseSchema(config=stream_service.get_config_snapshot())


@app.patch(
    "/api/v1/config",
    response_model=UpdateResultSchema,
    dependencies=[Depends(_require_admin_permission)],
)
async def patch_config(payload: ConfigUpdateRequestSchema) -> UpdateResultSchema:
    updates = _model_dump(payload)
    try:
        updated = await stream_service.update_config(updates)
    except ValueError as config_error:
        raise HTTPException(status_code=400, detail=str(config_error)) from config_error

    return UpdateResultSchema(
        updated_fields=updated,
        config=stream_service.get_config_snapshot(),
    )


@app.get(
    "/api/v1/metrics",
    response_model=MetricsResponseSchema,
    dependencies=[Depends(_require_read_permission)],
)
async def get_metrics() -> MetricsResponseSchema:
    return MetricsResponseSchema(**stream_service.get_metrics_snapshot())


@app.websocket("/api/v1/ws/stream")
async def websocket_stream_v1(websocket: WebSocket):
    await _handle_stream_connection(websocket)


@app.websocket("/ws/stream")
async def websocket_stream_legacy(websocket: WebSocket):
    await _handle_stream_connection(websocket)


async def _handle_stream_connection(websocket: WebSocket) -> None:
    is_authorized = await ensure_websocket_permission(
        websocket=websocket,
        config=runtime_config,
        permission="read",
    )
    if not is_authorized:
        return

    await stream_service.connect(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                payload = {"type": "raw", "value": message}

            if payload.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        LOGGER.info("WebSocket client disconnected.")
    except Exception:
        LOGGER.exception("Unexpected websocket error.")
    finally:
        await stream_service.disconnect(websocket)
