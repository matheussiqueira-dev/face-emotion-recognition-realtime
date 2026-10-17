from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from app.backend.dependencies import get_config, get_stream_service, require_admin, require_read
from app.backend.errors import ApiError
from app.backend.schemas import (
    ConfigResponseSchema,
    ConfigUpdateRequestSchema,
    ErrorResponseSchema,
    HealthResponseSchema,
    MetricsResponseSchema,
    RuntimeStatusSchema,
    UpdateResultSchema,
)
from app.backend.services import ProcessingBusy, StreamService
from app.core.config import AppConfig

# Fields that make the server open arbitrary files/URLs: only changeable when real auth is configured.
SOURCE_FIELDS = frozenset({"video_source"})

ERROR_RESPONSES = {
    400: {"model": ErrorResponseSchema},
    401: {"model": ErrorResponseSchema},
    403: {"model": ErrorResponseSchema},
    422: {"model": ErrorResponseSchema},
}

router = APIRouter(prefix="/api/v1", responses=ERROR_RESPONSES)


@router.get("/health", response_model=HealthResponseSchema, tags=["system"])
async def health_check() -> HealthResponseSchema:
    return HealthResponseSchema(timestamp_utc=datetime.now(timezone.utc).isoformat())


@router.get(
    "/config",
    response_model=ConfigResponseSchema,
    dependencies=[Depends(require_read)],
    tags=["config"],
)
async def get_runtime_config(service: StreamService = Depends(get_stream_service)) -> ConfigResponseSchema:
    return ConfigResponseSchema(config=service.get_config_snapshot())


@router.get(
    "/status", response_model=RuntimeStatusSchema, dependencies=[Depends(require_read)], tags=["system"]
)
async def get_runtime_status(service: StreamService = Depends(get_stream_service)) -> RuntimeStatusSchema:
    """Operational status of the shared server-video worker; health is HTTP liveness only."""
    return RuntimeStatusSchema(**service.get_status_snapshot())


@router.patch(
    "/config",
    response_model=UpdateResultSchema,
    dependencies=[Depends(require_admin)],
    tags=["config"],
)
async def patch_runtime_config(
    payload: ConfigUpdateRequestSchema,
    service: StreamService = Depends(get_stream_service),
    config: AppConfig = Depends(get_config),
) -> UpdateResultSchema:
    updates = payload.model_dump(exclude_none=True)
    if not config.auth_enabled and SOURCE_FIELDS.intersection(updates):
        raise ApiError(
            403,
            "Changing the video source requires API tokens (EMOTION_API_ADMIN_TOKEN) to be configured.",
        )
    try:
        updated = await service.update_config(updates)
    except ValueError as config_error:
        raise ApiError(400, str(config_error)) from config_error
    except ProcessingBusy as busy:
        raise ApiError(409, str(busy)) from busy

    return UpdateResultSchema(updated_fields=updated, config=service.get_config_snapshot())


@router.get(
    "/metrics",
    response_model=MetricsResponseSchema,
    dependencies=[Depends(require_read)],
    tags=["metrics"],
)
async def get_session_metrics(service: StreamService = Depends(get_stream_service)) -> MetricsResponseSchema:
    return MetricsResponseSchema(**service.get_metrics_snapshot())


@router.get(
    "/metrics/prometheus",
    response_class=PlainTextResponse,
    dependencies=[Depends(require_read)],
    tags=["metrics"],
)
async def get_prometheus_metrics(service: StreamService = Depends(get_stream_service)) -> PlainTextResponse:
    return PlainTextResponse(
        render_prometheus(service.get_metrics_snapshot()),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


def render_prometheus(snapshot: dict) -> str:
    """Prometheus text exposition of the session metrics (no client library needed for a handful of series)."""
    lines: list[str] = []

    def metric(name: str, kind: str, help_text: str, samples: list[tuple[str, float]]) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {kind}")
        for labels, value in samples:
            lines.append(f"{name}{labels} {value}")

    metric(
        "emotionai_frames_processed_total",
        "counter",
        "Frames processed in the current session.",
        [("", snapshot["frames_processed"])],
    )
    metric(
        "emotionai_face_detections_total",
        "counter",
        "Face detections in the current session.",
        [("", snapshot["detections_total"])],
    )
    metric(
        "emotionai_faces_analyzed_total",
        "counter",
        "Classified frame observations; not unique people or inference calls.",
        [("", snapshot["analyzed_faces"])],
    )
    metric(
        "emotionai_expression_observations_total",
        "counter",
        "Classified frame observations in the current session.",
        [("", snapshot["expression_observations"])],
    )
    metric(
        "emotionai_active_tracks",
        "gauge",
        "Faces tracked in the latest frame.",
        [("", snapshot["latest_tracks"])],
    )
    metric("emotionai_fps", "gauge", "Current processing frames per second.", [("", snapshot["current_fps"])])
    metric(
        "emotionai_session_uptime_seconds",
        "gauge",
        "Seconds since the session started.",
        [("", snapshot["uptime_seconds"])],
    )
    history = snapshot.get("sentiment_history") or [0.0]
    metric("emotionai_sentiment", "gauge", "Latest average sentiment in [-1, 1].", [("", history[-1])])
    metric(
        "emotionai_emotion_share_percent",
        "gauge",
        "Share of each dominant emotion in the session.",
        [(f'{{emotion="{label}"}}', share) for label, share in snapshot["dominant_distribution"].items()],
    )
    return "\n".join(lines) + "\n"
