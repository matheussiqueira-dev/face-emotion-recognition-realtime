from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app import __version__


class FaceBoxSchema(BaseModel):
    x: int
    y: int
    w: int
    h: int


class EmotionSchema(BaseModel):
    dominant: str = Field(description="Label with the highest smoothed score (see emotion_smoothing).")
    scores: dict[str, float] = Field(description="Per-label scores in percent, smoothed across inferences.")
    raw_dominant: str = Field(description="Label returned by the model for the latest inference.")


class TrackSchema(BaseModel):
    id: int
    box: FaceBoxSchema
    emotion: EmotionSchema | None = None


class StreamFrameDataSchema(BaseModel):
    fps: float
    tracks: list[TrackSchema]
    analysis_enabled: bool


class StreamEnvelopeSchema(BaseModel):
    frame: str = Field(..., description="JPEG frame encoded as base64 string.")
    data: StreamFrameDataSchema


class HealthResponseSchema(BaseModel):
    status: str = "ok"
    timestamp_utc: str
    version: str = __version__


class ErrorDetailSchema(BaseModel):
    code: str = Field(..., examples=["unauthorized"])
    message: str
    request_id: str | None = None
    details: Any | None = None


class ErrorResponseSchema(BaseModel):
    error: ErrorDetailSchema


class ConfigResponseSchema(BaseModel):
    config: dict[str, Any]


class ConfigUpdateRequestSchema(BaseModel):
    # Reject unknown fields: silently dropping a typo would report success without changing anything.
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    video_source: str | None = None
    loop_video: bool | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    detect_scale: float | None = None
    scale_factor: float | None = None
    min_neighbors: int | None = None
    min_face_size: int | None = None
    emotion_interval: float | None = None
    min_emotion_score: float | None = None
    enable_emotion_analysis: bool | None = None
    emotion_smoothing: float | None = None
    track_ttl: float | None = None
    iou_threshold: float | None = None
    max_fps: int | None = None
    jpeg_quality: int | None = None


class MetricsResponseSchema(BaseModel):
    started_at: str
    last_frame_at: str | None
    uptime_seconds: float
    frames_processed: int
    detections_total: int
    analyzed_faces: int
    expression_observations: int = Field(
        description="Classified frame observations; not unique people or inference calls."
    )
    latest_tracks: int
    peak_tracks: int
    current_fps: float
    average_fps: float
    dominant_distribution: dict[str, float]
    sentiment_history: list[float]


class UpdateResultSchema(BaseModel):
    updated_fields: list[str]
    config: dict[str, Any]


class RuntimeStatusSchema(BaseModel):
    state: Literal["idle", "starting", "running", "stopping", "degraded", "failed"]
    error_code: str | None
    clients: int
