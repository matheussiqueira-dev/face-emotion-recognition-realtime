from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import Field


class FaceBoxSchema(BaseModel):
    x: int
    y: int
    w: int
    h: int


class EmotionSchema(BaseModel):
    dominant: str
    scores: dict[str, float]


class TrackSchema(BaseModel):
    id: int
    box: FaceBoxSchema
    emotion: EmotionSchema | None = None


class StreamFrameDataSchema(BaseModel):
    fps: float
    tracks: list[TrackSchema]


class StreamEnvelopeSchema(BaseModel):
    frame: str = Field(..., description="JPEG frame encoded as base64 string.")
    data: StreamFrameDataSchema


class HealthResponseSchema(BaseModel):
    status: str = "ok"
    timestamp_utc: str
    version: str = "2.0.0"


class ConfigResponseSchema(BaseModel):
    config: dict[str, Any]


class ConfigUpdateRequestSchema(BaseModel):
    video_source: str | None = None
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
    track_ttl: float | None = None
    iou_threshold: float | None = None
    max_fps: int | None = None
    jpeg_quality: int | None = None
    show_fps: bool | None = None
    theme_color: str | None = None


class MetricsResponseSchema(BaseModel):
    started_at: str
    last_frame_at: str | None
    uptime_seconds: float
    frames_processed: int
    detections_total: int
    analyzed_faces: int
    latest_tracks: int
    peak_tracks: int
    current_fps: float
    average_fps: float
    dominant_distribution: dict[str, float]
    sentiment_history: list[float]


class UpdateResultSchema(BaseModel):
    updated_fields: list[str]
    config: dict[str, Any]
