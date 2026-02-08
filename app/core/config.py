from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in TRUE_VALUES


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_csv(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return list(default)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or list(default)


def _env_token(name: str) -> str | None:
    return os.getenv(name) or os.getenv("EMOTION_API_TOKEN") or None


@dataclass(slots=True)
class AppConfig:
    """Application configuration with runtime validation."""

    # Video Input/Output
    video_source: str = field(default_factory=lambda: os.getenv("EMOTION_VIDEO_SOURCE", "0"))
    width: int | None = field(default_factory=lambda: _env_int("EMOTION_WIDTH", 1280))
    height: int | None = field(default_factory=lambda: _env_int("EMOTION_HEIGHT", 720))
    fps: float | None = field(default_factory=lambda: _env_float("EMOTION_CAPTURE_FPS", 30.0))
    output_path: Path = field(default_factory=lambda: Path(os.getenv("EMOTION_OUTPUT_PATH", "output/recordings")))
    record_session: bool = field(default_factory=lambda: _env_bool("EMOTION_RECORD_SESSION", True))

    # Detection Settings
    detect_scale: float = field(default_factory=lambda: _env_float("EMOTION_DETECT_SCALE", 0.5))
    scale_factor: float = field(default_factory=lambda: _env_float("EMOTION_SCALE_FACTOR", 1.1))
    min_neighbors: int = field(default_factory=lambda: _env_int("EMOTION_MIN_NEIGHBORS", 5))
    min_face_size: int = field(default_factory=lambda: _env_int("EMOTION_MIN_FACE_SIZE", 30))
    cascade_path: Path | None = field(default_factory=lambda: Path(os.getenv("EMOTION_CASCADE_PATH")) if os.getenv("EMOTION_CASCADE_PATH") else None)

    # Analysis Settings
    emotion_interval: float = field(default_factory=lambda: _env_float("EMOTION_INTERVAL", 0.5))
    min_emotion_score: float = field(default_factory=lambda: _env_float("EMOTION_MIN_SCORE", 0.0))
    enable_emotion_analysis: bool = field(default_factory=lambda: _env_bool("EMOTION_ENABLE_ANALYSIS", True))

    # Tracking Settings
    track_ttl: float = field(default_factory=lambda: _env_float("EMOTION_TRACK_TTL", 1.5))
    iou_threshold: float = field(default_factory=lambda: _env_float("EMOTION_IOU_THRESHOLD", 0.3))

    # Performance
    max_fps: int = field(default_factory=lambda: _env_int("EMOTION_MAX_FPS", 30))
    enable_gpu: bool = field(default_factory=lambda: _env_bool("EMOTION_ENABLE_GPU", True))
    jpeg_quality: int = field(default_factory=lambda: _env_int("EMOTION_JPEG_QUALITY", 80))

    # API & Security
    cors_origins: list[str] = field(default_factory=lambda: _env_csv("EMOTION_CORS_ORIGINS", ["*"]))
    api_read_token: str | None = field(default_factory=lambda: _env_token("EMOTION_API_READ_TOKEN"))
    api_admin_token: str | None = field(default_factory=lambda: _env_token("EMOTION_API_ADMIN_TOKEN"))

    # UI/Visuals
    draw_landmarks: bool = field(default_factory=lambda: _env_bool("EMOTION_DRAW_LANDMARKS", True))
    show_fps: bool = field(default_factory=lambda: _env_bool("EMOTION_SHOW_FPS", True))
    theme_color: str = field(default_factory=lambda: os.getenv("EMOTION_THEME_COLOR", "#0f766e"))

    def __post_init__(self) -> None:
        if self.width is not None and self.width <= 0:
            raise ValueError("width must be greater than zero.")
        if self.height is not None and self.height <= 0:
            raise ValueError("height must be greater than zero.")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("fps must be greater than zero.")
        if not 0.1 <= self.detect_scale <= 1.0:
            raise ValueError("detect_scale must be in range [0.1, 1.0].")
        if self.scale_factor < 1.01:
            raise ValueError("scale_factor must be >= 1.01.")
        if self.min_neighbors < 1:
            raise ValueError("min_neighbors must be >= 1.")
        if self.min_face_size < 10:
            raise ValueError("min_face_size must be >= 10.")
        if self.emotion_interval <= 0:
            raise ValueError("emotion_interval must be greater than zero.")
        if not 0.0 <= self.min_emotion_score <= 100.0:
            raise ValueError("min_emotion_score must be in range [0, 100].")
        if self.track_ttl <= 0:
            raise ValueError("track_ttl must be greater than zero.")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in range [0, 1].")
        if self.max_fps <= 0:
            raise ValueError("max_fps must be greater than zero.")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in range [1, 100].")
        if self.cascade_path is not None and not self.cascade_path.exists():
            raise ValueError(f"cascade_path does not exist: {self.cascade_path}")

    def as_dict(self, include_secrets: bool = False) -> dict[str, Any]:
        data = asdict(self)
        data["output_path"] = str(self.output_path)
        data["cascade_path"] = str(self.cascade_path) if self.cascade_path else None
        if not include_secrets:
            data["api_read_token"] = "***" if self.api_read_token else None
            data["api_admin_token"] = "***" if self.api_admin_token else None
        return data

    def update_from_dict(self, updates: dict[str, Any]) -> list[str]:
        valid_fields = {item.name for item in fields(self)}
        unknown_keys = sorted(key for key in updates if key not in valid_fields)
        if unknown_keys:
            keys = ", ".join(unknown_keys)
            raise ValueError(f"Unknown configuration fields: {keys}")

        changed: list[str] = []
        for key, value in updates.items():
            parsed = value
            if key in {"output_path", "cascade_path"} and value is not None:
                parsed = Path(value)
            elif key == "cors_origins" and isinstance(value, str):
                parsed = [item.strip() for item in value.split(",") if item.strip()]
            if getattr(self, key) != parsed:
                setattr(self, key, parsed)
                changed.append(key)

        self.__post_init__()
        return changed
