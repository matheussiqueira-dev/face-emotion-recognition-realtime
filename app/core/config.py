from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

TRUE_VALUES = {"1", "true", "yes", "on"}
FACE_DETECTORS = frozenset({"haar", "yunet"})


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


def _env_path(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value) if value else None


@dataclass(slots=True)
class AppConfig:
    """Application configuration with runtime validation."""

    # Video Input/Output
    video_source: str = field(default_factory=lambda: os.getenv("EMOTION_VIDEO_SOURCE", "0"))
    width: int | None = field(default_factory=lambda: _env_int("EMOTION_WIDTH", 1280))
    height: int | None = field(default_factory=lambda: _env_int("EMOTION_HEIGHT", 720))
    fps: float | None = field(default_factory=lambda: _env_float("EMOTION_CAPTURE_FPS", 30.0))
    # Restart file sources at EOF so demos keep running instead of freezing on the last frame.
    loop_video: bool = field(default_factory=lambda: _env_bool("EMOTION_LOOP_VIDEO", True))

    # Detection Settings
    detect_scale: float = field(default_factory=lambda: _env_float("EMOTION_DETECT_SCALE", 0.5))
    scale_factor: float = field(default_factory=lambda: _env_float("EMOTION_SCALE_FACTOR", 1.1))
    min_neighbors: int = field(default_factory=lambda: _env_int("EMOTION_MIN_NEIGHBORS", 5))
    min_face_size: int = field(default_factory=lambda: _env_int("EMOTION_MIN_FACE_SIZE", 30))
    cascade_path: Path | None = field(default_factory=lambda: _env_path("EMOTION_CASCADE_PATH"))
    # "haar" ships with OpenCV. "yunet" is a small CNN detector (opencv_zoo, MIT) that also finds tilted and
    # partially turned faces; its .onnx file is not bundled, so it needs yunet_model_path.
    face_detector: str = field(
        default_factory=lambda: os.getenv("EMOTION_FACE_DETECTOR", "haar").strip().lower()
    )
    yunet_model_path: Path | None = field(default_factory=lambda: _env_path("EMOTION_YUNET_MODEL"))

    # Analysis Settings
    emotion_interval: float = field(default_factory=lambda: _env_float("EMOTION_INTERVAL", 0.5))
    min_emotion_score: float = field(default_factory=lambda: _env_float("EMOTION_MIN_SCORE", 0.0))
    enable_emotion_analysis: bool = field(default_factory=lambda: _env_bool("EMOTION_ENABLE_ANALYSIS", True))
    # Exponential smoothing of per-track scores: weight kept from the previous estimate (0 = raw output).
    # Single-frame classifications flicker; a sustained change still wins within a few inferences.
    emotion_smoothing: float = field(default_factory=lambda: _env_float("EMOTION_SMOOTHING", 0.5))

    # Tracking Settings
    track_ttl: float = field(default_factory=lambda: _env_float("EMOTION_TRACK_TTL", 1.5))
    iou_threshold: float = field(default_factory=lambda: _env_float("EMOTION_IOU_THRESHOLD", 0.3))

    # Performance
    max_fps: int = field(default_factory=lambda: _env_int("EMOTION_MAX_FPS", 30))
    jpeg_quality: int = field(default_factory=lambda: _env_int("EMOTION_JPEG_QUALITY", 80))

    # Browser camera mode (frames uploaded by the client)
    max_frame_bytes: int = field(default_factory=lambda: _env_int("EMOTION_MAX_FRAME_BYTES", 2_000_000))
    # Checked from the JPEG header before decoding: a small file can declare a huge resolution.
    max_frame_pixels: int = field(default_factory=lambda: _env_int("EMOTION_MAX_FRAME_PIXELS", 3840 * 2160))
    max_analyze_sessions: int = field(default_factory=lambda: _env_int("EMOTION_MAX_ANALYZE_SESSIONS", 4))
    max_stream_clients: int = field(default_factory=lambda: _env_int("EMOTION_MAX_STREAM_CLIENTS", 16))

    # API & Security
    cors_origins: list[str] = field(default_factory=lambda: _env_csv("EMOTION_CORS_ORIGINS", []))
    api_read_token: str | None = field(default_factory=lambda: _env_token("EMOTION_API_READ_TOKEN"))
    api_admin_token: str | None = field(default_factory=lambda: _env_token("EMOTION_API_ADMIN_TOKEN"))

    def __post_init__(self) -> None:
        for name in (
            "fps",
            "detect_scale",
            "scale_factor",
            "emotion_interval",
            "min_emotion_score",
            "emotion_smoothing",
            "track_ttl",
            "iou_threshold",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
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
        if not 0.0 <= self.emotion_smoothing < 1.0:
            raise ValueError("emotion_smoothing must be in range [0, 1).")
        if self.track_ttl <= 0:
            raise ValueError("track_ttl must be greater than zero.")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in range [0, 1].")
        if self.max_fps <= 0:
            raise ValueError("max_fps must be greater than zero.")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in range [1, 100].")
        if self.max_frame_bytes < 10_000:
            raise ValueError("max_frame_bytes must be >= 10000.")
        if self.max_frame_pixels < 64 * 64:
            raise ValueError("max_frame_pixels must be >= 4096.")
        if self.max_analyze_sessions < 1:
            raise ValueError("max_analyze_sessions must be >= 1.")
        if self.max_stream_clients < 1:
            raise ValueError("max_stream_clients must be >= 1.")
        if self.face_detector not in FACE_DETECTORS:
            raise ValueError(f"face_detector must be one of: {', '.join(sorted(FACE_DETECTORS))}.")
        if self.face_detector == "yunet" and (
            self.yunet_model_path is None or not self.yunet_model_path.exists()
        ):
            raise ValueError(
                "face_detector 'yunet' requires yunet_model_path pointing to an existing .onnx file."
            )
        if self.cascade_path is not None and not self.cascade_path.exists():
            raise ValueError(f"cascade_path does not exist: {self.cascade_path}")

    def as_dict(self, include_secrets: bool = False) -> dict[str, Any]:
        data = asdict(self)
        data["cascade_path"] = str(self.cascade_path) if self.cascade_path else None
        data["yunet_model_path"] = str(self.yunet_model_path) if self.yunet_model_path else None
        if not include_secrets:
            data["api_read_token"] = "***" if self.api_read_token else None
            data["api_admin_token"] = "***" if self.api_admin_token else None
        return data

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_read_token or self.api_admin_token)

    def update_from_dict(self, updates: dict[str, Any]) -> list[str]:
        """Apply updates atomically: validate on a copy so a rejected update leaves this config untouched."""
        valid_fields = {item.name for item in fields(self)}
        unknown_keys = sorted(key for key in updates if key not in valid_fields)
        if unknown_keys:
            keys = ", ".join(unknown_keys)
            raise ValueError(f"Unknown configuration fields: {keys}")

        parsed_updates: dict[str, Any] = {}
        for key, value in updates.items():
            parsed = value
            if key in ("cascade_path", "yunet_model_path") and value is not None:
                parsed = Path(value)
            elif key == "cors_origins" and isinstance(value, str):
                parsed = [item.strip() for item in value.split(",") if item.strip()]
            parsed_updates[key] = parsed

        candidate = replace(self, **parsed_updates)  # runs __post_init__ validation

        changed: list[str] = []
        for key, value in parsed_updates.items():
            if getattr(self, key) != value:
                setattr(self, key, getattr(candidate, key))
                changed.append(key)
        return changed
