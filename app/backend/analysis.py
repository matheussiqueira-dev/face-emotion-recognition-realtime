"""Browser-camera mode: the client uploads JPEG frames and receives detections back.

Frames never touch disk and are discarded after analysis, which keeps the demo usable with the viewer's own
webcam (no device passthrough to the server) and limits what the server retains.
"""

from __future__ import annotations

import asyncio
import io
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any, AsyncIterator

import numpy as np

from app.backend.serializers import serialize_track
from app.core.config import AppConfig
from app.core.pipeline import FramePipeline

PipelineFactory = Callable[[], FramePipeline]


class SessionLimitReached(Exception):
    pass


class FrameRejected(ValueError):
    pass


JPEG_MAGIC = bytes.fromhex("ffd8ff")  # SOI marker followed by the next segment marker


def check_jpeg_header(data: bytes, max_pixels: int) -> None:
    """Validate format and resolution from the header alone, before any pixel buffer is allocated.

    Size limits on the payload are not enough: a ~1.5 MB JPEG can declare 10000x10000 pixels and
    decode into ~300 MB. cv2.imdecode would also accept any format it supports, not just JPEG.
    """
    if not data.startswith(JPEG_MAGIC):
        raise FrameRejected("Frame must be a JPEG image.")

    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as image:  # Lazy: parses the header only.
            width, height = image.size
    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        raise FrameRejected("Frame could not be decoded as an image.") from error
    if width * height > max_pixels:
        raise FrameRejected(f"Frame resolution {width}x{height} exceeds {max_pixels} pixels.")


def decode_jpeg(data: bytes) -> np.ndarray:
    import cv2

    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise FrameRejected("Frame could not be decoded as an image.")
    return frame


class BrowserAnalysisService:
    def __init__(self, config: AppConfig, pipeline_factory: PipelineFactory | None = None):
        self.config = config
        self._pipeline_factory = pipeline_factory or (lambda: FramePipeline.from_config(replace(config)))
        # Each session runs CPU-heavy inference; cap concurrency instead of degrading every session.
        self._slots = asyncio.Semaphore(config.max_analyze_sessions)

    @asynccontextmanager
    async def session(self) -> AsyncIterator["BrowserAnalysisSession"]:
        if self._slots.locked():
            raise SessionLimitReached()
        async with self._slots:
            # Building a pipeline imports TensorFlow / warms up the model: keep it off the event loop.
            pipeline = await asyncio.to_thread(self._pipeline_factory)
            yield BrowserAnalysisSession(pipeline, self.config.max_frame_bytes, self.config.max_frame_pixels)


class BrowserAnalysisSession:
    def __init__(self, pipeline: FramePipeline, max_frame_bytes: int, max_frame_pixels: int):
        self._pipeline = pipeline
        self._max_frame_bytes = max_frame_bytes
        self._max_frame_pixels = max_frame_pixels

    @property
    def analysis_enabled(self) -> bool:
        return (
            self._pipeline.config.enable_emotion_analysis
            and bool(getattr(self._pipeline.analyzer, "enabled", True))
            and not getattr(self._pipeline.analyzer, "last_error_code", None)
        )

    async def analyze(self, data: bytes) -> dict[str, Any]:
        if len(data) > self._max_frame_bytes:
            raise FrameRejected(f"Frame exceeds {self._max_frame_bytes} bytes.")
        check_jpeg_header(data, self._max_frame_pixels)
        started = time.perf_counter()
        tracks = await asyncio.to_thread(self._process, data)
        return {
            "type": "analysis",
            "tracks": [serialize_track(track) for track in tracks],
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "analysis_enabled": self.analysis_enabled,
        }

    def _process(self, data: bytes) -> list[Any]:
        return self._pipeline.process(decode_jpeg(data))
