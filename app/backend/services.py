from __future__ import annotations

import asyncio
import base64
import logging
import threading
from collections.abc import Iterable
from typing import Any
from typing import TYPE_CHECKING

from fastapi import WebSocket

from app.core.config import AppConfig
from app.core.metrics import SessionMetrics

if TYPE_CHECKING:
    from app.core.processor import Track


LOGGER = logging.getLogger(__name__)


class StreamService:
    """Coordinates video processing lifecycle and websocket broadcasts."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._processor_lock = threading.Lock()
        self._processor: Any = None
        self._thread: threading.Thread | None = None

        self._metrics_lock = threading.Lock()
        self._metrics = SessionMetrics()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._frame_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=2)
        self._broadcast_task: asyncio.Task[None] | None = None

        self._clients: set[WebSocket] = set()
        self._clients_lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._loop = asyncio.get_running_loop()
        async with self._clients_lock:
            self._clients.add(websocket)

        if self._broadcast_task is None or self._broadcast_task.done():
            self._broadcast_task = asyncio.create_task(self._broadcast_loop())

        self._start_processor()

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._clients_lock:
            self._clients.discard(websocket)
            no_clients = not self._clients

        if no_clients:
            await self.stop_processing()

    async def shutdown(self) -> None:
        await self.stop_processing()
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
            self._broadcast_task = None

    async def update_config(self, updates: dict[str, Any]) -> list[str]:
        changed_fields = self.config.update_from_dict(updates)
        if not changed_fields:
            return []

        with self._processor_lock:
            running = bool(self._processor and self._processor.is_running)

        if running:
            await self.restart_processing()
        return changed_fields

    async def restart_processing(self) -> None:
        await self.stop_processing()
        has_clients = bool(self._clients)
        if has_clients:
            self._start_processor()

    async def stop_processing(self) -> None:
        await asyncio.to_thread(self._stop_processing_sync)

    def get_config_snapshot(self) -> dict[str, Any]:
        return self.config.as_dict(include_secrets=False)

    def get_metrics_snapshot(self) -> dict[str, Any]:
        with self._metrics_lock:
            return self._metrics.snapshot()

    def _start_processor(self) -> None:
        with self._processor_lock:
            if self._processor and self._processor.is_running:
                return

            self._metrics = SessionMetrics()
            from app.core.processor import VideoProcessor

            self._processor = VideoProcessor(self.config)

            thread = threading.Thread(
                target=self._run_processor,
                name="emotion-video-processor",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            LOGGER.info("Video processor started.")

    def _run_processor(self) -> None:
        assert self._processor is not None
        try:
            self._processor.run(callback=self._on_frame)
        except Exception:
            LOGGER.exception("Processor stopped due to an unexpected error.")
        finally:
            LOGGER.info("Video processor thread finished.")

    def _stop_processing_sync(self) -> None:
        with self._processor_lock:
            processor = self._processor
            thread = self._thread

        if processor:
            processor.stop()
        if thread and thread.is_alive():
            thread.join(timeout=3.0)

        with self._processor_lock:
            self._processor = None
            self._thread = None

    def _on_frame(self, frame, tracks: Iterable["Track"], fps: float) -> None:
        if self._loop is None:
            return

        encoded = self._encode_frame(frame)
        if encoded is None:
            return

        payload = {
            "frame": encoded,
            "data": {
                "fps": round(float(fps), 2),
                "tracks": [self._serialize_track(track) for track in tracks],
            },
        }

        with self._metrics_lock:
            self._metrics.register_frame(
                tracks=tracks,
                fps=fps,
                min_score=self.config.min_emotion_score,
            )

        self._loop.call_soon_threadsafe(self._queue_payload, payload)

    def _queue_payload(self, payload: dict[str, Any]) -> None:
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self._frame_queue.put_nowait(payload)

    async def _broadcast_loop(self) -> None:
        try:
            while True:
                payload = await self._frame_queue.get()

                async with self._clients_lock:
                    clients = list(self._clients)

                if not clients:
                    continue

                disconnected: list[WebSocket] = []
                for ws in clients:
                    try:
                        await ws.send_json(payload)
                    except Exception:
                        disconnected.append(ws)

                if disconnected:
                    async with self._clients_lock:
                        for ws in disconnected:
                            self._clients.discard(ws)
                        no_clients = not self._clients

                    if no_clients:
                        await self.stop_processing()
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("Broadcast loop failed.")

    def _encode_frame(self, frame) -> str | None:
        import cv2

        quality = int(max(1, min(100, self.config.jpeg_quality)))
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None
        return base64.b64encode(buffer).decode("utf-8")

    @staticmethod
    def _serialize_track(track: Any) -> dict[str, Any]:
        serialized: dict[str, Any] = {
            "id": track.id,
            "box": {
                "x": track.box.x,
                "y": track.box.y,
                "w": track.box.w,
                "h": track.box.h,
            },
            "emotion": None,
        }
        if track.emotion:
            serialized["emotion"] = {
                "dominant": track.emotion.dominant,
                "scores": track.emotion.scores,
            }
        return serialized
