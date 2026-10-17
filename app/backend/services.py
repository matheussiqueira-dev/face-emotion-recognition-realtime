from __future__ import annotations

import asyncio
import base64
import logging
import threading
from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from fastapi import WebSocket
from starlette.websockets import WebSocketState

from app.backend.serializers import serialize_track
from app.core.config import AppConfig
from app.core.metrics import SessionMetrics

if TYPE_CHECKING:
    from app.core.processor import VideoProcessor
    from app.core.tracker import Track

ProcessorFactory = Callable[[AppConfig], "VideoProcessor"]


def _default_processor_factory(config: AppConfig) -> "VideoProcessor":
    # Imported lazily: the processor pulls in TensorFlow, which the API should not need to start.
    from app.core.processor import VideoProcessor

    return VideoProcessor(config)


LOGGER = logging.getLogger(__name__)


class ProcessingBusy(RuntimeError):
    """A previous worker must finish before configuration/restart can proceed."""


class StreamService:
    """Coordinates video processing lifecycle and websocket broadcasts."""

    def __init__(
        self, config: AppConfig, processor_factory: ProcessorFactory | None = None, stop_timeout: float = 3.0
    ):
        self.config = config
        self._processor_factory = processor_factory or _default_processor_factory
        self._processor_lock = threading.Lock()
        self._processor: Any = None
        self._thread: threading.Thread | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._stop_timeout = stop_timeout
        self._status = "idle"
        self._error_code: str | None = None
        self._generation = 0

        self._metrics_lock = threading.Lock()
        self._metrics = SessionMetrics()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._frame_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=2)
        self._broadcast_task: asyncio.Task[None] | None = None

        self._clients: set[WebSocket] = set()
        self._clients_lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> bool:
        if websocket.application_state is not WebSocketState.CONNECTED:
            await websocket.accept()
        self._loop = asyncio.get_running_loop()
        async with self._clients_lock:
            if len(self._clients) >= self.config.max_stream_clients:
                await websocket.close(code=1013, reason="Too many stream clients.")
                return False
            self._clients.add(websocket)

        if self._broadcast_task is None or self._broadcast_task.done():
            self._broadcast_task = asyncio.create_task(self._broadcast_loop())

        await self._start_processor_async()
        return True

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._clients_lock:
            self._clients.discard(websocket)
            no_clients = not self._clients

        if no_clients:
            async with self._lifecycle_lock:
                if not self._clients:
                    await asyncio.to_thread(self._stop_processing_sync)

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
        async with self._lifecycle_lock:
            candidate = replace(self.config)
            changed_fields = candidate.update_from_dict(updates)
            if not changed_fields:
                return []
            if not await asyncio.to_thread(self._stop_processing_sync):
                raise ProcessingBusy("The previous video worker is still stopping. Retry shortly.")
            self.config.update_from_dict(updates)
            if self._clients:
                await asyncio.to_thread(self._start_processor)
            return changed_fields

    async def restart_processing(self) -> None:
        async with self._lifecycle_lock:
            if not await asyncio.to_thread(self._stop_processing_sync):
                raise ProcessingBusy("The previous video worker is still stopping. Retry shortly.")
            if self._clients:
                await asyncio.to_thread(self._start_processor)

    async def stop_processing(self) -> bool:
        async with self._lifecycle_lock:
            return await asyncio.to_thread(self._stop_processing_sync)

    def get_status_snapshot(self) -> dict[str, Any]:
        # Never wait on a lock held during model warmup from the HTTP event loop.
        return {"state": self._status, "error_code": self._error_code, "clients": len(self._clients)}

    def get_config_snapshot(self) -> dict[str, Any]:
        return self.config.as_dict(include_secrets=False)

    def get_metrics_snapshot(self) -> dict[str, Any]:
        with self._metrics_lock:
            return self._metrics.snapshot()

    async def _start_processor_async(self) -> None:
        # Building the processor imports TensorFlow and warms up the model, which takes seconds;
        # running it in a worker thread keeps the event loop (HTTP, other websockets) responsive.
        async with self._lifecycle_lock:
            await asyncio.to_thread(self._start_processor)

    def _start_processor(self) -> None:
        with self._processor_lock:
            # Check the thread rather than processor.is_running: the flag is only set once run()
            # starts, so a concurrent caller could otherwise start a second processor.
            if self._thread and self._thread.is_alive():
                if self._status == "stopping":
                    raise ProcessingBusy("The previous video worker is still stopping.")
                return

            self._status = "starting"
            self._error_code = None
            self._generation += 1
            with self._metrics_lock:
                self._metrics = SessionMetrics()
            try:
                self._processor = self._processor_factory(replace(self.config))
            except Exception:
                self._status = "failed"
                self._error_code = "processor_initialization_failed"
                raise

            thread = threading.Thread(
                target=self._run_processor,
                args=(self._processor, self._generation),
                name="emotion-video-processor",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            LOGGER.info("Video processor started.")

    def _run_processor(self, processor: Any, generation: int) -> None:
        try:
            processor.run(callback=lambda frame, tracks, fps: self._on_frame(frame, tracks, fps, generation))
        except Exception:
            self._status = "failed"
            self._error_code = "video_processing_failed"
            LOGGER.exception("Processor stopped due to an unexpected error.")
        finally:
            if self._status != "failed":
                self._status = "idle"
            LOGGER.info("Video processor thread finished.")

    def _stop_processing_sync(self) -> bool:
        self._generation += 1
        with self._processor_lock:
            processor = self._processor
            thread = self._thread

        if processor:
            self._status = "stopping"
            processor.stop()
        if thread and thread.is_alive():
            thread.join(timeout=self._stop_timeout)
            if thread.is_alive():
                self._error_code = "worker_stop_timeout"
                return False

        with self._processor_lock:
            self._processor = None
            self._thread = None
        self._status = "idle"
        self._error_code = None
        return True

    def _on_frame(self, frame, tracks: Iterable["Track"], fps: float, generation: int | None = None) -> None:
        generation = self._generation if generation is None else generation
        if self._loop is None or generation != self._generation:
            return

        encoded = self._encode_frame(frame)
        if encoded is None:
            return

        processor = self._processor
        pipeline = getattr(processor, "pipeline", None)
        analysis_enabled = bool(
            self.config.enable_emotion_analysis
            and getattr(getattr(pipeline, "analyzer", None), "enabled", True)
            and not getattr(getattr(pipeline, "analyzer", None), "last_error_code", None)
        )
        if self._status == "stopping":
            return
        self._status = "running" if analysis_enabled else "degraded"
        self._error_code = None if analysis_enabled else "emotion_analysis_unavailable"
        tracks = list(tracks)
        payload = {
            "frame": encoded,
            "data": {
                "fps": round(float(fps), 2),
                "tracks": [serialize_track(track) for track in tracks],
                "analysis_enabled": analysis_enabled,
            },
        }

        with self._metrics_lock:
            self._register_metrics(tracks, fps)

        self._loop.call_soon_threadsafe(self._queue_payload, payload, generation)

    def _register_metrics(self, tracks: list["Track"], fps: float) -> None:
        # No min_score here: FramePipeline gates on the raw model confidence, so every label it emits is
        # already above the threshold even when smoothing lowers the displayed scores.
        self._metrics.register_frame(tracks=tracks, fps=fps)

    def _queue_payload(self, payload: dict[str, Any], generation: int) -> None:
        if generation != self._generation:
            return
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

                async def send(ws: WebSocket) -> WebSocket | None:
                    try:
                        await asyncio.wait_for(ws.send_json(payload), timeout=1.0)
                        return None
                    except Exception:
                        LOGGER.warning("Failed to send frame; dropping websocket client.")
                        try:
                            await asyncio.wait_for(ws.close(code=1013), timeout=0.1)
                        except Exception:
                            pass
                        return ws

                disconnected = [
                    ws for ws in await asyncio.gather(*(send(ws) for ws in clients)) if ws is not None
                ]

                if disconnected:
                    async with self._clients_lock:
                        for ws in disconnected:
                            self._clients.discard(ws)
                        no_clients = not self._clients

                    if no_clients:
                        async with self._lifecycle_lock:
                            if not self._clients:
                                await asyncio.to_thread(self._stop_processing_sync)
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
