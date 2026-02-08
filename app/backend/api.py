import cv2
import asyncio
import logging
import base64
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.core.processor import VideoProcessor, Track
from app.core.config import AppConfig

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Face Emotion Recognition API")

# Mount static files for the frontend
frontend_path = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

config = AppConfig()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

processor = VideoProcessor(config)
processor_thread: Optional[threading.Thread] = None
processor_lock = threading.Lock()


class StreamBroadcaster:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def register(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.add(websocket)

    async def unregister(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        async with self._lock:
            disconnected = []
            for client in self._clients:
                try:
                    await client.send_json(message)
                except Exception:
                    disconnected.append(client)
            for client in disconnected:
                self._clients.discard(client)

    def send_from_thread(self, message: Dict[str, Any]) -> None:
        if not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self._loop)

    def client_count(self) -> int:
        return len(self._clients)

@app.get("/")
async def get_index():
    with open(frontend_path / "index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/health")
async def get_health():
    return JSONResponse(content={"status": "ok"})


@app.get("/api/config")
async def get_config():
    return JSONResponse(content={
        "video_source": config.video_source,
        "emotion_interval": config.emotion_interval,
        "detect_scale": config.detect_scale,
        "min_face_size": config.min_face_size,
        "track_ttl": config.track_ttl,
        "iou_threshold": config.iou_threshold,
    })


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()
    if not hasattr(app.state, "broadcaster"):
        app.state.broadcaster = StreamBroadcaster(loop)
    broadcaster: StreamBroadcaster = app.state.broadcaster
    await broadcaster.register(websocket)

    def frame_callback(frame, tracks, fps):
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_base64 = base64.b64encode(buffer).decode("utf-8")
        payload = {
            "frame": frame_base64,
            "data": {
                "fps": round(fps, 1),
                "tracks": [_serialize_track(track) for track in tracks],
            },
        }
        broadcaster.send_from_thread(payload)

    _ensure_processor_running(frame_callback)

    try:
        while True:
            # Keep connection alive and handle client messages (like settings updates)
            await websocket.receive_text()
            # Handle messages here if needed
    except WebSocketDisconnect:
        await broadcaster.unregister(websocket)
        _stop_processor_if_idle(broadcaster)
        LOGGER.info("Client disconnected")
    except Exception as e:
        LOGGER.error(f"WebSocket error: {e}")
        await broadcaster.unregister(websocket)
        _stop_processor_if_idle(broadcaster)


def _serialize_track(track: Track) -> Dict[str, Any]:
    data = {
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
        data["emotion"] = {
            "dominant": track.emotion.dominant,
            "scores": track.emotion.scores,
        }
    return data


def _ensure_processor_running(callback) -> None:
    global processor_thread
    with processor_lock:
        if processor_thread and processor_thread.is_alive():
            return
        processor_thread = threading.Thread(
            target=processor.run,
            args=(callback,),
            daemon=True,
        )
        processor_thread.start()


def _stop_processor_if_idle(broadcaster: StreamBroadcaster) -> None:
    global processor_thread
    if broadcaster.client_count() > 0:
        return
    with processor_lock:
        processor.stop()
        if processor_thread:
            processor_thread.join(timeout=1.0)
            processor_thread = None
