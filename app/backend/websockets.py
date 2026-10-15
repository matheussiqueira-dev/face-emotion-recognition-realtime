from __future__ import annotations

import json
import logging

from fastapi import APIRouter
from fastapi import WebSocket
from fastapi import WebSocketDisconnect

from app.backend.analysis import FrameRejected
from app.backend.analysis import SessionLimitReached
from app.backend.security import Permission
from app.backend.security import authorize_websocket


LOGGER = logging.getLogger(__name__)

# RFC 6455: 1011 = server error, 1013 = try again later.
WS_CLOSE_INTERNAL_ERROR = 1011
WS_CLOSE_TRY_AGAIN_LATER = 1013

router = APIRouter()


@router.websocket("/api/v1/ws/stream")
async def websocket_stream_v1(websocket: WebSocket) -> None:
    await _handle_stream_connection(websocket)


@router.websocket("/ws/stream")
async def websocket_stream_legacy(websocket: WebSocket) -> None:
    await _handle_stream_connection(websocket)


async def _handle_stream_connection(websocket: WebSocket) -> None:
    """Server-side video source (camera index or file) broadcast to every connected dashboard."""
    state = websocket.app.state
    if not await authorize_websocket(websocket, state.config, Permission.READ):
        return

    service = state.stream_service
    try:
        await service.connect(websocket)
        while True:
            message = await websocket.receive_text()
            if _is_ping(message):
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        LOGGER.info("WebSocket client disconnected.")
    except Exception:
        LOGGER.exception("Unexpected websocket error.")
        await _close_quietly(websocket, WS_CLOSE_INTERNAL_ERROR, "Stream failed to start.")
    finally:
        await service.disconnect(websocket)


@router.websocket("/api/v1/ws/analyze")
async def websocket_analyze(websocket: WebSocket) -> None:
    """Browser-camera mode: binary JPEG frames in, {"type": "analysis", tracks, latency_ms} out.

    The client sends the next frame after receiving the previous result (stop-and-wait), which applies
    natural backpressure: a slow server lowers the frame rate instead of building an unbounded queue.
    """
    state = websocket.app.state
    if not await authorize_websocket(websocket, state.config, Permission.READ):
        return

    await websocket.accept()
    try:
        async with state.analysis_service.session() as session:
            await websocket.send_json({"type": "ready"})
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("text") is not None:
                    if _is_ping(message["text"]):
                        await websocket.send_json({"type": "pong"})
                    continue
                try:
                    result = await session.analyze(message.get("bytes") or b"")
                except FrameRejected as rejected:
                    await websocket.send_json({"type": "error", "code": "frame_rejected", "message": str(rejected)})
                    continue
                await websocket.send_json(result)
    except SessionLimitReached:
        await _close_quietly(websocket, WS_CLOSE_TRY_AGAIN_LATER, "Too many concurrent camera sessions.")
    except WebSocketDisconnect:
        pass
    except Exception:
        LOGGER.exception("Browser analysis session failed.")
        await _close_quietly(websocket, WS_CLOSE_INTERNAL_ERROR, "Analysis failed.")


def _is_ping(message: str) -> bool:
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and payload.get("type") == "ping"


async def _close_quietly(websocket: WebSocket, code: int, reason: str) -> None:
    try:
        await websocket.close(code=code, reason=reason)
    except RuntimeError:
        pass  # Already closed by the peer.
