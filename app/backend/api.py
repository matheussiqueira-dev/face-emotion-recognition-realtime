import cv2
import json
import asyncio
import logging
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from .processor import VideoProcessor, Track
from .config import AppConfig

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Face Emotion Recognition API")

# Mount static files for the frontend
frontend_path = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

processor = VideoProcessor(AppConfig())

@app.get("/")
async def get_index():
    with open(frontend_path / "index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    def frame_callback(frame, tracks, fps):
        # Encode frame to base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare data
        data = {
            "fps": round(fps, 1),
            "tracks": []
        }
        
        for track in tracks:
            track_data = {
                "id": track.id,
                "box": {"x": track.box.x, "y": track.box.y, "w": track.box.w, "h": track.box.h},
                "emotion": None
            }
            if track.emotion:
                track_data["emotion"] = {
                    "dominant": track.emotion.dominant,
                    "scores": track.emotion.scores
                }
            data["tracks"].append(track_data)
            
        # Send via websocket (this needs to be handled carefully in a separate thread/task)
        # For simplicity in this demo, we'll use a queue or similar if needed, 
        # but here we'll try to just wrap it in a future.
        try:
            asyncio.run_coroutine_threadsafe(
                websocket.send_json({"frame": frame_base64, "data": data}),
                loop
            )
        except Exception as e:
            pass

    loop = asyncio.get_event_loop()
    
    # Run the processor in a separate thread
    import threading
    thread = threading.Thread(target=processor.run, args=(frame_callback,), daemon=True)
    thread.start()

    try:
        while True:
            # Keep connection alive and handle client messages (like settings updates)
            msg = await websocket.receive_text()
            # Handle messages here if needed
    except WebSocketDisconnect:
        processor.stop()
        LOGGER.info("Client disconnected")
    except Exception as e:
        LOGGER.error(f"WebSocket error: {e}")
        processor.stop()
