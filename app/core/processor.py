import sys
import cv2
import threading
import time
import logging
import numpy as np
from typing import Optional, Protocol
from .config import AppConfig
from .pipeline import FramePipeline
from .tracker import Track

LOGGER = logging.getLogger(__name__)

class FrameCallback(Protocol):
    def __call__(self, frame: np.ndarray, tracks: list[Track], fps: float) -> None: ...

class VideoProcessor:
    """Orchestrates the video capture, detection, analysis, and tracking flow."""
    
    def __init__(self, config: AppConfig, pipeline: Optional[FramePipeline] = None):
        self.config = config
        self.pipeline = pipeline or FramePipeline.from_config(config)
        self._stop_requested = threading.Event()
        self._is_running = threading.Event()

    def run(self, callback: Optional[FrameCallback] = None):
        """Starts the video processing loop."""
        self._stop_requested.clear()
        self._is_running.set()

        source = str(self.config.video_source)
        if source.isdigit():
            # DirectShow only exists on Windows; elsewhere let OpenCV pick the backend (e.g. V4L2).
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            cap = cv2.VideoCapture(int(source), backend)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            LOGGER.error(f"Could not open video source: {source}")
            self._is_running.clear()
            return

        # Configure capture settings
        if self.config.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        if self.config.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if self.config.fps:
            cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        
        last_time = time.time()
        
        try:
            while not self._stop_requested.is_set():
                frame_loop_start = time.perf_counter()
                success, frame = cap.read()
                if not success:
                    if source.isdigit():
                        continue
                    # End of file/stream: rewind once for looping demos, otherwise stop.
                    if self.config.loop_video and cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                        success, frame = cap.read()
                    if not success:
                        break

                now = time.time()
                dt = now - last_time
                last_time = now
                fps = 1.0 / dt if dt > 0 else 0

                tracks = self.pipeline.process(frame, now=now)

                # Callback for UI or streaming
                if callback:
                    try:
                        callback(frame, tracks, fps)
                    except Exception as callback_error:
                        LOGGER.exception("Frame callback failed: %s", callback_error)
                
                # Internal display if no callback (legacy mode)
                elif not callback:
                    self._default_draw(frame, tracks, fps)
                    cv2.imshow("Face Emotion Recognition - Senior Edition", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                # Frame pacing to cap CPU usage.
                target_frame_time = 1.0 / max(1, self.config.max_fps)
                elapsed = time.perf_counter() - frame_loop_start
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)

        finally:
            cap.release()
            # Windows only exist in legacy display mode; headless OpenCV builds lack GUI support.
            if not callback:
                cv2.destroyAllWindows()
            self._is_running.clear()

    def stop(self):
        self._stop_requested.set()

    @property
    def is_running(self) -> bool:
        return self._is_running.is_set()

    def _default_draw(self, frame: np.ndarray, tracks: list[Track], fps: float):
        """Default visualization for internal display."""
        for track in tracks:
            b = track.box
            color = (99, 102, 241) # Indigo (BGR)
            cv2.rectangle(frame, (b.x, b.y), (b.x + b.w, b.y + b.h), color, 2)
            
            label = f"ID:{track.id}"
            if track.emotion:
                label += f" | {track.emotion.dominant} ({max(track.emotion.scores.values()):.0f}%)"
            
            cv2.putText(frame, label, (b.x, b.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
