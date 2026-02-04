import cv2
import time
import logging
import numpy as np
from typing import Optional, Tuple, Protocol
from .config import AppConfig
from .detector import FaceDetector, FaceBox
from .analyzer import EmotionAnalyzer, EmotionResult
from .tracker import FaceTracker, Track

LOGGER = logging.getLogger(__name__)

class FrameCallback(Protocol):
    def __call__(self, frame: np.ndarray, tracks: list[Track], fps: float) -> None: ...

class VideoProcessor:
    """Orchestrates the video capture, detection, analysis, and tracking flow."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.detector = FaceDetector(config.cascade_path)
        self.analyzer = EmotionAnalyzer(enabled=True)
        self.tracker = FaceTracker(ttl=config.track_ttl, iou_threshold=config.iou_threshold)
        self._stop_requested = False

    def run(self, callback: Optional[FrameCallback] = None):
        """Starts the video processing loop."""
        source = self.config.video_source
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            LOGGER.error(f"Could not open video source: {source}")
            return

        # Configure capture settings
        if self.config.width: cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        if self.config.height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        
        last_time = time.time()
        
        try:
            while not self._stop_requested:
                success, frame = cap.read()
                if not success:
                    if not source.isdigit(): # If video file, loop or stop
                        break
                    continue

                now = time.time()
                dt = now - last_time
                last_time = now
                fps = 1.0 / dt if dt > 0 else 0

                # 1. Detection
                detections = self.detector.detect(
                    frame, 
                    scale=self.config.detect_scale,
                    min_neighbors=self.config.min_neighbors,
                    min_size=self.config.min_face_size
                )

                # 2. Tracking
                tracks = self.tracker.update(detections)

                # 3. Analyze emotions for untracked or interval-reached tracks
                for track in tracks:
                    if (now - track.last_analyzed) >= self.config.emotion_interval:
                        face_img = self._crop_face(frame, track.box)
                        if face_img.size > 0:
                            emotion = self.analyzer.analyze(face_img)
                            if emotion:
                                track.emotion = emotion
                                track.last_analyzed = now
                                track.history.append(emotion.dominant)
                                if len(track.history) > 50:
                                    track.history.pop(0)

                # 4. Callback for UI or streaming
                if callback:
                    callback(frame, tracks, fps)
                
                # Internal display if no callback (legacy mode)
                elif not callback:
                    self._default_draw(frame, tracks, fps)
                    cv2.imshow("Face Emotion Recognition - Senior Edition", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def stop(self):
        self._stop_requested = True

    def _crop_face(self, frame: np.ndarray, box: FaceBox) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1 = max(0, box.x), max(0, box.y)
        x2, y2 = min(w, box.x + box.w), min(h, box.y + box.h)
        return frame[y1:y2, x1:x2]

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
