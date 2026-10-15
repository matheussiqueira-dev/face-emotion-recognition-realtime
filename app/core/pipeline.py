from __future__ import annotations

import time
from typing import Optional, Protocol

import numpy as np

from .config import AppConfig
from .models import EmotionResult, FaceBox
from .tracker import FaceTracker, Track


class FaceDetectorPort(Protocol):
    def detect(
        self,
        frame: np.ndarray,
        scale: float = 1.0,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: int = 30,
    ) -> list[FaceBox]: ...


class EmotionAnalyzerPort(Protocol):
    def analyze(self, face_img: np.ndarray) -> Optional[EmotionResult]: ...


class FramePipeline:
    """Per-frame domain logic shared by every video source: detect faces, track them, classify emotions.

    Collaborators are injected so the pipeline can be tested without OpenCV models or TensorFlow.
    """

    def __init__(
        self,
        config: AppConfig,
        detector: FaceDetectorPort,
        analyzer: EmotionAnalyzerPort,
        tracker: FaceTracker | None = None,
    ):
        self.config = config
        self.detector = detector
        self.analyzer = analyzer
        self.tracker = tracker or FaceTracker(ttl=config.track_ttl, iou_threshold=config.iou_threshold)

    @classmethod
    def from_config(cls, config: AppConfig) -> "FramePipeline":
        # Imported lazily: building the analyzer imports TensorFlow and warms up the model.
        from .analyzer import EmotionAnalyzer
        from .detector import FaceDetector

        return cls(
            config=config,
            detector=FaceDetector(config.cascade_path),
            analyzer=EmotionAnalyzer(enabled=config.enable_emotion_analysis),
        )

    def process(self, frame: np.ndarray, now: float | None = None) -> list[Track]:
        now = time.time() if now is None else now
        detections = self.detector.detect(
            frame,
            scale=self.config.detect_scale,
            scale_factor=self.config.scale_factor,
            min_neighbors=self.config.min_neighbors,
            min_size=self.config.min_face_size,
        )
        tracks = self.tracker.update(detections)

        for track in tracks:
            if (now - track.last_analyzed) < self.config.emotion_interval:
                continue
            face_img = crop_face(frame, track.box)
            if face_img.size == 0:
                continue
            emotion = self.analyzer.analyze(face_img)
            if emotion:
                max_score = max(emotion.scores.values()) if emotion.scores else 0.0
                if max_score >= self.config.min_emotion_score:
                    track.emotion = emotion
                    track.history.append(emotion.dominant)
                    if len(track.history) > 50:
                        track.history.pop(0)
                track.last_analyzed = now
        return tracks


def crop_face(frame: np.ndarray, box: FaceBox) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1 = max(0, box.x), max(0, box.y)
    x2, y2 = min(w, box.x + box.w), min(h, box.y + box.h)
    return frame[y1:y2, x1:x2]
