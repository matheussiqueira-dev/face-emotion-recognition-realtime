from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

import cv2
import numpy as np

from .models import FaceBox

if TYPE_CHECKING:
    from .config import AppConfig
    from .pipeline import FaceDetectorPort


class FaceDetector:
    """Handles face detection using Haar Cascades."""

    def __init__(self, cascade_path: Optional[Path] = None):
        self.cascade_path = self._resolve_cascade_path(cascade_path)
        # OpenCV's native file loader can reject Unicode paths on Windows.
        # Python reads the path correctly; OpenCV parses the same XML from memory.
        cascade_xml = self.cascade_path.read_text(encoding="utf-8")
        storage = cv2.FileStorage(cascade_xml, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
        self.classifier = cv2.CascadeClassifier()
        try:
            self.classifier.read(storage.getFirstTopLevelNode())
        finally:
            storage.release()

        if self.classifier.empty():
            raise RuntimeError(f"Could not load Haar Cascade from {self.cascade_path}")

    def detect(
        self,
        frame: np.ndarray,
        scale: float = 1.0,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: int = 30,
    ) -> List[FaceBox]:
        scale = max(0.1, min(scale, 1.0))
        scale_factor = max(1.01, scale_factor)
        if scale != 1.0:
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        else:
            small_frame = frame

        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Improve detection in different lighting

        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(int(min_size * scale), int(min_size * scale)),
        )

        inv_scale = 1.0 / scale
        return [
            FaceBox(int(x * inv_scale), int(y * inv_scale), int(w * inv_scale), int(h * inv_scale))
            for (x, y, w, h) in faces
        ]

    @staticmethod
    def _resolve_cascade_path(custom_path: Optional[Path]) -> Path:
        if custom_path and custom_path.exists():
            return custom_path

        # Default locations
        search_paths = [
            Path.cwd() / "haarcascade_frontalface_default.xml",
            Path(__file__).parent.parent.parent / "haarcascade_frontalface_default.xml",
        ]

        if hasattr(cv2, "data"):
            search_paths.append(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")

        for path in search_paths:
            if path.exists():
                return path

        raise FileNotFoundError("Haar Cascade XML not found. Please provide a valid path.")


class YuNetDetector:
    """Face detection with OpenCV's YuNet CNN (cv2.FaceDetectorYN, model from opencv/opencv_zoo, MIT).

    Same port as FaceDetector. Haar-specific tuning (scale_factor, min_neighbors) does not apply; the model's
    own score threshold and non-maximum suppression replace it.
    """

    SCORE_THRESHOLD = 0.8
    NMS_THRESHOLD = 0.3
    TOP_K = 50

    def __init__(self, model_path: Optional[Path] = None, backend: Any = None):
        if backend is None:
            if model_path is None:
                raise ValueError("YuNetDetector needs a model path.")
            # Load from memory for the same reason as the Haar cascade: OpenCV's own file loader can
            # reject Unicode paths on Windows.
            backend = cv2.FaceDetectorYN.create(
                "onnx",
                np.frombuffer(model_path.read_bytes(), dtype=np.uint8),
                np.empty(0, dtype=np.uint8),
                (320, 320),
                self.SCORE_THRESHOLD,
                self.NMS_THRESHOLD,
                self.TOP_K,
            )
        self.model_path = model_path
        self._backend = backend

    def detect(
        self,
        frame: np.ndarray,
        scale: float = 1.0,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: int = 30,
    ) -> List[FaceBox]:
        scale = max(0.1, min(scale, 1.0))
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale) if scale != 1.0 else frame
        height, width = small_frame.shape[:2]
        # The input size must match every frame: sources can change resolution between sessions.
        self._backend.setInputSize((width, height))
        _, faces = self._backend.detect(small_frame)
        if faces is None:
            return []

        inv_scale = 1.0 / scale
        boxes = []
        for row in faces:
            x, y, w, h = (round(float(value) * inv_scale) for value in row[:4])
            if w >= min_size and h >= min_size:
                boxes.append(FaceBox(x, y, w, h))
        return boxes


def create_detector(config: AppConfig) -> FaceDetectorPort:
    if config.face_detector == "yunet":
        return YuNetDetector(config.yunet_model_path)
    return FaceDetector(config.cascade_path)
