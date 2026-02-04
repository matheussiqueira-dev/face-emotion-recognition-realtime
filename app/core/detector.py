import cv2
import numpy as np
from pathlib import Optional, Path
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def center(self) -> Tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0

class FaceDetector:
    """Handles face detection using Haar Cascades."""
    
    def __init__(self, cascade_path: Optional[Path] = None):
        self.cascade_path = self._resolve_cascade_path(cascade_path)
        self.classifier = cv2.CascadeClassifier(str(self.cascade_path))
        
        if self.classifier.empty():
            raise RuntimeError(f"Could not load Haar Cascade from {self.cascade_path}")

    def detect(self, frame: np.ndarray, scale: float = 1.0, min_neighbors: int = 5, min_size: int = 30) -> List[FaceBox]:
        if scale != 1.0:
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        else:
            small_frame = frame

        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray) # Improve detection in different lighting

        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=min_neighbors,
            minSize=(int(min_size * scale), int(min_size * scale))
        )

        inv_scale = 1.0 / scale
        return [
            FaceBox(
                int(x * inv_scale),
                int(y * inv_scale),
                int(w * inv_scale),
                int(h * inv_scale)
            ) for (x, y, w, h) in faces
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
