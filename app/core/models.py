"""Domain value objects. Deliberately dependency-free (no OpenCV / TensorFlow imports) so any layer can use them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


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


@dataclass
class EmotionResult:
    dominant: str
    scores: Dict[str, float]
    timestamp: float
