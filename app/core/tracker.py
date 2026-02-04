import time
from typing import List, Optional, Iterable
from dataclasses import dataclass, field
from .detector import FaceBox
from .analyzer import EmotionResult

@dataclass
class Track:
    id: int
    box: FaceBox
    start_time: float
    last_seen: float
    last_analyzed: float = 0.0
    emotion: Optional[EmotionResult] = None
    history: List[str] = field(default_factory=list)

class FaceTracker:
    """Tracks faces across frames using Intersection Over Union (IOU)."""
    
    def __init__(self, ttl: float = 1.5, iou_threshold: float = 0.3):
        self.ttl = ttl
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, detections: Iterable[FaceBox]) -> List[Track]:
        now = time.time()
        active_tracks = []
        unused_detections = list(detections)
        
        # Match existing tracks to new detections
        for track in self.tracks:
            best_iou = 0.0
            best_det_idx = -1
            
            for i, det in enumerate(unused_detections):
                score = self._calculate_iou(track.box, det)
                if score > best_iou:
                    best_iou = score
                    best_det_idx = i
            
            if best_det_idx != -1 and best_iou >= self.iou_threshold:
                det = unused_detections.pop(best_det_idx)
                track.box = det
                track.last_seen = now
                active_tracks.append(track)
        
        # Create new tracks for unmatched detections
        for det in unused_detections:
            new_track = Track(
                id=self._next_id,
                box=det,
                start_time=now,
                last_seen=now
            )
            self._next_id += 1
            active_tracks.append(new_track)
            self.tracks.append(new_track)

        # Cleanup expired tracks
        self.tracks = [t for t in self.tracks if (now - t.last_seen) <= self.ttl]
        
        return active_tracks

    @staticmethod
    def _calculate_iou(boxA: FaceBox, boxB: FaceBox) -> float:
        xA = max(boxA.x, boxB.x)
        yA = max(boxA.y, boxB.y)
        xB = min(boxA.x + boxA.w, boxB.x + boxB.w)
        yB = min(boxA.y + boxA.h, boxB.y + boxB.h)

        interWidth = max(0, xB - xA)
        interHeight = max(0, yB - yA)
        interArea = interWidth * interHeight

        unionArea = boxA.area + boxB.area - interArea
        return interArea / float(unionArea) if unionArea > 0 else 0.0
