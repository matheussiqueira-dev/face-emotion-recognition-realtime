import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .models import EmotionResult, FaceBox


@dataclass
class Track:
    id: int
    box: FaceBox
    start_time: float
    last_seen: float
    last_analyzed: float = 0.0
    emotion: Optional[EmotionResult] = None
    history: List[str] = field(default_factory=list)
    # Smoothed scores carried between inferences, and the latest unsmoothed label for transparency.
    smoothed_scores: Optional[Dict[str, float]] = None
    raw_dominant: Optional[str] = None


class FaceTracker:
    """Tracks faces across frames using Intersection Over Union (IOU)."""

    def __init__(self, ttl: float = 1.5, iou_threshold: float = 0.3):
        self.ttl = ttl
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, detections: Iterable[FaceBox], now: float | None = None) -> List[Track]:
        now = time.time() if now is None else now
        # Expire before matching, otherwise a face can resurrect an already expired track.
        self.tracks = [track for track in self.tracks if now - track.last_seen <= self.ttl]
        active_tracks: List[Track] = []
        detection_list = list(detections)

        # Global greedy matching: take track/detection pairs from the highest overlap down. Matching one
        # track at a time lets an older track take a box that belongs to its neighbour, swapping IDs and
        # mixing per-track state (smoothed scores). A handful of faces keeps the pair list tiny.
        candidates = sorted(
            (
                (iou, track_idx, det_idx)
                for track_idx, track in enumerate(self.tracks)
                for det_idx, det in enumerate(detection_list)
                if (iou := self._calculate_iou(track.box, det)) >= self.iou_threshold
            ),
            key=lambda candidate: candidate[0],
            reverse=True,
        )
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for _, track_idx, det_idx in candidates:
            if track_idx in matched_tracks or det_idx in matched_detections:
                continue
            matched_tracks.add(track_idx)
            matched_detections.add(det_idx)
            track = self.tracks[track_idx]
            track.box = detection_list[det_idx]
            track.last_seen = now
            active_tracks.append(track)

        # Create new tracks for unmatched detections
        unused_detections = [det for i, det in enumerate(detection_list) if i not in matched_detections]
        for det in unused_detections:
            new_track = Track(id=self._next_id, box=det, start_time=now, last_seen=now)
            self._next_id += 1
            active_tracks.append(new_track)
            self.tracks.append(new_track)

        # Cleanup expired tracks
        self.tracks = [t for t in self.tracks if (now - t.last_seen) <= self.ttl]

        return sorted(active_tracks, key=lambda track: track.id)

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
