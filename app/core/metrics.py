from __future__ import annotations

import time
from collections import Counter
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Iterable


EMOTION_SENTIMENT_WEIGHTS: dict[str, float] = {
    "happy": 1.0,
    "surprise": 0.5,
    "neutral": 0.0,
    "sad": -0.5,
    "fear": -0.7,
    "angry": -0.8,
    "disgust": -1.0,
}


@dataclass(slots=True)
class SessionMetrics:
    started_at: float = field(default_factory=time.time)
    last_frame_at: float = 0.0
    frames_processed: int = 0
    detections_total: int = 0
    analyzed_faces: int = 0
    peak_tracks: int = 0
    latest_tracks: int = 0
    current_fps: float = 0.0
    average_fps: float = 0.0
    dominant_emotions: Counter[str] = field(default_factory=Counter)
    sentiment_history: deque[float] = field(default_factory=lambda: deque(maxlen=240))

    def register_frame(self, tracks: Iterable[Any], fps: float, min_score: float = 0.0) -> None:
        track_list = list(tracks)
        self.frames_processed += 1
        self.last_frame_at = time.time()
        self.latest_tracks = len(track_list)
        self.detections_total += len(track_list)
        self.peak_tracks = max(self.peak_tracks, len(track_list))
        self.current_fps = round(max(0.0, fps), 2)
        self.average_fps = round(
            ((self.average_fps * (self.frames_processed - 1)) + self.current_fps) / self.frames_processed,
            2,
        )

        sentiment_values: list[float] = []
        for track in track_list:
            emotion = getattr(track, "emotion", None)
            if not emotion:
                continue
            scores = getattr(emotion, "scores", {}) or {}
            dominant = str(getattr(emotion, "dominant", "")).lower()
            max_score = max(scores.values()) if scores else 0.0
            if max_score < min_score:
                continue

            self.analyzed_faces += 1
            self.dominant_emotions[dominant] += 1
            sentiment_values.append(EMOTION_SENTIMENT_WEIGHTS.get(dominant, 0.0))

        if sentiment_values:
            self.sentiment_history.append(round(sum(sentiment_values) / len(sentiment_values), 3))
        elif self.sentiment_history:
            self.sentiment_history.append(self.sentiment_history[-1])
        else:
            self.sentiment_history.append(0.0)

    def snapshot(self) -> dict[str, Any]:
        started_at = datetime.fromtimestamp(self.started_at, tz=timezone.utc).isoformat()
        last_frame_at = (
            datetime.fromtimestamp(self.last_frame_at, tz=timezone.utc).isoformat()
            if self.last_frame_at
            else None
        )
        total_emotions = sum(self.dominant_emotions.values()) or 1
        dominant_distribution = {
            label: round((count / total_emotions) * 100.0, 2)
            for label, count in sorted(self.dominant_emotions.items())
        }
        return {
            "started_at": started_at,
            "last_frame_at": last_frame_at,
            "uptime_seconds": round(max(0.0, time.time() - self.started_at), 2),
            "frames_processed": self.frames_processed,
            "detections_total": self.detections_total,
            "analyzed_faces": self.analyzed_faces,
            "latest_tracks": self.latest_tracks,
            "peak_tracks": self.peak_tracks,
            "current_fps": self.current_fps,
            "average_fps": self.average_fps,
            "dominant_distribution": dominant_distribution,
            "sentiment_history": list(self.sentiment_history),
        }
