from dataclasses import dataclass

from app.core.metrics import SessionMetrics


@dataclass
class FakeEmotion:
    dominant: str
    scores: dict[str, float]


@dataclass
class FakeTrack:
    id: int
    emotion: FakeEmotion | None


def test_session_metrics_registers_emotions_and_sentiment():
    metrics = SessionMetrics()
    tracks = [
        FakeTrack(id=1, emotion=FakeEmotion(dominant="happy", scores={"happy": 88.0})),
        FakeTrack(id=2, emotion=FakeEmotion(dominant="sad", scores={"sad": 72.0})),
    ]
    metrics.register_frame(tracks=tracks, fps=27.5, min_score=0.0)

    snapshot = metrics.snapshot()
    assert snapshot["frames_processed"] == 1
    assert snapshot["detections_total"] == 2
    assert snapshot["analyzed_faces"] == 2
    assert snapshot["current_fps"] == 27.5
    assert "happy" in snapshot["dominant_distribution"]
    assert snapshot["sentiment_history"]


def test_session_metrics_respects_min_score_filter():
    metrics = SessionMetrics()
    tracks = [FakeTrack(id=1, emotion=FakeEmotion(dominant="fear", scores={"fear": 35.0}))]
    metrics.register_frame(tracks=tracks, fps=20.0, min_score=50.0)

    snapshot = metrics.snapshot()
    assert snapshot["analyzed_faces"] == 0
    assert snapshot["dominant_distribution"] == {}
