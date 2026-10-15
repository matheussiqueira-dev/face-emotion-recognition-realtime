import numpy as np

from app.core.pipeline import FramePipeline
from tests.conftest import FakeAnalyzer, FakeDetector, make_config

FRAME = np.zeros((120, 160, 3), dtype=np.uint8)


def test_pipeline_tracks_faces_and_classifies_emotion():
    pipeline = FramePipeline(make_config(), detector=FakeDetector(), analyzer=FakeAnalyzer())
    [track] = pipeline.process(FRAME, now=100.0)
    assert track.id == 1
    assert track.emotion.dominant == "happy"
    assert track.history == ["happy"]


def test_pipeline_respects_emotion_interval():
    analyzer = FakeAnalyzer()
    pipeline = FramePipeline(make_config(emotion_interval=0.5), detector=FakeDetector(), analyzer=analyzer)

    pipeline.process(FRAME, now=100.0)
    pipeline.process(FRAME, now=100.2)  # Same track, inside the interval: no new inference.
    assert analyzer.calls == 1

    pipeline.process(FRAME, now=100.6)
    assert analyzer.calls == 2


def test_pipeline_ignores_low_confidence_emotions():
    pipeline = FramePipeline(
        make_config(min_emotion_score=95.0), detector=FakeDetector(), analyzer=FakeAnalyzer(score=60.0)
    )
    [track] = pipeline.process(FRAME, now=100.0)
    assert track.emotion is None


def test_pipeline_with_no_faces_returns_no_tracks():
    pipeline = FramePipeline(make_config(), detector=FakeDetector(boxes=[]), analyzer=FakeAnalyzer())
    assert pipeline.process(FRAME, now=100.0) == []
