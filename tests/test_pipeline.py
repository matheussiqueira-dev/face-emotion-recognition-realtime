import numpy as np

from app.core.models import FaceBox
from app.core.pipeline import FramePipeline, crop_face
from app.core.tracker import FaceTracker
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


def test_failed_analysis_respects_retry_interval():
    class FailingAnalyzer:
        calls = 0

        def analyze(self, face_img):
            self.calls += 1
            return None

    analyzer = FailingAnalyzer()
    pipeline = FramePipeline(make_config(), FakeDetector(), analyzer)
    pipeline.process(FRAME, now=100)
    pipeline.process(FRAME, now=100.1)
    assert analyzer.calls == 1
    pipeline.process(FRAME, now=100.6)
    assert analyzer.calls == 2


def test_disabled_analysis_does_not_invoke_model():
    analyzer = FakeAnalyzer()
    pipeline = FramePipeline(make_config(enable_emotion_analysis=False), FakeDetector(), analyzer)
    [track] = pipeline.process(FRAME, now=100)
    assert track.emotion is None
    assert analyzer.calls == 0


def test_crop_outside_frame_does_not_wrap_negative_indices():
    assert crop_face(FRAME, FaceBox(-100, -100, 10, 10)).size == 0
    assert crop_face(FRAME, FaceBox(200, 200, 10, 10)).size == 0


def test_expired_track_is_not_resurrected_by_matching_box():
    tracker = FaceTracker(ttl=1)
    box = FaceBox(10, 10, 20, 20)
    first = tracker.update([box], now=100)[0]
    second = tracker.update([box], now=102)[0]
    assert second.id != first.id
