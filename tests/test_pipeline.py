import numpy as np

from app.core.models import EmotionResult, FaceBox
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


class ScriptedAnalyzer:
    """Returns one predefined result per call, to exercise behaviour across consecutive inferences."""

    def __init__(self, *score_sets: dict[str, float]):
        self._score_sets = list(score_sets)
        self.calls = 0

    def analyze(self, face_img):
        scores = self._score_sets[min(self.calls, len(self._score_sets) - 1)]
        self.calls += 1
        return EmotionResult(dominant=max(scores, key=scores.get), scores=dict(scores), timestamp=0.0)


HAPPY = {"happy": 80.0, "sad": 20.0}
SAD = {"happy": 30.0, "sad": 70.0}


def test_smoothing_blends_scores_across_inferences():
    analyzer = ScriptedAnalyzer(HAPPY, SAD)
    pipeline = FramePipeline(make_config(emotion_smoothing=0.5), FakeDetector(), analyzer)

    pipeline.process(FRAME, now=100.0)
    [track] = pipeline.process(FRAME, now=100.6)

    # One noisy "sad" inference after a confident "happy" one does not flip the label.
    assert track.emotion.scores == {"happy": 55.0, "sad": 45.0}
    assert track.emotion.dominant == "happy"
    assert track.raw_dominant == "sad"


def test_smoothing_follows_a_sustained_change():
    analyzer = ScriptedAnalyzer(HAPPY, SAD, SAD, SAD)
    pipeline = FramePipeline(make_config(emotion_smoothing=0.5), FakeDetector(), analyzer)

    for step in range(4):
        [track] = pipeline.process(FRAME, now=100.0 + step * 0.6)

    assert track.emotion.dominant == "sad"


def test_zero_smoothing_reports_raw_model_output():
    analyzer = ScriptedAnalyzer(HAPPY, SAD)
    pipeline = FramePipeline(make_config(emotion_smoothing=0.0), FakeDetector(), analyzer)

    pipeline.process(FRAME, now=100.0)
    [track] = pipeline.process(FRAME, now=100.6)

    assert track.emotion.scores == SAD
    assert track.emotion.dominant == "sad"


def test_smoothing_state_is_per_track():
    analyzer = ScriptedAnalyzer(HAPPY, SAD)
    detector = FakeDetector(boxes=[FaceBox(10, 10, 40, 40)])
    pipeline = FramePipeline(make_config(emotion_smoothing=0.9), detector, analyzer)

    pipeline.process(FRAME, now=100.0)
    detector.boxes = [FaceBox(100, 70, 40, 40)]  # Different face: no overlap, new track.
    [track] = pipeline.process(FRAME, now=100.1)

    assert track.id == 2
    assert track.emotion.scores == SAD


def test_low_confidence_inference_does_not_update_smoothed_scores():
    analyzer = ScriptedAnalyzer(HAPPY, {"happy": 10.0, "sad": 15.0})
    pipeline = FramePipeline(
        make_config(emotion_smoothing=0.5, min_emotion_score=50.0), FakeDetector(), analyzer
    )

    pipeline.process(FRAME, now=100.0)
    [track] = pipeline.process(FRAME, now=100.6)
    assert track.emotion is None
    assert track.smoothed_scores == HAPPY


def test_tracker_matches_best_overlaps_first_so_neighbours_keep_their_ids():
    tracker = FaceTracker(ttl=1, iou_threshold=0.3)
    left, right = tracker.update([FaceBox(0, 0, 40, 40), FaceBox(16, 0, 40, 40)], now=100)

    # Both faces move left. The first-created track overlaps the right face's new box more than its own,
    # so matching tracks one at a time would hand it that box and lose the right face's identity.
    moved_right, moved_left = FaceBox(14, 0, 40, 40), FaceBox(-16, 0, 40, 40)
    tracks = tracker.update([moved_right, moved_left], now=100.1)

    by_id = {track.id: track.box for track in tracks}
    assert by_id == {left.id: moved_left, right.id: moved_right}
