import os
from pathlib import Path

import numpy as np
import pytest

from app.core.config import AppConfig
from app.core.detector import YuNetDetector
from app.core.models import FaceBox

FRAME = np.zeros((200, 400, 3), dtype=np.uint8)


class FakeYuNet:
    """Stands in for cv2.FaceDetectorYN: rows of x, y, w, h, 5 landmark points, score."""

    def __init__(self, faces: np.ndarray | None):
        self.faces = faces
        self.input_sizes: list[tuple[int, int]] = []

    def setInputSize(self, size):
        self.input_sizes.append(tuple(size))

    def detect(self, image):
        return (0 if self.faces is None else len(self.faces)), self.faces


def face_row(x, y, w, h, score=0.95):
    return [x, y, w, h, *([0.0] * 10), score]


def test_yunet_boxes_are_scaled_back_to_frame_coordinates():
    backend = FakeYuNet(np.array([face_row(10.4, 20.6, 30.0, 40.0)], dtype=np.float32))
    detector = YuNetDetector(backend=backend)

    [box] = detector.detect(FRAME, scale=0.5, min_size=10)

    assert backend.input_sizes == [(200, 100)]  # The model runs on the downscaled frame.
    assert box == FaceBox(21, 41, 60, 80)  # 10.4 * 2 = 20.8 rounds to 21.


def test_yunet_without_faces_returns_empty_list():
    detector = YuNetDetector(backend=FakeYuNet(None))
    assert detector.detect(FRAME) == []


def test_yunet_drops_faces_below_min_size():
    backend = FakeYuNet(np.array([face_row(0, 0, 12, 12), face_row(50, 50, 40, 40)], dtype=np.float32))
    detector = YuNetDetector(backend=backend)

    assert detector.detect(FRAME, scale=1.0, min_size=30) == [FaceBox(50, 50, 40, 40)]


def test_yunet_requires_a_model_or_backend():
    with pytest.raises(ValueError):
        YuNetDetector()


def test_yunet_loads_the_onnx_model_from_memory(tmp_path, monkeypatch):
    import cv2

    model = tmp_path / "modelo_ç.onnx"  # Non-ASCII path: OpenCV's own file loader can reject it on Windows.
    model.write_bytes(b"fake-onnx-bytes")
    calls = []

    def fake_create(framework, buffer_model, buffer_config, input_size, score, nms, top_k):
        calls.append((framework, bytes(buffer_model), buffer_config.size, input_size, score, nms, top_k))
        return FakeYuNet(None)

    monkeypatch.setattr(cv2.FaceDetectorYN, "create", fake_create)
    detector = YuNetDetector(model)

    assert calls == [("onnx", b"fake-onnx-bytes", 0, (320, 320), 0.8, 0.3, 50)]
    assert detector.model_path == model
    assert detector.detect(FRAME) == []


def test_create_detector_builds_yunet_when_configured(tmp_path, monkeypatch):
    import cv2

    from app.core.detector import create_detector

    model = tmp_path / "yunet.onnx"
    model.write_bytes(b"x")
    monkeypatch.setattr(cv2.FaceDetectorYN, "create", lambda *args: FakeYuNet(None))

    detector = create_detector(AppConfig(face_detector="yunet", yunet_model_path=model))
    assert isinstance(detector, YuNetDetector)


def test_config_rejects_unknown_detector_and_missing_model(tmp_path):
    with pytest.raises(ValueError):
        AppConfig(face_detector="mtcnn")
    with pytest.raises(ValueError):
        AppConfig(face_detector="yunet", yunet_model_path=None)
    with pytest.raises(ValueError):
        AppConfig(face_detector="yunet", yunet_model_path=tmp_path / "missing.onnx")


def test_yunet_detector_cannot_be_switched_through_the_api(client):
    from tests.conftest import ADMIN

    response = client.patch("/api/v1/config", headers=ADMIN, json={"face_detector": "yunet"})
    assert response.status_code == 422


# Opt-in check against the real model (MIT, opencv/opencv_zoo, face_detection_yunet_2023mar.onnx).
# Not downloaded by the test suite: set EMOTION_YUNET_MODEL to a local copy to run it.
REAL_MODEL = os.getenv("EMOTION_YUNET_MODEL")
DEMO_VIDEO = Path(__file__).resolve().parent.parent / "Face-emotions.mp4"


@pytest.mark.skipif(not REAL_MODEL, reason="EMOTION_YUNET_MODEL not set")
def test_real_yunet_model_finds_a_face_in_the_demo_video():
    import cv2

    if not DEMO_VIDEO.exists():
        pytest.skip("demo video not present")
    capture = cv2.VideoCapture(str(DEMO_VIDEO))
    ok, frame = capture.read()
    capture.release()
    assert ok

    detector = YuNetDetector(Path(REAL_MODEL))
    assert detector.detect(np.zeros((120, 160, 3), dtype=np.uint8)) == []
    assert detector.detect(frame, scale=1.0, min_size=30)


def test_create_detector_defaults_to_haar():
    from app.core.detector import FaceDetector, create_detector

    assert isinstance(create_detector(AppConfig()), FaceDetector)
