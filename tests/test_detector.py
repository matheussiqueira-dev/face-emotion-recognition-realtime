import shutil
from pathlib import Path

import cv2
import numpy as np

from app.core.detector import FaceDetector


def test_detector_loads_cascade_from_unicode_path(tmp_path):
    cascade_path = tmp_path / "programação" / "haarcascade_frontalface_default.xml"
    cascade_path.parent.mkdir()
    shutil.copyfile(
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml",
        cascade_path,
    )

    detector = FaceDetector(cascade_path)

    assert not detector.classifier.empty()
    assert detector.detect(np.zeros((120, 160, 3), dtype=np.uint8)) == []
