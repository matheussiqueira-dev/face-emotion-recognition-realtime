import logging
import time
from typing import Optional

import numpy as np

from .models import EmotionResult

DEEPFACE_IMPORT_ERROR: Optional[Exception] = None
try:
    from deepface import DeepFace

    HAS_DEEPFACE = True
except Exception as import_error:
    # Not only ImportError: when a TensorFlow native library is blocked or missing (e.g. Windows
    # Application Control), TensorFlow raises its own errors such as NotFoundError during import.
    HAS_DEEPFACE = False
    DEEPFACE_IMPORT_ERROR = import_error

LOGGER = logging.getLogger(__name__)


class EmotionAnalyzer:
    """Analyzes facial expressions to detect emotions."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and HAS_DEEPFACE
        self.last_error_code: str | None = "model_import_failed" if enabled and not HAS_DEEPFACE else None
        if not HAS_DEEPFACE and enabled:
            LOGGER.error(
                "DeepFace could not be imported (%s). Emotion analysis will be disabled.",
                DEEPFACE_IMPORT_ERROR,
            )

        if self.enabled:
            self.enabled = self._warmup()

    def analyze(self, face_img: np.ndarray) -> Optional[EmotionResult]:
        if not self.enabled or face_img.size == 0:
            return None

        try:
            # DeepFace.analyze returns a list of dictionaries if multiple faces are detected,
            # but we pass a cropped single face.
            results = DeepFace.analyze(
                img_path=face_img,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="skip",
                silent=True,
            )

            if not results:
                self.last_error_code = "empty_model_result"
                return None

            result = results[0] if isinstance(results, list) else results

            emotion = EmotionResult(
                dominant=result["dominant_emotion"],
                # DeepFace returns numpy.float32 values, which are not JSON serializable.
                scores={name: float(score) for name, score in result["emotion"].items()},
                timestamp=time.time(),
            )
            self.last_error_code = None
            return emotion
        except Exception as e:
            self.last_error_code = "inference_failed"
            LOGGER.warning("Emotion analysis failed: %s", e)
            return None

    def _warmup(self) -> bool:
        """Warm up the model with a dummy image to avoid lag on first detection."""
        dummy = np.zeros((48, 48, 3), dtype=np.uint8)
        try:
            DeepFace.analyze(
                dummy, actions=["emotion"], enforce_detection=False, detector_backend="skip", silent=True
            )
            LOGGER.info("Emotion model warmed up successfully.")
            return True
        except Exception as e:
            self.last_error_code = "model_warmup_failed"
            LOGGER.warning(f"Model warmup failed: {e}")
            return False
