import time
import logging
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass

try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
except ImportError:
    HAS_DEEPFACE = False

LOGGER = logging.getLogger(__name__)

@dataclass
class EmotionResult:
    dominant: str
    scores: Dict[str, float]
    timestamp: float

class EmotionAnalyzer:
    """Analyzes facial expressions to detect emotions."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and HAS_DEEPFACE
        if not HAS_DEEPFACE and enabled:
            LOGGER.error("DeepFace is not installed. Emotion analysis will be disabled.")
        
        if self.enabled:
            self._warmup()

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
                silent=True
            )
            
            if not results:
                return None
                
            result = results[0] if isinstance(results, list) else results
            
            return EmotionResult(
                dominant=result["dominant_emotion"],
                scores=result["emotion"],
                timestamp=time.time()
            )
        except Exception as e:
            LOGGER.debug(f"Emotion analysis failed: {e}")
            return None

    def _warmup(self):
        """Warm up the model with a dummy image to avoid lag on first detection."""
        dummy = np.zeros((48, 48, 3), dtype=np.uint8)
        try:
            DeepFace.analyze(dummy, actions=["emotion"], enforce_detection=False, silent=True)
            LOGGER.info("Emotion model warmed up successfully.")
        except Exception as e:
            LOGGER.warning(f"Model warmup failed: {e}")
