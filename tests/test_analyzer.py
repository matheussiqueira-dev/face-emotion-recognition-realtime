import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def load_analyzer(monkeypatch, engine):
    """Load the adapter with a fake engine; unit tests never import TensorFlow/download weights."""
    monkeypatch.setitem(sys.modules, "deepface", SimpleNamespace(DeepFace=engine))
    spec = importlib.util.spec_from_file_location("app.core._test_analyzer", Path("app/core/analyzer.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EmotionAnalyzer


def test_failed_warmup_is_exposed_as_disabled_analysis(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("Model unavailable")

    analyzer = load_analyzer(monkeypatch, SimpleNamespace(analyze=fail))()
    assert analyzer.enabled is False
    assert analyzer.last_error_code == "model_warmup_failed"


def test_warmup_uses_cropped_face_contract(monkeypatch):
    calls = []

    def analyze(*args, **kwargs):
        calls.append(kwargs)
        return []

    analyzer = load_analyzer(monkeypatch, SimpleNamespace(analyze=analyze))()
    assert analyzer.enabled
    assert calls[0]["detector_backend"] == "skip"
