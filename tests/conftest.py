from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from app.backend.analysis import BrowserAnalysisService
from app.backend.api import create_app
from app.core.config import AppConfig
from app.core.models import EmotionResult, FaceBox
from app.core.pipeline import FramePipeline

READ = {"Authorization": "Bearer read-token"}
ADMIN = {"Authorization": "Bearer admin-token"}


class FakeDetector:
    def __init__(self, boxes: list[FaceBox] | None = None):
        self.boxes = boxes if boxes is not None else [FaceBox(10, 10, 40, 40)]

    def detect(self, frame, scale=1.0, scale_factor=1.1, min_neighbors=5, min_size=30):
        return list(self.boxes)


@dataclass
class FakeAnalyzer:
    dominant: str = "happy"
    score: float = 90.0
    calls: int = 0
    results: list[EmotionResult] = field(default_factory=list)

    def analyze(self, face_img):
        self.calls += 1
        return EmotionResult(
            dominant=self.dominant,
            scores={self.dominant: self.score, "neutral": 100 - self.score},
            timestamp=0.0,
        )


def make_config(**overrides) -> AppConfig:
    values = {"api_read_token": "read-token", "api_admin_token": "admin-token", "video_source": "missing.mp4"}
    values.update(overrides)
    return AppConfig(**values)


def fake_pipeline_factory(config: AppConfig):
    return lambda: FramePipeline(config, detector=FakeDetector(), analyzer=FakeAnalyzer())


@pytest.fixture()
def config() -> AppConfig:
    return make_config()


@pytest.fixture()
def client(config):
    app = create_app(config, analysis_service=BrowserAnalysisService(config, fake_pipeline_factory(config)))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def open_client():
    """App with auth disabled (local/demo mode)."""
    config = make_config(api_read_token=None, api_admin_token=None)
    app = create_app(config, analysis_service=BrowserAnalysisService(config, fake_pipeline_factory(config)))
    with TestClient(app) as test_client:
        yield test_client
