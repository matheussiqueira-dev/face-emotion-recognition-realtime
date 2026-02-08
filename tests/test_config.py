import pytest

from app.core.config import AppConfig


def test_config_update_accepts_known_fields():
    config = AppConfig()
    changed = config.update_from_dict({"max_fps": 24, "emotion_interval": 0.7})
    assert "max_fps" in changed
    assert "emotion_interval" in changed
    assert config.max_fps == 24
    assert config.emotion_interval == 0.7


def test_config_rejects_unknown_fields():
    config = AppConfig()
    with pytest.raises(ValueError):
        config.update_from_dict({"unknown_field": 1})


def test_config_validation_rejects_invalid_fps():
    config = AppConfig()
    with pytest.raises(ValueError):
        config.update_from_dict({"max_fps": 0})
