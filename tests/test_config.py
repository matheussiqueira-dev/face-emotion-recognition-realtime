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


def test_rejected_update_is_atomic():
    config = AppConfig()
    with pytest.raises(ValueError):
        config.update_from_dict({"emotion_interval": 0.9, "max_fps": 0})
    assert config.max_fps == 30
    assert config.emotion_interval == 0.5


def test_auth_enabled_reflects_tokens():
    assert AppConfig(api_read_token=None, api_admin_token=None).auth_enabled is False
    assert AppConfig(api_read_token=None, api_admin_token="secret").auth_enabled is True


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_config_rejects_non_finite_timing(value):
    with pytest.raises(ValueError, match="finite"):
        AppConfig(emotion_interval=value)
