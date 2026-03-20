import pytest

from config import AppConfig


def test_app_config_normalizes_labels_and_aliases():
    config = AppConfig(
        target_labels=("Cup", "Cell Phone"),
        ignored_scene_labels=frozenset({" Person "}),
        default_target_label=" Phone ",
        label_aliases={" Phone ": "Cell Phone"},
    )

    assert config.target_labels == ("cup", "cell phone")
    assert config.ignored_scene_labels == frozenset({"person"})
    assert config.default_target_label == "cell phone"
    assert config.normalize_target("PHONE") == "cell phone"


def test_app_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="detection_confidence"):
        AppConfig(detection_confidence=0)

    with pytest.raises(ValueError, match="speech_input_seconds"):
        AppConfig(speech_input_seconds=0)

    with pytest.raises(ValueError, match="tts_provider"):
        AppConfig(tts_provider="unsupported")
