from types import SimpleNamespace

import numpy as np
import pytest

import pipeline as pipeline_module
from config import AppConfig
from nlu import IntentResult


class DummyAudio:
    def __init__(self, *args, **kwargs):
        self.messages = []

    def speak(self, message):
        self.messages.append(message)

    def close(self):
        return None


class DummyDetector:
    backend_name = "dummy_detector"

    def __init__(self, *args, **kwargs):
        self.detections = []

    def detect_all(self, frame, restrict_to_allowed=True):
        return list(self.detections)


class DummyHandTracker:
    backend = "dummy_hand_tracker"

    def __init__(self, *args, **kwargs):
        self.center = None
        self.landmarks = None
        self.drawer = None
        self.hand_connections = None
        self.available = True
        self.error_message = ""

    def detect(self, frame):
        return self.center, self.landmarks

    def close(self):
        return None


class DummyDepthEstimator:
    backend_name = "dummy_depth"

    def update(self, frame):
        pass

    def estimate_distance_cm(self, point_a, point_b, frame_shape, cm_per_pixel=0.18):
        import math
        dx = point_b[0] - point_a[0]
        dy = point_b[1] - point_a[1]
        return round(math.hypot(dx, dy) * cm_per_pixel, 1)


class DummySpeechInput:
    def __init__(self, *args, **kwargs):
        self.response = ("hello", None)
        self.available = True

    def listen_once(self):
        return self.response

    def is_available(self):
        return self.available


@pytest.fixture
def pipeline_factory(monkeypatch):
    monkeypatch.setattr(pipeline_module, "AudioGuide", DummyAudio)
    monkeypatch.setattr(pipeline_module, "create_detector", lambda *a, **kw: DummyDetector())
    monkeypatch.setattr(pipeline_module, "create_hand_tracker", lambda *a, **kw: DummyHandTracker())
    monkeypatch.setattr(pipeline_module, "create_depth_estimator", lambda *a, **kw: DummyDepthEstimator())
    monkeypatch.setattr(pipeline_module, "SpeechInput", DummySpeechInput)
    monkeypatch.setattr(pipeline_module, "pick", lambda options: options[0])

    def make_pipeline(**config_kwargs):
        config = AppConfig(use_tts=False, **config_kwargs)
        return pipeline_module.QuerySnapshotPipeline(config)

    return make_pipeline


def make_landmarks(*points):
    return SimpleNamespace(
        landmark=[SimpleNamespace(x=x, y=y) for x, y in points],
    )


def test_update_live_state_filters_ignored_labels_and_tracks_scene(pipeline_factory):
    pipeline = pipeline_factory()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pipeline.detector.detections = [
        {"label": "person", "confidence": 0.99, "bbox": (0, 0, 10, 10), "center": (5, 5), "area": 100},
        {"label": "cup", "confidence": 0.90, "bbox": (10, 10, 30, 30), "center": (20, 20), "area": 400},
        {"label": "cup", "confidence": 0.70, "bbox": (12, 12, 26, 26), "center": (19, 19), "area": 196},
        {"label": "pen", "confidence": 0.80, "bbox": (40, 40, 60, 60), "center": (50, 50), "area": 400},
    ]
    pipeline.hand_tracker.center = (25, 25)
    pipeline.hand_tracker.landmarks = make_landmarks((0.1, 0.2), (0.2, 0.3), (0.3, 0.2))

    pipeline.update_live_state(frame)

    assert [d["label"] for d in pipeline.live_detections] == ["cup", "pen", "cup"]
    assert pipeline.last_scene_objects == ["cup", "pen"]
    assert pipeline.last_raw_detection_text == "cup(0.90), pen(0.80), cup(0.70)"
    assert pipeline.live_hand_box == (10, 20, 30, 30)
    assert pipeline.last_snapshot is frame


def test_answer_direction_requires_target_when_none_selected(pipeline_factory):
    pipeline = pipeline_factory()

    result = pipeline.answer_direction()

    assert result == "Please tell me which object you want first."
    assert pipeline.audio.messages[-1] == result


def test_answer_direction_handles_missing_target_and_missing_hand(pipeline_factory):
    pipeline = pipeline_factory()

    not_found = pipeline.answer_direction("cup")
    assert not_found == "I can't find the cup."

    pipeline.live_detections = [
        {"label": "cup", "confidence": 0.9, "bbox": (80, 40, 120, 80), "center": (100, 60), "area": 1600},
    ]
    show_hand = pipeline.answer_direction("cup")

    assert show_hand == "I found the cup. Show your hand in the camera so I can guide you."


def test_answer_direction_reports_unavailable_hand_tracking(pipeline_factory):
    pipeline = pipeline_factory()
    pipeline.hand_tracker.available = False
    pipeline.hand_tracker.error_message = "mediapipe unavailable"
    pipeline.live_detections = [
        {"label": "cup", "confidence": 0.9, "bbox": (80, 40, 120, 80), "center": (100, 60), "area": 1600},
    ]

    result = pipeline.answer_direction("cup")

    assert result == (
        "Hand tracking is unavailable right now, so I cannot guide your hand position yet. "
        "Please check the MediaPipe setup and try again."
    )


def test_answer_direction_returns_guidance_with_relative_position_details(pipeline_factory):
    pipeline = pipeline_factory(x_threshold=20, y_threshold=20)
    pipeline.live_detections = [
        {"label": "cup", "confidence": 0.9, "bbox": (80, 40, 120, 80), "center": (100, 60), "area": 1600},
    ]
    pipeline.live_hand_center = (0, 60)

    result = pipeline.answer_direction("mug")

    assert "cup" in result
    assert "9 o'clock" in result
    assert "move your hand left" in result


def test_answer_grasp_status_confirms_when_close_enough(pipeline_factory):
    pipeline = pipeline_factory()
    pipeline.live_detections = [
        {"label": "cup", "confidence": 0.9, "bbox": (40, 40, 80, 80), "center": (60, 60), "area": 1600},
    ]
    pipeline.live_hand_center = (20, 60)
    pipeline.live_hand_box = (30, 30, 90, 90)

    result = pipeline.answer_grasp_status("cup")

    assert result == "Yes, you are holding the cup. What would you like to do next?"


def test_answer_grasp_status_returns_guidance_when_not_confirmed(pipeline_factory):
    pipeline = pipeline_factory(x_threshold=20, y_threshold=20, near_threshold=40)
    pipeline.live_detections = [
        {"label": "pen", "confidence": 0.9, "bbox": (140, 40, 180, 80), "center": (160, 60), "area": 1600},
    ]
    pipeline.live_hand_center = (0, 60)
    pipeline.live_hand_box = (0, 0, 20, 20)

    result = pipeline.answer_grasp_status("pen")

    assert result.startswith("Not yet.")
    assert "9 o'clock" in result


def test_handle_voice_text_routes_intents_and_unknown(monkeypatch, pipeline_factory):
    pipeline = pipeline_factory()
    pipeline.last_scene_objects = ["cup", "pen"]
    calls = []

    monkeypatch.setattr(
        pipeline.nlu,
        "parse",
        lambda text: IntentResult("ask_direction", target="cup", raw_text=text),
    )
    monkeypatch.setattr(
        pipeline,
        "answer_direction",
        lambda target=None: calls.append(target) or "direction response",
    )

    assert pipeline.handle_voice_text("Where is the cup?") == "direction response"
    assert calls == ["cup"]

    monkeypatch.setattr(
        pipeline.nlu,
        "parse",
        lambda text: IntentResult("unknown", raw_text=text),
    )

    assert pipeline.handle_voice_text("Tell me a joke") == (
        "Sorry, I didn't understand that. You can ask what objects are in front of you, "
        "where an object is, or whether you got it."
    )


def test_listen_for_command_surfaces_voice_errors(pipeline_factory):
    pipeline = pipeline_factory()
    pipeline.speech_input.response = (None, "microphone unavailable")

    text, err = pipeline.listen_for_command()

    assert text is None
    assert err == "Voice input failed: microphone unavailable"
    assert pipeline.last_voice_status == "voice error: microphone unavailable"


def test_listen_for_command_returns_raw_error_when_voice_backend_is_unavailable(pipeline_factory):
    pipeline = pipeline_factory()
    pipeline.speech_input.available = False
    pipeline.speech_input.response = (None, "whisper CLI not found; press k to type")

    text, err = pipeline.listen_for_command()

    assert text is None
    assert err == "whisper CLI not found; press k to type"
    assert pipeline.audio.messages == ["I'm listening."]
