from types import SimpleNamespace

import numpy as np

import speech_input as speech_input_module
from config import AppConfig


class DummyWhisperModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, language, fp16):
        self.calls.append((audio, language, fp16))
        return {"text": "  fallback transcript  "}


class DummyWhisperModule:
    def __init__(self, model):
        self.model = model
        self.loaded_models = []

    def load_model(self, model_name):
        self.loaded_models.append(model_name)
        return self.model


class DummyTemporaryDirectory:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def test_speech_input_uses_python_backend_when_cli_is_missing(monkeypatch):
    model = DummyWhisperModel()
    whisper_module = DummyWhisperModule(model)

    monkeypatch.setattr(speech_input_module.SpeechInput, "_has_command", staticmethod(lambda command: False))
    monkeypatch.setattr(
        speech_input_module.SpeechInput,
        "_import_sounddevice",
        lambda self: (SimpleNamespace(), None),
    )

    def fake_import_whisper(self):
        self._whisper_module = whisper_module
        return None

    monkeypatch.setattr(
        speech_input_module.SpeechInput,
        "_import_whisper_module",
        fake_import_whisper,
    )

    speech_input = speech_input_module.SpeechInput()

    assert speech_input.is_available() is True
    assert speech_input.uses_python_backend is True


def test_speech_input_falls_back_to_python_when_cli_fails(monkeypatch):
    model = DummyWhisperModel()
    whisper_module = DummyWhisperModule(model)
    frames = np.array([[0], [16384], [-16384]], dtype=np.int16)

    monkeypatch.setattr(
        speech_input_module.SpeechInput,
        "_has_command",
        staticmethod(lambda command: command == "whisper"),
    )
    monkeypatch.setattr(
        speech_input_module.SpeechInput,
        "_import_sounddevice",
        lambda self: (SimpleNamespace(), None),
    )

    def fake_import_whisper(self):
        self._whisper_module = whisper_module
        return None

    monkeypatch.setattr(
        speech_input_module.SpeechInput,
        "_import_whisper_module",
        fake_import_whisper,
    )
    monkeypatch.setattr(
        speech_input_module.SpeechInput,
        "_record_audio",
        lambda self, sounddevice_module: frames,
    )
    monkeypatch.setattr(
        speech_input_module.SpeechInput,
        "_write_wav_file",
        lambda self, wav_path, audio_frames: None,
    )
    monkeypatch.setattr(
        speech_input_module.SpeechInput,
        "_transcribe_with_cli",
        lambda self, wav_path, txt_path: (None, "whisper failed: ffmpeg not found"),
    )
    monkeypatch.setattr(
        speech_input_module.tempfile,
        "TemporaryDirectory",
        lambda: DummyTemporaryDirectory("C:\\temp\\speech-input-test"),
    )

    speech_input = speech_input_module.SpeechInput()
    text, err = speech_input.listen_once()

    assert err is None
    assert text == "fallback transcript"
    assert speech_input.uses_python_backend is True
    assert whisper_module.loaded_models == ["base"]
    assert model.calls[0][1:] == ("en", False)


def test_speech_input_prefers_python_backend_when_ffmpeg_is_missing(monkeypatch):
    model = DummyWhisperModel()
    whisper_module = DummyWhisperModule(model)

    monkeypatch.setattr(
        speech_input_module.SpeechInput,
        "_has_command",
        staticmethod(lambda command: command == "whisper"),
    )
    monkeypatch.setattr(
        speech_input_module.SpeechInput,
        "_import_sounddevice",
        lambda self: (SimpleNamespace(), None),
    )

    def fake_import_whisper(self):
        self._whisper_module = whisper_module
        return None

    monkeypatch.setattr(
        speech_input_module.SpeechInput,
        "_import_whisper_module",
        fake_import_whisper,
    )

    speech_input = speech_input_module.SpeechInput()

    assert speech_input.is_available() is True
    assert speech_input.uses_python_backend is True


def test_speech_input_reports_friendly_retry_message_for_empty_transcript(monkeypatch):
    model = DummyWhisperModel()
    whisper_module = DummyWhisperModule(model)
    frames = np.array([[0], [16384], [-16384]], dtype=np.int16)

    monkeypatch.setattr(
        speech_input_module.SpeechInput,
        "_has_command",
        staticmethod(lambda command: command in {"whisper", "ffmpeg"}),
    )
    monkeypatch.setattr(
        speech_input_module.SpeechInput,
        "_import_sounddevice",
        lambda self: (SimpleNamespace(), None),
    )

    def fake_import_whisper(self):
        self._whisper_module = whisper_module
        return None

    monkeypatch.setattr(
        speech_input_module.SpeechInput,
        "_import_whisper_module",
        fake_import_whisper,
    )
    monkeypatch.setattr(
        speech_input_module.SpeechInput,
        "_record_audio",
        lambda self, sounddevice_module: frames,
    )
    monkeypatch.setattr(
        speech_input_module.SpeechInput,
        "_write_wav_file",
        lambda self, wav_path, audio_frames: None,
    )
    monkeypatch.setattr(
        speech_input_module.SpeechInput,
        "_transcribe_with_cli",
        lambda self, wav_path, txt_path: (None, speech_input_module.NO_SPEECH_RETRY_MESSAGE),
    )
    monkeypatch.setattr(
        speech_input_module.SpeechInput,
        "_transcribe_with_python",
        lambda self, audio_frames: (None, speech_input_module.NO_SPEECH_RETRY_MESSAGE),
    )
    monkeypatch.setattr(
        speech_input_module.tempfile,
        "TemporaryDirectory",
        lambda: DummyTemporaryDirectory("C:\\temp\\speech-input-empty"),
    )

    speech_input = speech_input_module.SpeechInput()
    text, err = speech_input.listen_once()

    assert text is None
    assert err == speech_input_module.NO_SPEECH_RETRY_MESSAGE


def test_pipeline_uses_longer_default_speech_input_window():
    config = AppConfig()

    assert config.speech_input_seconds == 6.0
