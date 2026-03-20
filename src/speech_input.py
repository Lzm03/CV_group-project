import logging
import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np


logger = logging.getLogger(__name__)
EMPTY_TRANSCRIPT_MARKER = "empty transcript"
NO_SPEECH_RETRY_MESSAGE = (
    "I didn't catch any speech. Please start speaking right after the listening prompt, "
    "speak a little louder, and try again."
)


class SpeechInput:
    def __init__(self, seconds: float = 6.0, sample_rate: int = 16000, whisper_model: str = "base", language: str = "en"):
        self.seconds = seconds
        self.sample_rate = sample_rate
        self.whisper_model = whisper_model
        self.language = language
        self._backend = None
        self._whisper_module = None
        self._whisper_model_instance = None
        self._availability_error = self._detect_availability()

    @staticmethod
    def _has_command(command: str) -> bool:
        return shutil.which(command) is not None

    def _import_sounddevice(self):
        try:
            import sounddevice as sd
        except Exception as e:
            return None, e
        return sd, None

    def _import_whisper_module(self):
        if self._whisper_module is not None:
            return None
        try:
            import whisper
        except Exception as e:
            return e
        self._whisper_module = whisper
        return None

    def _detect_availability(self):
        _, sounddevice_error = self._import_sounddevice()
        if sounddevice_error is not None:
            return f"sounddevice not available: {sounddevice_error}"

        whisper_cli_available = self._has_command("whisper")
        ffmpeg_available = self._has_command("ffmpeg")

        if whisper_cli_available and ffmpeg_available:
            self._backend = "cli"
            return None

        whisper_error = self._import_whisper_module()
        if whisper_error is None:
            if whisper_cli_available and not ffmpeg_available:
                logger.info("ffmpeg executable not found; using Python Whisper backend instead of CLI")
            self._backend = "python"
            return None

        return (
            "Whisper CLI with ffmpeg is not available, and openai-whisper is unavailable; "
            "press k to type, or install openai-whisper for voice input"
        )

    def _record_audio(self, sounddevice_module):
        frames = sounddevice_module.rec(
            int(self.seconds * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
        )
        sounddevice_module.wait()
        return np.asarray(frames)

    def _write_wav_file(self, wav_path: str, frames: np.ndarray):
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(np.asarray(frames).tobytes())

    def _transcribe_with_cli(self, wav_path: str, txt_path: str):
        cmd = [
            "whisper",
            wav_path,
            "--model",
            self.whisper_model,
            "--language",
            self.language,
            "--output_format",
            "txt",
            "--output_dir",
            os.path.dirname(txt_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            return None, f"whisper failed: {e.stderr[-300:] if e.stderr else str(e)}"
        if not os.path.exists(txt_path):
            return None, "whisper did not produce transcript"
        with open(txt_path, "r", encoding="utf-8") as transcript_file:
            text = transcript_file.read().strip()
        if not text:
            return None, NO_SPEECH_RETRY_MESSAGE
        return text, None

    def _load_whisper_model(self):
        whisper_error = self._import_whisper_module()
        if whisper_error is not None:
            raise RuntimeError(f"openai-whisper import failed: {whisper_error}") from whisper_error
        if self._whisper_model_instance is None:
            logger.info("Loading Whisper model '%s' for local transcription", self.whisper_model)
            self._whisper_model_instance = self._whisper_module.load_model(self.whisper_model)
        return self._whisper_model_instance

    def _transcribe_with_python(self, frames: np.ndarray):
        try:
            model = self._load_whisper_model()
            audio = np.asarray(frames, dtype=np.float32).reshape(-1) / 32768.0
            result = model.transcribe(audio, language=self.language, fp16=False)
        except Exception as e:
            return None, f"python whisper failed: {e}"

        text = (result.get("text") or "").strip()
        if not text:
            return None, NO_SPEECH_RETRY_MESSAGE
        return text, None

    @staticmethod
    def _is_empty_transcript_error(error_message: str | None) -> bool:
        if not error_message:
            return False
        normalized = error_message.lower()
        return EMPTY_TRANSCRIPT_MARKER in normalized or error_message == NO_SPEECH_RETRY_MESSAGE

    def _combine_transcription_errors(self, primary_error: str, fallback_error: str):
        if self._is_empty_transcript_error(primary_error) and self._is_empty_transcript_error(fallback_error):
            return NO_SPEECH_RETRY_MESSAGE
        if self._is_empty_transcript_error(primary_error) and not fallback_error:
            return NO_SPEECH_RETRY_MESSAGE
        if fallback_error == NO_SPEECH_RETRY_MESSAGE:
            return fallback_error
        return f"{primary_error} | fallback also failed: {fallback_error}"

    def _transcribe_audio(self, frames: np.ndarray, wav_path: str, txt_path: str):
        if self._backend == "python":
            return self._transcribe_with_python(frames)

        if self._backend == "cli":
            text, err = self._transcribe_with_cli(wav_path, txt_path)
            if err is None:
                return text, None

            logger.warning("Whisper CLI failed; attempting Python fallback: %s", err)
            fallback_text, fallback_err = self._transcribe_with_python(frames)
            if fallback_err is None:
                self._backend = "python"
                return fallback_text, None
            return None, self._combine_transcription_errors(err, fallback_err)

        whisper_error = self._import_whisper_module()
        if whisper_error is None:
            self._backend = "python"
            return self._transcribe_with_python(frames)

        if self._has_command("whisper") and self._has_command("ffmpeg"):
            self._backend = "cli"
            return self._transcribe_with_cli(wav_path, txt_path)

        return None, self._availability_error

    @property
    def backend(self):
        return self._backend

    @property
    def uses_cli_backend(self):
        return self._backend == "cli"

    @property
    def uses_python_backend(self):
        return self._backend == "python"

    @property
    def availability_error(self):
        return self._availability_error

    def is_available(self):
        return self._availability_error is None

    def listen_once(self):
        if self._availability_error is not None:
            return None, self._availability_error

        sounddevice_module, sounddevice_error = self._import_sounddevice()
        if sounddevice_error is not None:
            return None, f"sounddevice not available: {sounddevice_error}"

        with tempfile.TemporaryDirectory() as td:
            wav_path = os.path.join(td, "speech.wav")
            txt_path = os.path.join(td, "speech.txt")
            try:
                frames = self._record_audio(sounddevice_module)
                self._write_wav_file(wav_path, frames)
            except Exception as e:
                return None, f"recording failed: {e}"

            try:
                return self._transcribe_audio(frames, wav_path, txt_path)
            except Exception as e:
                logger.warning("Speech parsing failed: %s", e)
                return None, f"speech parsing failed: {e}"
