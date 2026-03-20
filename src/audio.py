import atexit
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import requests


logger = logging.getLogger(__name__)

TEMP_AUDIO_PREFIX = "cv-guide-"
TEMP_AUDIO_SUFFIX = ".mp3"
TEMP_CLEANUP_ATTEMPTS = 10
TEMP_CLEANUP_DELAY_SEC = 1.0
WINDOWS_PYTTSX3_SCRIPT = (
    "import sys;"
    "import pyttsx3;"
    "engine = pyttsx3.init();"
    "engine.say(sys.argv[1]);"
    "engine.runAndWait();"
    "engine.stop()"
)


class AudioGuide:
    def __init__(self, cooldown_sec: float = 1.2, enabled: bool = True, provider: str = "auto", minimax_voice_id: str = "", minimax_model: str = "speech-02-hd"):
        self.cooldown_sec = cooldown_sec
        self.enabled = enabled
        self.provider = provider
        self.minimax_voice_id = minimax_voice_id or os.environ.get("MINIMAX_VOICE_ID", "")
        self.minimax_model = minimax_model
        self.last_message = None
        self.last_time = 0.0
        self._tts_engine = None
        self._session = requests.Session()
        self._pending_temp_files = set()
        self._cleanup_lock = threading.Lock()
        self._closed = False
        atexit.register(self.close)
        self._cleanup_stale_temp_files()

    def _speak_say(self, message: str):
        if shutil.which("say") is None:
            raise RuntimeError("macOS 'say' command is not available")
        subprocess.Popen(["say", message])

    def _speak_pyttsx3(self, message: str):
        try:
            import pyttsx3
        except Exception as e:
            raise RuntimeError(f"pyttsx3 not available: {e}") from e

        if sys.platform.startswith("win"):
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                subprocess.run(
                    [sys.executable, "-c", WINDOWS_PYTTSX3_SCRIPT, message],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    creationflags=creationflags,
                )
            except subprocess.CalledProcessError as e:
                stderr = (e.stderr or e.stdout or str(e)).strip()
                raise RuntimeError(f"pyttsx3 subprocess failed: {stderr}") from e
            except subprocess.TimeoutExpired as e:
                raise RuntimeError("pyttsx3 subprocess timed out") from e
            return

        if self._tts_engine is None:
            self._tts_engine = pyttsx3.init()
        self._tts_engine.say(message)
        self._tts_engine.runAndWait()

    def _play_audio_file(self, path: str):
        try:
            if sys.platform == "darwin" and shutil.which("afplay") is not None:
                subprocess.run(["afplay", path], check=True)
                return
            if sys.platform.startswith("win"):
                os.startfile(path)
                return
            if shutil.which("xdg-open") is not None:
                subprocess.Popen(["xdg-open", path])
                return
            raise RuntimeError("No supported audio player is available for MiniMax output")
        finally:
            self._schedule_temp_cleanup(path)

    def _cleanup_stale_temp_files(self):
        temp_dir = Path(tempfile.gettempdir())
        cutoff = time.time() - (12 * 60 * 60)
        for stale_file in temp_dir.glob(f"{TEMP_AUDIO_PREFIX}*{TEMP_AUDIO_SUFFIX}"):
            try:
                if stale_file.stat().st_mtime < cutoff:
                    stale_file.unlink(missing_ok=True)
            except OSError:
                logger.debug("Could not remove stale audio file: %s", stale_file, exc_info=True)

    def _delete_temp_file(self, path: str, attempts: int = TEMP_CLEANUP_ATTEMPTS):
        target_path = Path(path)
        for attempt in range(attempts):
            try:
                target_path.unlink(missing_ok=True)
                return
            except PermissionError:
                if attempt == attempts - 1:
                    logger.warning("Audio temp file is still locked and could not be removed: %s", path)
                    return
                time.sleep(TEMP_CLEANUP_DELAY_SEC)
            except OSError:
                logger.debug("Failed to remove audio temp file: %s", path, exc_info=True)
                return

    def _schedule_temp_cleanup(self, path: str):
        with self._cleanup_lock:
            self._pending_temp_files.add(path)

        def cleanup():
            try:
                self._delete_temp_file(path)
            finally:
                with self._cleanup_lock:
                    self._pending_temp_files.discard(path)

        threading.Thread(target=cleanup, daemon=True).start()

    def _write_temp_audio(self, audio_bytes: bytes) -> str:
        with tempfile.NamedTemporaryFile(
            delete=False,
            prefix=TEMP_AUDIO_PREFIX,
            suffix=TEMP_AUDIO_SUFFIX,
        ) as audio_file:
            audio_file.write(audio_bytes)
            return audio_file.name

    def _speak_minimax(self, message: str):
        api_key = os.environ.get("MINIMAX_API_KEY", "")
        group_id = os.environ.get("MINIMAX_GROUP_ID", "")
        if not api_key or not group_id or not self.minimax_voice_id:
            raise RuntimeError("MINIMAX_API_KEY, MINIMAX_GROUP_ID, or voice_id missing")

        url = f"https://api.minimax.chat/v1/t2a_v2?GroupId={group_id}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.minimax_model,
            "text": message,
            "stream": False,
            "voice_setting": {
                "voice_id": self.minimax_voice_id,
                "speed": 1.0,
                "vol": 1.0,
                "pitch": 0,
            },
            "audio_setting": {
                "sample_rate": 32000,
                "bitrate": 128000,
                "format": "mp3",
                "channel": 1,
            },
        }
        r = self._session.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        audio_hex = data.get("data", {}).get("audio") or data.get("audio")
        audio_url = data.get("data", {}).get("audio_url") or data.get("audio_url")

        if audio_hex:
            tmp_path = self._write_temp_audio(bytes.fromhex(audio_hex))
            self._play_audio_file(tmp_path)
            return

        if audio_url:
            audio_resp = self._session.get(audio_url, timeout=30)
            audio_resp.raise_for_status()
            tmp_path = self._write_temp_audio(audio_resp.content)
            self._play_audio_file(tmp_path)
            return

        raise RuntimeError(f"MiniMax response missing audio: {data}")

    def _provider_order(self):
        if self.provider == "auto":
            return ["pyttsx3", "minimax", "say"]
        if self.provider == "minimax":
            return ["minimax", "pyttsx3", "say"]
        if self.provider == "say":
            return ["say", "pyttsx3"]
        if self.provider == "pyttsx3":
            return ["pyttsx3", "say"]
        return [self.provider]

    def speak(self, message: str):
        now = time.time()
        if message == self.last_message and (now - self.last_time) < self.cooldown_sec:
            return
        self.last_message = message
        self.last_time = now
        logger.info("Guide: %s", message)
        if not self.enabled:
            return
        errors = []
        for provider in self._provider_order():
            try:
                if provider == "minimax":
                    self._speak_minimax(message)
                elif provider == "pyttsx3":
                    self._speak_pyttsx3(message)
                elif provider == "say":
                    self._speak_say(message)
                else:
                    raise RuntimeError(f"Unsupported TTS provider: {provider}")
                return
            except Exception as e:
                errors.append(f"{provider}: {e}")
        logger.error("All TTS providers failed | %s", " | ".join(errors))

    def close(self):
        if self._closed:
            return
        self._closed = True

        if self._tts_engine is not None:
            try:
                self._tts_engine.stop()
            except Exception:
                logger.debug("Failed to stop pyttsx3 engine cleanly", exc_info=True)
            self._tts_engine = None

        self._session.close()

        with self._cleanup_lock:
            pending_files = list(self._pending_temp_files)
            self._pending_temp_files.clear()
        for path in pending_files:
            self._delete_temp_file(path, attempts=1)
