import os
import subprocess
import tempfile
import time
from pathlib import Path

import requests


class AudioGuide:
    def __init__(self, cooldown_sec: float = 1.2, enabled: bool = True, provider: str = "say", minimax_voice_id: str = "", minimax_model: str = "speech-02-hd"):
        self.cooldown_sec = cooldown_sec
        self.enabled = enabled
        self.provider = provider
        self.minimax_voice_id = minimax_voice_id or os.environ.get("MINIMAX_VOICE_ID", "")
        self.minimax_model = minimax_model
        self.last_message = None
        self.last_time = 0.0

    def _speak_say(self, message: str):
        subprocess.Popen(["say", message])

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
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        audio_hex = data.get("data", {}).get("audio") or data.get("audio")
        audio_url = data.get("data", {}).get("audio_url") or data.get("audio_url")

        if audio_hex:
            audio_bytes = bytes.fromhex(audio_hex)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                f.write(audio_bytes)
                tmp_path = f.name
            subprocess.Popen(["afplay", tmp_path])
            return

        if audio_url:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                tmp_path = f.name
            audio_resp = requests.get(audio_url, timeout=30)
            audio_resp.raise_for_status()
            Path(tmp_path).write_bytes(audio_resp.content)
            subprocess.Popen(["afplay", tmp_path])
            return

        raise RuntimeError(f"MiniMax response missing audio: {data}")

    def speak(self, message: str):
        now = time.time()
        if message == self.last_message and (now - self.last_time) < self.cooldown_sec:
            return
        self.last_message = message
        self.last_time = now
        print(f"[GUIDE] {message}")
        if not self.enabled:
            return
        try:
            if self.provider == "minimax":
                self._speak_minimax(message)
            else:
                self._speak_say(message)
        except Exception as e:
            print(f"[AUDIO ERROR] {e} | fallback to say")
            try:
                self._speak_say(message)
            except Exception as e2:
                print(f"[AUDIO FALLBACK ERROR] {e2}")
