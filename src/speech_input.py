import os
import subprocess
import tempfile
import wave

import numpy as np


class SpeechInput:
    def __init__(self, seconds: float = 4.0, sample_rate: int = 16000):
        self.seconds = seconds
        self.sample_rate = sample_rate

    def listen_once(self):
        try:
            import sounddevice as sd
        except Exception as e:
            return None, f"sounddevice not available: {e}"

        with tempfile.TemporaryDirectory() as td:
            wav_path = os.path.join(td, "speech.wav")
            txt_path = os.path.join(td, "speech.txt")
            try:
                frames = sd.rec(int(self.seconds * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype="int16")
                sd.wait()
                with wave.open(wav_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(np.asarray(frames).tobytes())
            except Exception as e:
                return None, f"recording failed: {e}"

            try:
                cmd = ["whisper", wav_path, "--model", "base", "--language", "en", "--output_format", "txt", "--output_dir", td]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                if not os.path.exists(txt_path):
                    return None, "whisper did not produce transcript"
                text = open(txt_path, "r", encoding="utf-8").read().strip()
                return text, None
            except subprocess.CalledProcessError as e:
                return None, f"whisper failed: {e.stderr[-300:] if e.stderr else str(e)}"
            except Exception as e:
                return None, f"speech parsing failed: {e}"
