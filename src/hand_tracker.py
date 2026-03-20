import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace


logger = logging.getLogger(__name__)
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


class HandTracker:
    def __init__(self, model_path: str = "hand_landmarker.task"):
        self.hands = None
        self.drawer = None
        self.hand_connections = None
        self.available = False
        self.error_message = ""
        self.model_path = model_path
        self.backend = "unavailable"
        self._last_timestamp_ms = 0
        try:
            import mediapipe as mp
            self.mp = mp
            if hasattr(mp, "solutions"):
                self._init_legacy_hands(mp)
            else:
                self._init_task_hands(mp)
        except Exception as e:
            self.error_message = f"MediaPipe not ready: {e}"
            logger.warning(self.error_message)

    def _resolve_model_path(self):
        candidate = Path(self.model_path)
        if candidate.is_absolute() and candidate.exists():
            return candidate

        project_root = Path(__file__).resolve().parent.parent
        src_root = Path(__file__).resolve().parent
        for base_dir in (project_root, src_root):
            resolved = base_dir / candidate
            if resolved.exists():
                return resolved
        return None

    def _init_legacy_hands(self, mp):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self.drawer = mp.solutions.drawing_utils
        self.hand_connections = mp.solutions.hands.HAND_CONNECTIONS
        self.available = True
        self.backend = "legacy"

    def _init_task_hands(self, mp):
        model_path = self._resolve_model_path()
        if model_path is None:
            version = getattr(mp, "__version__", "unknown")
            self.error_message = (
                f"MediaPipe {version} requires a Hand Landmarker task model, but "
                f"'{self.model_path}' was not found. Download the official model from "
                f"{HAND_LANDMARKER_MODEL_URL} and place it in the project root."
            )
            logger.warning(self.error_message)
            return

        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self.hands = vision.HandLandmarker.create_from_options(options)
        self.hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
        self.available = True
        self.backend = "tasks"

    def _next_timestamp_ms(self):
        timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    @staticmethod
    def _wrap_landmarks(landmarks):
        return SimpleNamespace(landmark=list(landmarks))

    def detect(self, frame_bgr):
        if self.hands is None:
            return None, None
        try:
            frame_rgb = frame_bgr[:, :, ::-1]
            if self.backend == "tasks":
                image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb.copy())
                results = self.hands.detect_for_video(image, self._next_timestamp_ms())
                if not results.hand_landmarks:
                    return None, None
                hand_landmarks = self._wrap_landmarks(results.hand_landmarks[0])
            else:
                results = self.hands.process(frame_rgb)
                if not results.multi_hand_landmarks:
                    return None, None
                hand_landmarks = results.multi_hand_landmarks[0]
            h, w = frame_bgr.shape[:2]
            coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            center_x = sum(p[0] for p in coords) // len(coords)
            center_y = sum(p[1] for p in coords) // len(coords)
            return (center_x, center_y), hand_landmarks
        except Exception as e:
            logger.warning("Hand tracking failed: %s", e)
            return None, None

    def close(self):
        if self.hands is None:
            return
        close_fn = getattr(self.hands, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                logger.debug("Failed to close hand tracker cleanly", exc_info=True)
        self.hands = None
