import logging
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace


logger = logging.getLogger(__name__)
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# COCO-17 pose keypoint indices for wrists
_COCO_LEFT_WRIST = 9
_COCO_RIGHT_WRIST = 10


def _make_landmark(x_norm: float, y_norm: float):
    return SimpleNamespace(x=x_norm, y=y_norm, z=0.0)


class BaseHandTracker(ABC):
    """Common interface for all hand tracking backends."""

    available: bool = False
    error_message: str = ""
    hand_connections = None
    backend: str = "unavailable"

    @abstractmethod
    def detect(self, frame_bgr) -> tuple[tuple | None, object | None]:
        """Return (hand_center_px, hand_landmarks) or (None, None)."""

    def close(self):
        pass


class MediaPipeHandTracker(BaseHandTracker):
    """Hand tracker using MediaPipe Hands (legacy or Tasks API)."""

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
        self.backend = "mediapipe_legacy"

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
        self.backend = "mediapipe_tasks"

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
            if "tasks" in self.backend:
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
            logger.warning("MediaPipe hand tracking failed: %s", e)
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


class YOLOPoseHandTracker(BaseHandTracker):
    """Hand tracker using YOLO-Pose: extracts wrist keypoints from human pose estimation."""

    def __init__(self, model_name: str = "yolov8n-pose.pt"):
        self.model = None
        self.hand_connections = None  # no skeleton connections for wrist-only output
        self.available = False
        self.error_message = ""
        self.backend = f"yolo_pose:{model_name}"
        model_path = Path(model_name)
        if not model_path.is_absolute():
            local = Path(__file__).resolve().parent / model_path
            if local.exists():
                model_path = local
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(model_path))
            self.available = True
            logger.info("YOLO-Pose loaded: %s", model_name)
        except Exception as e:
            self.error_message = f"YOLO-Pose not ready: {e}"
            logger.warning(self.error_message)

    def detect(self, frame_bgr):
        if self.model is None:
            return None, None
        h, w = frame_bgr.shape[:2]
        try:
            results = self.model(frame_bgr, verbose=False)
            for result in results:
                if result.keypoints is None or len(result.keypoints) == 0:
                    continue
                # pick the most confident person
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                best_idx = int(boxes.conf.argmax())
                kp_xy = result.keypoints.xy[best_idx]   # [17, 2] pixel coords
                kp_conf = result.keypoints.conf[best_idx]  # [17]

                l_conf = float(kp_conf[_COCO_LEFT_WRIST])
                r_conf = float(kp_conf[_COCO_RIGHT_WRIST])
                wrist_idx = _COCO_LEFT_WRIST if l_conf >= r_conf else _COCO_RIGHT_WRIST

                wx = float(kp_xy[wrist_idx, 0])
                wy = float(kp_xy[wrist_idx, 1])
                if wx == 0 and wy == 0:
                    continue

                center = (int(wx), int(wy))
                lm = _make_landmark(wx / w, wy / h)
                hand_landmarks = SimpleNamespace(landmark=[lm])
                return center, hand_landmarks
        except Exception as e:
            logger.warning("YOLO-Pose hand tracking failed: %s", e)
        return None, None


class HolisticHandTracker(BaseHandTracker):
    """Hand tracker using MediaPipe HandLandmarker Tasks API (no pose dependency).

    Uses the newer Tasks API for hand landmark detection, providing similar
    functionality to the legacy mp.solutions.holistic when pose model is unavailable.
    """

    def __init__(self):
        self.hand_landmarker = None
        self.hand_connections = None
        self.available = False
        self.error_message = ""
        self.backend = "mediapipe_holistic"
        self._last_timestamp_ms = 0
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
            self.mp = mp
            self._vision = vision

            # Load HandLandmarker using Tasks API
            model_path = Path("hand_landmarker.task")
            if not model_path.is_absolute():
                project_root = Path(__file__).resolve().parent.parent
                for base_dir in (project_root, Path(__file__).resolve().parent):
                    resolved = base_dir / model_path
                    if resolved.exists():
                        model_path = resolved
                        break
            hand_options = vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.4,
                min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.4,
            )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
            self.hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
            self.available = True
            logger.info("MediaPipe Holistic (HandLandmarker Tasks API) loaded")
        except Exception as e:
            self.error_message = f"MediaPipe Holistic not ready: {e}"
            logger.warning(self.error_message)

    def _next_timestamp_ms(self):
        timestamp_ms = int(__import__("time").monotonic() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def detect(self, frame_bgr):
        if self.hand_landmarker is None:
            return None, None
        try:
            frame_rgb = frame_bgr[:, :, ::-1]
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb.copy())
            hand_result = self.hand_landmarker.detect_for_video(image, self._next_timestamp_ms())

            if not hand_result.hand_landmarks:
                return None, None

            raw = hand_result.hand_landmarks[0]
            # Tasks API returns List[NormalizedLandmark]; legacy returns NormalizedLandmarkList with .landmark
            h, w = frame_bgr.shape[:2]
            if hasattr(raw, 'landmark'):
                landmarks = raw.landmark
            else:
                landmarks = raw  # already a plain list
            coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            center_x = sum(p[0] for p in coords) // len(coords)
            center_y = sum(p[1] for p in coords) // len(coords)
            # Wrap in SimpleNamespace so .landmark is available, matching MediaPipeHandTracker
            hand_landmarks = SimpleNamespace(landmark=landmarks)
            return (center_x, center_y), hand_landmarks
        except Exception as e:
            logger.warning("Holistic hand tracking failed: %s", e)
            return None, None

    def close(self):
        if self.hand_landmarker:
            try:
                self.hand_landmarker.close()
            except Exception:
                pass
            self.hand_landmarker = None


# Backward-compatible alias
HandTracker = MediaPipeHandTracker

_HAND_TRACKER_CYCLE = ["mediapipe", "yolo_pose", "holistic"]


def create_hand_tracker(backend: str, model_path: str = "hand_landmarker.task") -> BaseHandTracker:
    """Factory: create a hand tracker by backend name."""
    if backend == "yolo_pose":
        return YOLOPoseHandTracker()
    if backend == "holistic":
        return HolisticHandTracker()
    return MediaPipeHandTracker(model_path)


def next_hand_tracker_backend(current: str) -> str:
    """Return the next backend name in the cycle.

    Strips the colon suffix from 'current' (e.g. 'mediapipe_tasks' -> 'mediapipe',
    'yolo_pose:yolov8n-pose.pt' -> 'yolo_pose', 'mediapipe_holistic' -> 'mediapipe')
    to match entries in _HAND_TRACKER_CYCLE.
    """
    base = current.split(":")[0].replace("mediapipe_holistic", "holistic")
    if base not in _HAND_TRACKER_CYCLE:
        base = "mediapipe"
    try:
        idx = _HAND_TRACKER_CYCLE.index(base)
    except ValueError:
        idx = 0
    return _HAND_TRACKER_CYCLE[(idx + 1) % len(_HAND_TRACKER_CYCLE)]
