import logging
from abc import ABC, abstractmethod
from pathlib import Path


logger = logging.getLogger(__name__)

# SSD MobileNet v2 COCO class names (index = class_id, 0 = background)
_SSD_COCO_NAMES = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table",
    "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


class BaseDetector(ABC):
    """Common interface for all object detectors."""

    backend_name: str = "unknown"

    @abstractmethod
    def detect_all(self, frame, restrict_to_allowed: bool = True) -> list[dict]:
        """Return list of dicts with keys: label, confidence, bbox, center, area."""

    def detect_target(self, frame, requested_label=None) -> dict | None:
        candidates = self.detect_all(frame, restrict_to_allowed=True)
        if requested_label:
            candidates = [c for c in candidates if c["label"] == requested_label]
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x["area"], x["confidence"]), reverse=True)
        return candidates[0]


class YOLODetector(BaseDetector):
    """Object detector using Ultralytics YOLO (any variant: yolo11s, yolo8n, yolo8s, …)."""

    def __init__(self, model_name: str, allowed_labels: tuple[str, ...], confidence: float = 0.35):
        self.allowed_labels = set(allowed_labels)
        self.confidence = confidence
        self.model = None
        self.backend_name = model_name
        model_path = Path(model_name)
        if not model_path.is_absolute():
            local = Path(__file__).resolve().parent / model_path
            if local.exists():
                model_path = local
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(model_path))
        except Exception as e:
            logger.warning("YOLO not ready (%s): %s", model_name, e)

    def detect_all(self, frame, restrict_to_allowed: bool = True) -> list[dict]:
        if self.model is None:
            return []
        candidates = []
        try:
            results = self.model(frame, verbose=False)
            for result in results:
                names = result.names
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = names.get(cls_id, str(cls_id))
                    if conf < self.confidence:
                        continue
                    if restrict_to_allowed and label not in self.allowed_labels:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    area = max(0, x2 - x1) * max(0, y2 - y1)
                    candidates.append({
                        "label": label,
                        "confidence": conf,
                        "bbox": (x1, y1, x2, y2),
                        "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                        "area": area,
                    })
        except Exception as e:
            logger.warning("YOLO detection failed: %s", e)
        return candidates


class SSDDetector(BaseDetector):
    """Object detector using SSD MobileNet v2 via OpenCV DNN (no extra pip packages)."""

    backend_name = "ssd"

    def __init__(
        self,
        allowed_labels: tuple[str, ...],
        confidence: float = 0.35,
        model_pb: str = "models/ssd_mobilenet_v2_coco.pb",
        config_pbtxt: str = "models/ssd_mobilenet_v2_coco.pbtxt",
    ):
        self.allowed_labels = set(allowed_labels)
        self.confidence = confidence
        self.net = None

        project_root = Path(__file__).resolve().parent.parent
        pb_path = project_root / model_pb
        pbtxt_path = project_root / config_pbtxt

        if not pb_path.exists() or not pbtxt_path.exists():
            logger.warning(
                "SSD model files not found at %s / %s. "
                "Run scripts/download_ssd_model.py to download them.",
                pb_path, pbtxt_path,
            )
            return
        try:
            import cv2
            self.net = cv2.dnn.readNetFromTensorflow(str(pb_path), str(pbtxt_path))
            logger.info("SSD MobileNet v2 loaded from %s", pb_path)
        except Exception as e:
            logger.warning("SSD load failed: %s", e)

    def detect_all(self, frame, restrict_to_allowed: bool = True) -> list[dict]:
        if self.net is None:
            return []
        import cv2
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
        self.net.setInput(blob)
        try:
            out = self.net.forward()
        except Exception as e:
            logger.warning("SSD forward pass failed: %s", e)
            return []

        candidates = []
        for det in out[0, 0]:
            conf = float(det[2])
            if conf < self.confidence:
                continue
            cls_id = int(det[1])
            if cls_id < 0 or cls_id >= len(_SSD_COCO_NAMES):
                continue
            label = _SSD_COCO_NAMES[cls_id]
            if label in ("N/A", "background"):
                continue
            if restrict_to_allowed and label not in self.allowed_labels:
                continue
            x1 = max(0, int(det[3] * w))
            y1 = max(0, int(det[4] * h))
            x2 = min(w, int(det[5] * w))
            y2 = min(h, int(det[6] * h))
            area = max(0, x2 - x1) * max(0, y2 - y1)
            candidates.append({
                "label": label,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                "area": area,
            })
        return candidates


# Backward-compatible alias
TargetDetector = YOLODetector

_DETECTOR_CYCLE = ["yolo11s", "yolo8n", "yolo8s", "yolo8m", "ssd"]


def create_detector(backend: str, allowed_labels: tuple, confidence: float) -> BaseDetector:
    """Factory: create a detector by backend name."""
    if backend == "ssd":
        return SSDDetector(allowed_labels, confidence)
    model_name = backend if backend.endswith(".pt") else f"{backend}.pt"
    return YOLODetector(model_name, allowed_labels, confidence)


def next_detector_backend(current: str) -> str:
    """Return the next backend name in the cycle."""
    try:
        idx = _DETECTOR_CYCLE.index(current)
    except ValueError:
        idx = 0
    return _DETECTOR_CYCLE[(idx + 1) % len(_DETECTOR_CYCLE)]
