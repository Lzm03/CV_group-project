class TargetDetector:
    def __init__(self, model_name: str, allowed_labels: tuple[str, ...], confidence: float = 0.35):
        self.allowed_labels = set(allowed_labels)
        self.confidence = confidence
        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
        except Exception as e:
            print(f"[WARN] YOLO not ready: {e}")

    def _run(self, frame, restrict_to_allowed: bool = True):
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
            print(f"[WARN] detection failed: {e}")
        return candidates

    def detect_all(self, frame, restrict_to_allowed: bool = True):
        return self._run(frame, restrict_to_allowed=restrict_to_allowed)

    def detect_target(self, frame, requested_label=None):
        candidates = self.detect_all(frame, restrict_to_allowed=True)
        if requested_label:
            candidates = [c for c in candidates if c["label"] == requested_label]
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x["area"], x["confidence"]), reverse=True)
        return candidates[0]
