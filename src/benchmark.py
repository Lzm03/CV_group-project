import csv
import logging
from collections import deque
from datetime import datetime
from pathlib import Path


logger = logging.getLogger(__name__)

_EMA_ALPHA = 0.1  # exponential moving-average smoothing factor


class BenchmarkRecorder:
    """Records per-frame latency and accuracy metrics for all CV backends.

    Usage:
        recorder = BenchmarkRecorder()
        recorder.record_detection("yolo11s", ms=12.3, found=True, conf=0.87)
        recorder.record_hand_tracking("mediapipe", ms=4.1, found=True)
        recorder.record_depth("pixel", ms=0.1, pixel_cm=15.2, depth_cm=15.2)
        recorder.commit_frame()
        ...
        path = recorder.save_csv("logs")
    """

    def __init__(self, max_history: int = 1000):
        self._records: deque[dict] = deque(maxlen=max_history)
        self._current: dict = {}
        # Exponential moving averages for real-time overlay
        self._det_ms = 0.0
        self._hand_ms = 0.0
        self._depth_ms = 0.0
        self._det_backend = "—"
        self._hand_backend = "—"
        self._depth_backend = "—"
        self._pixel_dist_cm = 0.0
        self._depth_dist_cm = 0.0

    def record_detection(self, backend: str, ms: float, found: bool, conf: float = 0.0) -> None:
        self._current.update(
            detector=backend,
            det_ms=round(ms, 2),
            det_found=int(found),
            det_conf=round(conf, 3),
        )
        self._det_ms = (1 - _EMA_ALPHA) * self._det_ms + _EMA_ALPHA * ms
        self._det_backend = backend

    def record_hand_tracking(self, backend: str, ms: float, found: bool) -> None:
        self._current.update(
            hand_tracker=backend,
            hand_ms=round(ms, 2),
            hand_found=int(found),
        )
        self._hand_ms = (1 - _EMA_ALPHA) * self._hand_ms + _EMA_ALPHA * ms
        self._hand_backend = backend

    def record_depth(self, backend: str, ms: float, pixel_cm: float, depth_cm: float) -> None:
        self._current.update(
            depth_backend=backend,
            depth_ms=round(ms, 2),
            pixel_dist_cm=round(pixel_cm, 1),
            depth_dist_cm=round(depth_cm, 1),
        )
        self._depth_ms = (1 - _EMA_ALPHA) * self._depth_ms + _EMA_ALPHA * ms
        self._depth_backend = backend
        self._pixel_dist_cm = pixel_cm
        self._depth_dist_cm = depth_cm

    def commit_frame(self) -> None:
        """Finalise the current frame's record and append to history."""
        if not self._current:
            return
        self._current["timestamp"] = datetime.now().isoformat(timespec="milliseconds")
        self._records.append(dict(self._current))
        self._current = {}

    def reset(self) -> None:
        """Clear all recorded frames. Call this when switching backends to start fresh."""
        self._records.clear()
        self._current = {}

    def overlay_lines(self) -> list[str]:
        """Return short status strings suitable for on-screen overlay."""
        return [
            f"[Det]  {self._det_backend:<10} {self._det_ms:5.1f} ms",
            f"[Hand] {self._hand_backend:<10} {self._hand_ms:5.1f} ms",
            f"[Dep]  {self._depth_backend:<10} {self._depth_ms:5.1f} ms",
            f"[Dist] px={self._pixel_dist_cm:5.1f}cm  d={self._depth_dist_cm:5.1f}cm",
            f"[Rec]  {len(self._records)} frames",
        ]

    def save_csv(self, directory: str = "logs") -> str:
        """Save all recorded frames to a timestamped CSV and return the file path."""
        if not self._records:
            logger.warning("No benchmark data to save")
            return ""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        fieldnames = sorted({k for r in self._records for k in r})
        with open(fname, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._records)
        logger.info("Benchmark saved → %s  (%d frames)", fname, len(self._records))
        return str(fname)
