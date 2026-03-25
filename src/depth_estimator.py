import logging
import math
from abc import ABC, abstractmethod

import numpy as np


logger = logging.getLogger(__name__)

_DEPTH_CYCLE = ["pixel", "midas"]


class BaseDepthEstimator(ABC):
    """Common interface for distance/depth estimation methods."""

    backend_name: str = "unknown"

    @abstractmethod
    def update(self, frame) -> None:
        """Ingest a new camera frame (may update internal depth map)."""

    @abstractmethod
    def estimate_distance_cm(self, point_a: tuple, point_b: tuple, frame_shape: tuple) -> float:
        """Return estimated 3D distance in cm between two image-space points."""


class PixelDistanceEstimator(BaseDepthEstimator):
    """Baseline: 2-D Euclidean pixel distance scaled by a fixed cm/pixel factor."""

    backend_name = "pixel"

    def __init__(self, cm_per_pixel: float = 0.18):
        self.cm_per_pixel = cm_per_pixel

    def update(self, frame) -> None:
        pass  # stateless

    def estimate_distance_cm(self, point_a: tuple, point_b: tuple, frame_shape: tuple) -> float:
        dx = float(point_b[0] - point_a[0])
        dy = float(point_b[1] - point_a[1])
        return round(math.hypot(dx, dy) * self.cm_per_pixel, 1)


class MiDaSDepthEstimator(BaseDepthEstimator):
    """Monocular depth estimation using MiDaS_small via torch.hub.

    Produces a relative inverse-depth map (higher = closer).  The depth
    component is combined with the 2-D pixel distance to give a rough 3-D
    distance estimate for comparison against the pixel-only baseline.
    """

    backend_name = "midas"

    def __init__(self, cm_per_pixel: float = 0.18, update_every_n_frames: int = 3):
        self.cm_per_pixel = cm_per_pixel
        self.update_every_n_frames = update_every_n_frames
        self._frame_count = 0
        self._depth_map: np.ndarray | None = None
        self._model = None
        self._transform = None
        self._device = None
        self.available = False
        self._init_model()

    def _init_model(self):
        try:
            import torch
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small", trust_repo=True, verbose=False
            )
            self._model.to(self._device)
            self._model.eval()
            transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True, verbose=False
            )
            self._transform = transforms.small_transform
            self.available = True
            logger.info("MiDaS_small loaded on %s", self._device)
        except Exception as e:
            logger.warning("MiDaS not available (falling back to pixel distance): %s", e)

    def update(self, frame) -> None:
        if not self.available:
            return
        self._frame_count += 1
        if self._frame_count % self.update_every_n_frames != 0:
            return
        try:
            import cv2
            import torch
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch = self._transform(img_rgb).to(self._device)
            with torch.no_grad():
                pred = self._model(batch)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            self._depth_map = pred.cpu().numpy().astype(np.float32)
        except Exception as e:
            logger.warning("MiDaS update failed: %s", e)

    def estimate_distance_cm(self, point_a: tuple, point_b: tuple, frame_shape: tuple) -> float:
        dx = float(point_b[0] - point_a[0])
        dy = float(point_b[1] - point_a[1])
        dist_2d_cm = math.hypot(dx, dy) * self.cm_per_pixel

        if self._depth_map is None or not self.available:
            return round(dist_2d_cm, 1)

        h, w = frame_shape[:2]
        ax = max(0, min(int(point_a[0]), w - 1))
        ay = max(0, min(int(point_a[1]), h - 1))
        bx = max(0, min(int(point_b[0]), w - 1))
        by = max(0, min(int(point_b[1]), h - 1))

        d_min = float(self._depth_map.min())
        d_max = float(self._depth_map.max())
        d_range = (d_max - d_min) if d_max > d_min else 1.0

        # Normalize to [0, 1]: 1 = closest
        norm_a = (float(self._depth_map[ay, ax]) - d_min) / d_range
        norm_b = (float(self._depth_map[by, bx]) - d_min) / d_range
        # Convert to "distance from camera" (0=close, 1=far)
        inv_a = 1.0 - norm_a
        inv_b = 1.0 - norm_b
        # Scale depth diff to cm using a rough empirical factor (10× pixel scale)
        depth_diff_cm = abs(inv_a - inv_b) * self.cm_per_pixel * 100.0
        return round(math.hypot(dist_2d_cm, depth_diff_cm), 1)


def create_depth_estimator(backend: str, cm_per_pixel: float = 0.18,
                           update_every_n_frames: int = 3) -> BaseDepthEstimator:
    """Factory: create a depth estimator by backend name."""
    if backend == "midas":
        return MiDaSDepthEstimator(cm_per_pixel, update_every_n_frames)
    return PixelDistanceEstimator(cm_per_pixel)


def next_depth_backend(current: str) -> str:
    """Return the next backend name in the cycle."""
    try:
        idx = _DEPTH_CYCLE.index(current)
    except ValueError:
        idx = 0
    return _DEPTH_CYCLE[(idx + 1) % len(_DEPTH_CYCLE)]
