from collections import deque
from dataclasses import dataclass


@dataclass
class GuidanceState:
    message: str
    hand_center: tuple[int, int] | None = None
    target_center: tuple[int, int] | None = None
    distance_hint: float | None = None
    stable: bool = False
    hand_visible: bool = False
    target_visible: bool = False


class GuidancePolicy:
    def __init__(self, x_threshold: int, y_threshold: int, near_threshold: int, stable_frames_required: int = 4):
        self.x_threshold = x_threshold
        self.y_threshold = y_threshold
        self.near_threshold = near_threshold
        self.history = deque(maxlen=stable_frames_required)
        self.stable_frames_required = stable_frames_required

    def _push(self, message: str) -> tuple[str, bool]:
        self.history.append(message)
        stable = len(self.history) == self.stable_frames_required and len(set(self.history)) == 1
        return (self.history[-1], stable)

    def decide(self, hand_center, target_center):
        hand_visible = hand_center is not None
        target_visible = target_center is not None

        if not hand_visible and not target_visible:
            msg, stable = self._push("show hand and place target in view")
            return GuidanceState(msg, stable=stable, hand_visible=False, target_visible=False)
        if not target_visible:
            msg, stable = self._push("target not detected")
            return GuidanceState(msg, hand_center=hand_center, stable=stable, hand_visible=hand_visible, target_visible=False)
        if not hand_visible:
            msg, stable = self._push("show your hand to the camera")
            return GuidanceState(msg, target_center=target_center, stable=stable, hand_visible=False, target_visible=target_visible)

        hx, hy = hand_center
        tx, ty = target_center
        dx = tx - hx
        dy = ty - hy
        manhattan = abs(dx) + abs(dy)

        if manhattan < self.near_threshold:
            msg, stable = self._push("stop, object within reach, grasp now")
            return GuidanceState(msg, hand_center, target_center, manhattan, stable, True, True)

        # IMPORTANT: mirror left/right for user-facing camera guidance.
        # If target appears to the right of the hand in the image, tell user move LEFT.
        if abs(dx) > abs(dy):
            if dx > self.x_threshold:
                msg, stable = self._push("move left")
                return GuidanceState(msg, hand_center, target_center, manhattan, stable, True, True)
            if dx < -self.x_threshold:
                msg, stable = self._push("move right")
                return GuidanceState(msg, hand_center, target_center, manhattan, stable, True, True)
        if dy > self.y_threshold:
            msg, stable = self._push("move down")
            return GuidanceState(msg, hand_center, target_center, manhattan, stable, True, True)
        if dy < -self.y_threshold:
            msg, stable = self._push("move up")
            return GuidanceState(msg, hand_center, target_center, manhattan, stable, True, True)
        msg, stable = self._push("move closer")
        return GuidanceState(msg, hand_center, target_center, manhattan, stable, True, True)
