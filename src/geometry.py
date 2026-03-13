import math


def to_clock_direction(dx: float, dy: float) -> str:
    # Flip horizontal axis to match the user's real-world movement direction.
    # 12 o'clock = up, 3 = right, 6 = down, 9 = left
    dx = -dx
    angle = math.degrees(math.atan2(dy, dx))
    clock_angle = (90 - angle) % 360
    hour = int((clock_angle + 15) // 30) % 12
    if hour == 0:
        hour = 12
    return f"{hour} o'clock"


def to_cardinal_direction(dx: float, dy: float, x_threshold: float, y_threshold: float) -> str:
    # Flip horizontal axis to match the user's real-world movement direction.
    dx = -dx

    horizontal = ""
    vertical = ""

    if dx > x_threshold:
        horizontal = "right"
    elif dx < -x_threshold:
        horizontal = "left"

    if dy > y_threshold:
        vertical = "down"
    elif dy < -y_threshold:
        vertical = "up"

    if horizontal and vertical:
        return f"{vertical} and {horizontal}"
    if horizontal:
        return horizontal
    if vertical:
        return vertical
    return "almost straight ahead"


def movement_instruction(dx: float, dy: float, x_threshold: float, y_threshold: float) -> str:
    direction = to_cardinal_direction(dx, dy, x_threshold, y_threshold)
    if direction == "almost straight ahead":
        return "reach almost straight ahead"
    return f"move your hand {direction}"


def estimate_distance_cm(dx: float, dy: float, cm_per_pixel: float) -> float:
    dx = float(dx)
    dy = float(dy)
    pixel_distance = math.hypot(dx, dy)
    return round(pixel_distance * cm_per_pixel, 1)


def distance_phrase(distance: float, near_threshold: float) -> str:
    if distance < near_threshold * 0.8:
        return "within reach"
    if distance < near_threshold * 1.5:
        return "very close"
    if distance < near_threshold * 2.5:
        return "a little far"
    return "far"


def overlap_ratio(box_a, box_b):
    if not box_a or not box_b:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / min(a_area, b_area)
