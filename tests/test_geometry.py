from geometry import (
    A_LITTLE_FAR_RATIO,
    VERY_CLOSE_RATIO,
    WITHIN_REACH_RATIO,
    distance_phrase,
    estimate_distance_cm,
    movement_instruction,
    overlap_ratio,
    to_cardinal_direction,
    to_clock_direction,
)


def test_to_clock_direction_and_cardinal_direction_match_user_facing_camera():
    assert to_clock_direction(100, 0) == "9 o'clock"
    assert to_clock_direction(-100, 0) == "3 o'clock"
    assert to_cardinal_direction(100, 0, 10, 10) == "left"
    assert to_cardinal_direction(-100, -100, 10, 10) == "up and right"


def test_movement_instruction_and_distance_estimation():
    assert movement_instruction(0, 0, 10, 10) == "reach almost straight ahead"
    assert movement_instruction(0, 50, 10, 10) == "move your hand down"
    assert estimate_distance_cm(3, 4, 0.5) == 2.5


def test_distance_phrase_uses_named_threshold_ratios():
    near_threshold = 100

    assert distance_phrase(near_threshold * WITHIN_REACH_RATIO - 1, near_threshold) == "within reach"
    assert distance_phrase(near_threshold * VERY_CLOSE_RATIO - 1, near_threshold) == "very close"
    assert distance_phrase(near_threshold * A_LITTLE_FAR_RATIO - 1, near_threshold) == "a little far"
    assert distance_phrase(near_threshold * A_LITTLE_FAR_RATIO + 1, near_threshold) == "far"


def test_overlap_ratio_uses_smaller_box_as_reference():
    assert overlap_ratio((0, 0, 10, 10), (5, 5, 15, 15)) == 0.25
    assert overlap_ratio((0, 0, 10, 10), None) == 0.0
