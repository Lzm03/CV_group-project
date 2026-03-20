from guidance import GuidancePolicy


def test_guidance_policy_requests_visible_hand_and_target():
    policy = GuidancePolicy(x_threshold=10, y_threshold=10, near_threshold=20, stable_frames_required=2)

    result = policy.decide(None, None)

    assert result.message == "show hand and place target in view"
    assert result.hand_visible is False
    assert result.target_visible is False


def test_guidance_policy_marks_repeated_instruction_as_stable():
    policy = GuidancePolicy(x_threshold=10, y_threshold=10, near_threshold=20, stable_frames_required=2)

    first = policy.decide((0, 0), (100, 0))
    second = policy.decide((0, 0), (100, 0))

    assert first.message == "move left"
    assert second.stable is True


def test_guidance_policy_stops_when_target_is_within_reach():
    policy = GuidancePolicy(x_threshold=10, y_threshold=10, near_threshold=20)

    result = policy.decide((10, 10), (18, 18))

    assert result.message == "stop, object within reach, grasp now"
    assert result.distance_hint == 16
