from config import AppConfig
from nlu import SimpleNLU


def test_nlu_parses_scene_summary_queries():
    nlu = SimpleNLU(AppConfig())

    result = nlu.parse("What objects are in front of me?")

    assert result.intent == "scene_summary"
    assert result.target is None


def test_nlu_resolves_aliases_for_selection_and_direction_queries():
    nlu = SimpleNLU(AppConfig())

    select_result = nlu.parse("I want to pick up the phone")
    direction_result = nlu.parse("Which direction is the mug?")

    assert select_result.intent == "select_target"
    assert select_result.target == "cell phone"
    assert direction_result.intent == "ask_direction"
    assert direction_result.target == "cup"


def test_nlu_parses_grasp_status_and_unknown_queries():
    nlu = SimpleNLU(AppConfig())

    grasp_result = nlu.parse("Did I get the pen?")
    unknown_result = nlu.parse("Tell me a joke")

    assert grasp_result.intent == "ask_grasp_status"
    assert grasp_result.target == "pen"
    assert unknown_result.intent == "unknown"
