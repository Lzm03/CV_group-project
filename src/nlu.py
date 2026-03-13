import re
from dataclasses import dataclass


@dataclass
class IntentResult:
    intent: str
    target: str | None = None
    raw_text: str = ""


class SimpleNLU:
    def __init__(self, config):
        self.config = config
        self.supported = list(config.target_labels)

    def _find_target(self, text: str):
        text = text.lower().strip()
        text = text.replace("cellphone", "cell phone").replace("phone", "cell phone")
        for label in self.supported:
            if label in text:
                return label
        aliases = config_aliases = getattr(self.config, 'label_aliases', {})
        for alias, canonical in aliases.items():
            if alias in text:
                return canonical
        return None

    def parse(self, text: str) -> IntentResult:
        t = text.lower().strip()
        target = self._find_target(t)

        # Common Whisper confusions for scene-summary queries
        scene_like = [
            "what objects", "what can you see", "what is in front of me", "what's in front of me",
            "where you can see", "what you can see", "but you can see", "can you see",
            "what do you see", "tell me what you see", "tell me what is in front of me",
        ]
        if any(p in t for p in scene_like):
            return IntentResult("scene_summary", raw_text=text)

        select_like = [
            "i want to pick up", "i want the", "help me pick up", "pick up the",
            "grab the", "take the", "i want to grab", "i want to take",
        ]
        if any(p in t for p in select_like):
            return IntentResult("select_target", target=target, raw_text=text)

        direction_like = ["where is", "which direction", "where's", "where the", "direction of"]
        if any(p in t for p in direction_like):
            return IntentResult("ask_direction", target=target, raw_text=text)

        grasp_like = ["did i get", "am i holding", "do i have", "did i grab", "am i grabbing"]
        if any(p in t for p in grasp_like):
            return IntentResult("ask_grasp_status", target=target, raw_text=text)

        if target is not None:
            return IntentResult("select_target", target=target, raw_text=text)

        return IntentResult("unknown", raw_text=text)
