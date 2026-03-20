import re
from dataclasses import dataclass


PHONE_PATTERN = re.compile(r"\b(cellphone|phone)\b")
SPACE_PATTERN = re.compile(r"\s+")


@dataclass
class IntentResult:
    intent: str
    target: str | None = None
    raw_text: str = ""


class SimpleNLU:
    def __init__(self, config):
        self.config = config
        self.supported = tuple(config.target_labels)
        self.aliases = {
            self._normalize_text(alias): config.normalize_target(canonical)
            for alias, canonical in getattr(config, "label_aliases", {}).items()
        }
        self.scene_like = (
            "what objects",
            "what can you see",
            "what is in front of me",
            "what's in front of me",
            "where you can see",
            "what you can see",
            "but you can see",
            "can you see",
            "what do you see",
            "tell me what you see",
            "tell me what is in front of me",
        )
        self.select_like = (
            "i want to pick up",
            "i want the",
            "help me pick up",
            "pick up the",
            "grab the",
            "take the",
            "i want to grab",
            "i want to take",
        )
        self.direction_like = ("where is", "which direction", "where's", "where the", "direction of")
        self.grasp_like = ("did i get", "am i holding", "do i have", "did i grab", "am i grabbing")

    def _normalize_text(self, text: str) -> str:
        text = SPACE_PATTERN.sub(" ", text.lower().strip())
        return PHONE_PATTERN.sub("cell phone", text)

    def _find_target(self, text: str):
        text = self._normalize_text(text)
        for label in self.supported:
            if label in text:
                return label
        for alias, canonical in self.aliases.items():
            if alias in text:
                return canonical
        return None

    def parse(self, text: str) -> IntentResult:
        t = self._normalize_text(text)
        target = self._find_target(t)

        if any(p in t for p in self.scene_like):
            return IntentResult("scene_summary", raw_text=text)

        if any(p in t for p in self.select_like):
            return IntentResult("select_target", target=target, raw_text=text)

        if any(p in t for p in self.direction_like):
            return IntentResult("ask_direction", target=target, raw_text=text)

        if any(p in t for p in self.grasp_like):
            return IntentResult("ask_grasp_status", target=target, raw_text=text)

        if target is not None:
            return IntentResult("select_target", target=target, raw_text=text)

        return IntentResult("unknown", raw_text=text)
