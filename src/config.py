from dataclasses import dataclass, field


@dataclass
class AppConfig:
    camera_index: int = 0
    yolo_model: str = "yolo11s.pt"
    # Custom MVP target set requested by user.
    target_labels: tuple[str, ...] = ("pen", "paper", "cell phone", "cup")
    default_target_label: str | None = None
    detection_confidence: float = 0.15
    x_threshold: int = 60
    y_threshold: int = 50
    near_threshold: int = 120
    stable_frames_required: int = 4
    approx_cm_per_pixel: float = 0.18
    speech_cooldown_sec: float = 1.2
    speech_input_seconds: float = 6.0
    window_name: str = "Vision-Assisted Grasping MVP"
    use_tts: bool = True
    tts_provider: str = "auto"  # auto | pyttsx3 | minimax | say
    minimax_voice_id: str = ""
    minimax_model: str = "speech-02-hd"
    frame_width: int = 960
    frame_height: int = 540
    snapshot_burst_count: int = 5
    debug: bool = False
    hand_landmarker_model: str = "hand_landmarker.task"
    ignored_scene_labels: frozenset[str] = field(default_factory=lambda: frozenset({"person"}))
    label_aliases: dict[str, str] = field(default_factory=lambda: {
        "phone": "cell phone",
        "mobile": "cell phone",
        "cellphone": "cell phone",
        "paper sheet": "paper",
        "sheet": "paper",
        "paper": "paper",
        "pen": "pen",
        "cup": "cup",
        "mug": "cup",
    })

    def __post_init__(self):
        self.target_labels = tuple(label.strip().lower() for label in self.target_labels)
        self.ignored_scene_labels = frozenset(label.strip().lower() for label in self.ignored_scene_labels)
        self.label_aliases = {
            alias.strip().lower(): canonical.strip().lower()
            for alias, canonical in self.label_aliases.items()
        }
        if self.default_target_label:
            self.default_target_label = self.normalize_target(self.default_target_label)

        if not 0 < self.detection_confidence <= 1:
            raise ValueError("detection_confidence must be between 0 and 1")
        if self.approx_cm_per_pixel <= 0:
            raise ValueError("approx_cm_per_pixel must be positive")
        if self.speech_cooldown_sec < 0:
            raise ValueError("speech_cooldown_sec must be non-negative")
        if self.speech_input_seconds <= 0:
            raise ValueError("speech_input_seconds must be positive")
        for field_name in (
            "x_threshold",
            "y_threshold",
            "near_threshold",
            "stable_frames_required",
            "frame_width",
            "frame_height",
            "snapshot_burst_count",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.tts_provider not in {"auto", "pyttsx3", "minimax", "say"}:
            raise ValueError("tts_provider must be one of: auto, pyttsx3, minimax, say")

    def normalize_target(self, label: str) -> str:
        label = label.strip().lower()
        return self.label_aliases.get(label, label)
