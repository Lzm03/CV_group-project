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
    window_name: str = "Vision-Assisted Grasping MVP"
    use_tts: bool = True
    tts_provider: str = "minimax"  # minimax | say
    minimax_voice_id: str = ""
    minimax_model: str = "speech-02-hd"
    frame_width: int = 960
    frame_height: int = 540
    snapshot_burst_count: int = 5
    debug: bool = True
    ignored_scene_labels: tuple[str, ...] = ("person",)
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

    def normalize_target(self, label: str) -> str:
        label = label.strip().lower()
        return self.label_aliases.get(label, label)
