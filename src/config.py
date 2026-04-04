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
    # Known real-world widths (cm) for bbox-based scale auto-calibration.
    # Objects not listed here fall back to approx_cm_per_pixel.
    known_object_widths_cm: dict[str, float] = field(default_factory=lambda: {
        "cup": 10,         # typical mug diameter
        "cell phone": 7.2,  # typical smartphone width
        "paper": 21.0,      # A4 width
    })
    # Detection method backends (used for multi-method comparison)
    detector_backend: str = "yolo11s"       # yolo11s | yolo8n | yolo8s | yolo8m | ssd
    hand_tracker_backend: str = "mediapipe" # mediapipe | yolo_pose | holistic
    depth_estimator_backend: str = "pixel"  # pixel | midas
    depth_update_every_n_frames: int = 3
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
        valid_detectors = {"yolo11s", "yolo8n", "yolo8s", "yolo8m", "ssd"}
        if self.detector_backend not in valid_detectors:
            raise ValueError(f"detector_backend must be one of: {valid_detectors}")
        if self.hand_tracker_backend not in {"mediapipe", "yolo_pose", "holistic"}:
            raise ValueError("hand_tracker_backend must be one of: mediapipe, yolo_pose, holistic")
        if self.depth_estimator_backend not in {"pixel", "midas"}:
            raise ValueError("depth_estimator_backend must be one of: pixel, midas")
        if self.depth_update_every_n_frames < 1:
            raise ValueError("depth_update_every_n_frames must be >= 1")

    def normalize_target(self, label: str) -> str:
        label = label.strip().lower()
        return self.label_aliases.get(label, label)
