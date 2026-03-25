# Group Project MVP

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Interactive vision-assisted object grasping for visually impaired users.

## MVP goal
Build a real-time demo that:
- detects one target object category at a time
- tracks one hand from webcam input
- estimates relative hand/object position
- speaks simple guidance like: move left/right/up/down, closer, stop, grasp

## Proposed stack
- Python 3.10+
- OpenCV
- Ultralytics YOLO (object detection)
- MediaPipe Hands (hand tracking)
- pyttsx3 or MiniMax TTS for audio output

## MVP scope
- Fixed tabletop scene
- Single camera (laptop webcam or phone webcam)
- Supported scene/grasp targets: pen, paper, cup, cell phone
- One hand only
- Rule-based guidance policy

## Project structure
- `src/main.py` — app entry point
- `src/config.py` — app config and backend selection
- `src/detector.py` — object detection (YOLO / SSD multi-backend)
- `src/hand_tracker.py` — hand tracking (MediaPipe / YOLO-Pose / Holistic multi-backend)
- `src/depth_estimator.py` — distance estimation (pixel baseline / MiDaS depth)
- `src/benchmark.py` — per-frame latency and accuracy recorder
- `src/guidance.py` — hand-to-object guidance logic
- `src/audio.py` — speech output with anti-spam cooldown
- `src/pipeline.py` — end-to-end frame processing
- `docs/plan.md` — implementation roadmap
- `requirements.txt` — Python dependencies

## Quick start
1. Create a Python 3.10 or 3.11 virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Run from the project root: `python run.py`
4. Press `k` to type a command or `v` to use voice input
5. Observe overlay + spoken directions

### Runtime keyboard controls

| Key | Action |
|-----|--------|
| `v` | Voice input (Whisper) |
| `k` | Keyboard input |
| `1` | Cycle object detector backend |
| `2` | Cycle hand tracker backend |
| `3` | Cycle depth estimator backend |
| `b` | Save benchmark CSV to `logs/` |
| `q` | Quit |

---

## Multi-method comparison (CV evaluation)

The system supports hot-swapping CV backends at runtime (keys `1` / `2` / `3`) and records per-frame metrics automatically. Press `b` at any time to export a CSV for analysis.

### Scheme 1 — Object detection comparison

Three backends are available, switchable with key `1`:

| Backend | Model | Notes |
|---------|-------|-------|
| `yolo11s` *(default)* | YOLO11 Small | Best accuracy on pen/paper/cup/phone |
| `yolo8n` / `yolo8s` / `yolo8m` | YOLOv8 variants | Speed-accuracy trade-off comparison |
| `ssd` | SSD MobileNet v2 (OpenCV DNN) | Lightweight; no extra pip packages needed |

**First-time SSD setup** — download the model files (~67 MB) before switching to this backend:

```bash
python scripts/download_ssd_model.py
```

The files are saved to `models/` and loaded automatically on the next `ssd` switch.

**Distance auto-calibration**: when a cup, cell phone, or paper is detected, the system automatically estimates the cm/pixel scale from the object's bounding box size and its known real-world width. This removes the need to manually tune `approx_cm_per_pixel` for different camera distances. Known widths are defined in `config.py` under `known_object_widths_cm` and can be adjusted to match your objects.

### Scheme 2 — Hand tracker comparison

Three backends are available, switchable with key `2`:

| Backend | Method | Notes |
|---------|--------|-------|
| `mediapipe` *(default)* | MediaPipe Hands | 21-point hand skeleton; best accuracy for finger-level guidance |
| `yolo_pose` | YOLOv8-Pose (wrist keypoint) | Detects wrist from full-body pose; more robust when hand is partially occluded |
| `holistic` | MediaPipe Holistic | Full-body model; maintains hand tracking using body pose context |

No additional downloads are required — `ultralytics` and `mediapipe` are already listed in `requirements.txt`. The `yolov8n-pose.pt` model is fetched automatically by Ultralytics on first use.

All three backends return the same hand-center coordinate used for guidance, so switching does not affect guidance logic. The purpose is benchmarking: compare FPS, detection rate, and landmark jitter across methods.

### Scheme 3 — Distance / depth estimation comparison

Two backends are available, switchable with key `3`:

| Backend | Method | Notes |
|---------|--------|-------|
| `pixel` *(default)* | 2-D Euclidean × cm/pixel | Fast; auto-calibrated when object size is known |
| `midas` | MiDaS_small monocular depth | Adds Z-axis awareness; ~83 MB model, CPU-intensive |

MiDaS is loaded via `torch.hub` on first switch and cached automatically. Because depth inference is expensive, it runs every `depth_update_every_n_frames` frames (default 3) rather than every frame.

**When MiDaS is useful**: in top-down or angled camera setups, a hand can appear directly over a target in 2-D (pixel distance ≈ 0) while still being 15–20 cm above it in the Z-axis. The pixel method incorrectly reports "within reach" in this case; MiDaS provides the depth component needed to detect and report the gap.

Both methods record their distance estimates in the benchmark CSV, enabling a direct accuracy comparison.

### Reading the benchmark CSV

After pressing `b`, a file like `logs/benchmark_20250325_143022.csv` is created. Key columns:

| Column | Description |
|--------|-------------|
| `detector` | Active object detector backend |
| `det_ms` | Object detection latency (ms) |
| `det_conf` | Confidence of the best detection |
| `hand_tracker` | Active hand tracker backend |
| `hand_ms` | Hand tracking latency (ms) |
| `depth_backend` | Active depth estimator |
| `pixel_dist_cm` | 2-D pixel-based distance estimate (cm) |
| `depth_dist_cm` | 3-D depth-aware distance estimate (cm) |

Use this data in your course report to compare methods across FPS, accuracy, and latency dimensions.

---

## Windows / macOS notes
- Use the default `auto` TTS provider, which tries `pyttsx3` → MiniMax → `say` in order
- Voice input supports either the `whisper` CLI or the `openai-whisper` Python package
- For `mediapipe>=0.10.32`, download the official `hand_landmarker.task` model and place it in the project root before running
- The YOLO model path is resolved from `src/`, so launch from the repository root

## Recent updates
- Added three CV backend families for multi-method comparison (object detection, hand tracking, depth estimation)
- Distance estimation now auto-calibrates cm/pixel scale from detected object bounding box size
- Per-frame benchmark recorder saves latency and accuracy data to CSV for course report analysis
- Runtime key shortcuts (`1` / `2` / `3` / `b`) allow live backend switching and data export without restarting
- Audio output reuses HTTP sessions, cleans up temporary MP3 files, and closes TTS resources on exit
- Automated tests cover config, geometry, NLU, pipeline, and speech-input with an 80%+ coverage gate

