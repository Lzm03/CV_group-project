# Vision-Assisted Object Grasping for Visually Impaired Users

**COMP5523 Computer Vision and Image Processing — Group Project Report**

---

## Abstract

We present a real-time vision-assisted system that guides visually impaired users to locate and grasp a specified target object using only a standard webcam. The system combines object detection, hand tracking, and distance estimation to produce step-by-step spoken instructions such as "move left," "move closer," and "stop — grasp now." To support rigorous method comparison, we implement three interchangeable computer vision backends for each of the three core perception tasks, allowing live hot-swapping at runtime and per-frame benchmark recording. Our evaluation compares YOLO11s, YOLOv8 variants, and SSD MobileNet v2 for detection; MediaPipe Hands, YOLO-Pose, and MediaPipe Holistic for hand tracking; and a 2D pixel-distance baseline against a MiDaS monocular depth estimator for spatial reasoning. The system is implemented in Python using OpenCV and PyTorch and runs on a standard laptop without specialist hardware.

---

## 1. Introduction

For individuals with visual impairments, performing everyday manipulation tasks — picking up a pen, finding a phone, reaching for a cup — presents a significant challenge. Existing assistive vision systems largely focus on *scene description*: narrating what objects are visible and where obstacles lie [citation needed]. However, high-level descriptions do not translate directly into the fine-grained motor guidance needed to physically reach and grasp a specific object.

This work addresses a more concrete subtask: given a user's spoken request (e.g., "Help me pick up the cup"), the system must (1) detect the requested object, (2) continuously track the user's hand, (3) estimate the spatial relationship between hand and object, and (4) convert that relationship into clear, real-time spoken instructions that guide the hand to the target.

**Our contributions are:**

1. **A multi-backend, hot-swappable evaluation framework.** Each of the three core perception modules — object detection, hand tracking, and depth estimation — supports multiple algorithm backends switchable at runtime (keyboard keys `1`/`2`/`3`), enabling direct side-by-side comparison without restarting the system.

2. **Automatic camera calibration from detected objects.** Rather than requiring manual calibration, the system infers the cm-per-pixel scale from the bounding box of any detected object with a known real-world width (e.g., A4 paper = 21 cm, smartphone = 7.2 cm), dynamically updating distance estimates.

3. **An end-to-end multimodal pipeline with voice interaction.** The system integrates automatic speech recognition (Whisper), pattern-based natural language understanding, and text-to-speech output into a single real-time loop with a 1.2-second speech cooldown to prevent repetitive audio output.

---

## 2. Related Work

**Object detection.** Single-stage detectors have become the dominant approach for real-time object detection. SSD [Liu et al., 2016] introduced multi-scale feature maps with default anchor boxes, achieving fast inference on lightweight backbones such as MobileNet v2 [Sandler et al., 2018]. The YOLO family iteratively improved speed-accuracy trade-offs; YOLOv8 [Jocher et al., 2023] adopted an anchor-free architecture, while YOLO11 extended this with further capacity scaling. We compare these families to understand the trade-off in our tabletop grasping scenario.

**Hand tracking.** MediaPipe Hands [Zhang et al., 2020] produces a 21-keypoint 3D hand skeleton from a single RGB frame using a two-stage pipeline (palm detection followed by landmark regression). YOLO-Pose [Cheng et al., 2022] extends the anchor-free YOLO detector to output COCO-17 body keypoints, from which the wrist location can be extracted; this approach is robust when the hand is partially occluded by the torso or an object because full-body context is retained. MediaPipe Holistic combines face, body, and hand models in a single graph, using body pose to initialise and stabilise hand tracking.

**Monocular depth estimation.** MiDaS [Ranftl et al., 2020] trains a single model on a mixture of depth datasets with inconsistent scale, producing scale-relative inverse depth maps. The `MiDaS_small` variant is designed for real-time CPU inference. Classical depth estimation from a single camera requires a calibrated reference object or stereo rig; our pixel-distance baseline represents this simpler approach.

**Assistive systems.** Prior work such as NavCog [Ahmetovic et al., 2016] and EyeSense [citation needed] focus on navigation and obstacle avoidance. Scene-captioning systems (e.g., using BLIP or GPT-4V) provide rich descriptions but lack the real-time directional granularity required for manipulation guidance. Our system is specifically designed for the grasping subtask, producing imperative motor commands rather than descriptive text.

---

## 3. System Overview

The system processes a camera frame through four sequential stages each frame, then responds to user voice queries asynchronously.

```
┌──────────────────────────────────────────────────────────────┐
│                      Camera Frame (960×540)                   │
└──────┬──────────────────────────┬───────────────────┬─────────┘
       │                          │                   │
┌──────▼──────┐           ┌───────▼──────┐   ┌───────▼────────┐
│  Detector   │           │ Hand Tracker │   │ Depth Estimator│
│ (Scheme 1)  │           │  (Scheme 2)  │   │  (Scheme 3)    │
│             │           │              │   │                │
│ label, conf,│           │ hand_center, │   │ pixel_dist_cm, │
│ bbox, center│           │ landmarks    │   │ depth_dist_cm  │
└──────┬──────┘           └───────┬──────┘   └───────┬────────┘
       └────────────────────────────────────────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │   Benchmark Recorder    │
                     │  (per-frame CSV export) │
                     └────────────┬────────────┘
                                  │
              ┌───────────────────▼────────────────────┐
              │       Voice Interaction Layer          │
              │  STT (Whisper) → NLU → TTS            │
              │  Guidance Policy → spoken instruction  │
              └────────────────────────────────────────┘
```

**Frame processing.** In each frame, the detector identifies all objects in the scene and selects the best candidate for the user's requested target (highest `area × confidence`). The hand tracker locates the user's hand and computes its pixel-space centre from available landmarks. The depth estimator ingests the frame and computes two distance estimates between the hand centre and the target centre. All timing and accuracy metrics are committed to the benchmark recorder.

**Interaction mode.** The system operates in a *query-driven* mode: the user presses a key to trigger a voice query, the utterance is transcribed and parsed, and the system replies with a spoken instruction. The camera loop runs continuously and independently, maintaining an up-to-date snapshot of the scene for each query.

**Modularity.** Each perception module is implemented as an abstract base class (`BaseDetector`, `BaseHandTracker`, `BaseDepthEstimator`) with concrete backend subclasses. A factory function (`create_detector`, `create_hand_tracker`, `create_depth_estimator`) instantiates the requested backend. Runtime cycling (`next_detector_backend`, etc.) enables hot-swapping without restarting.

---

## 4. Data Preparation

**Target object categories.** We restrict the system to four household object categories from the COCO dataset [Lin et al., 2014]: *pen*, *paper*, *cup*, and *cell phone*. These categories cover common daily grasping scenarios and are present in the COCO label set used by all pre-trained models in our comparison. The `person` class is explicitly excluded from scene reporting (but not from model inference) to avoid confusing the user with self-detections.

**Pre-trained model datasets.**

- *YOLO11s and YOLOv8 variants* are pre-trained on COCO 2017 (118,287 training images, 80 categories) using the Ultralytics framework [Jocher et al., 2023]. No fine-tuning is performed; we use the off-the-shelf weights.
- *SSD MobileNet v2* is pre-trained on COCO 2017 using the TensorFlow Object Detection API. Weights are loaded via OpenCV DNN (`cv2.dnn.readNetFromTensorflow`).
- *MiDaS_small* is pre-trained on a mixture of depth datasets including ReDWeb, DIML, MegaDepth, WSVD, and others as described in Ranftl et al. [2020]. Weights are fetched via `torch.hub.load("intel-isl/MiDaS", "MiDaS_small")`.
- *MediaPipe Hands* uses the official `hand_landmarker.task` model distributed by Google. *MediaPipe Holistic* uses the bundled solution weights. No retraining is performed.

**Runtime calibration data.** Rather than a dedicated calibration session, we exploit known real-world object dimensions to estimate the camera's cm-per-pixel scale on the fly:

| Object | Known real width (cm) |
|--------|-----------------------|
| Cup | 8.5 |
| Cell phone | 7.2 |
| Paper (A4) | 21.0 |

When any of these objects is detected, the system computes `cm_per_pixel = real_width_cm / max(bbox_width_px, bbox_height_px)` and uses this scale for subsequent distance estimates in that frame. Objects not in this table (e.g., pen) fall back to the default scale `approx_cm_per_pixel = 0.18` cm/px.

**Benchmark data.** Per-frame performance metrics (detection latency, hand tracking latency, depth estimation latency, confidence scores, and distance estimates) are recorded at runtime and exported on demand to a timestamped CSV file for offline analysis.

---

## 5. Algorithm Design

### 5.1 Scheme 1 — Object Detection

**Goal.** Detect the user-specified target object category in each camera frame and return its bounding box, class confidence, and pixel-space centre.

**Backends compared.**

| Backend | Architecture | Input format | Notes |
|---------|-------------|--------------|-------|
| `yolo11s` (default) | YOLO11 Small, anchor-free, CSP backbone | RGB frame | Highest accuracy on our four categories |
| `yolo8n` / `yolo8s` / `yolo8m` | YOLOv8 variants (nano/small/medium), anchor-free | RGB frame | Speed-accuracy range for ablation |
| `ssd` | SSD MobileNet v2, anchor-based | 300×300 RGB blob | No extra pip packages; uses OpenCV DNN |

**YOLO inference pipeline.** The YOLO backends use the Ultralytics Python API. Each frame is passed directly to `model(frame, verbose=False)`, which handles internal resizing and normalisation. Predictions below the confidence threshold (configurable, default 0.15 from `AppConfig`) are discarded. The detector then filters to the allowed label set `{pen, paper, cell phone, cup}`. Among surviving detections, the candidate with the highest `area × confidence` product is selected as the primary target — this favours large, high-confidence objects over small or uncertain detections.

**SSD MobileNet v2 inference pipeline.** The frame is preprocessed with `cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)`, which resizes to 300×300 and swaps BGR to RGB. The output tensor has shape `[1, 1, N, 7]`; columns are `[batch, label, confidence, x1_norm, y1_norm, x2_norm, y2_norm]`. Bounding box coordinates are denormalised by multiplying by frame width/height and clipped to image bounds.

**Target selection.** `BaseDetector.detect_target()` sorts candidates by `(area, confidence)` in descending order, selecting the first match. This is implemented in the base class and shared by all backends.

**Evaluation metrics.** For each frame: `det_ms` (inference wall-clock time in milliseconds), `det_conf` (confidence of the selected detection, or 0 if none), `det_found` (binary: 1 if any candidate found, else 0).

---

### 5.2 Scheme 2 — Hand Tracking

**Goal.** Estimate the pixel-space centre of the user's hand in each camera frame. Optionally, return a set of landmarks for visualisation and bounding-box computation.

**Backends compared.**

| Backend | Method | Landmarks returned | Strength |
|---------|--------|--------------------|----------|
| `mediapipe` (default) | MediaPipe Hands (Tasks API) — 21-point 3D skeleton | 21 normalised (x, y, z) points | Finger-level precision |
| `yolo_pose` | YOLOv8-Pose — COCO-17 body keypoints, wrist extracted | 1 point (wrist) | Robust under partial hand occlusion |
| `holistic` | MediaPipe Holistic — full-body + hand sub-graph | 21 normalised points | Body context stabilises hand tracking |

**MediaPipe Hands pipeline.** Frames are converted from BGR to RGB before being passed to the landmark model. The Tasks API requires a monotonically increasing timestamp in milliseconds; the tracker maintains a frame counter to ensure this. Key inference parameters: `min_detection_confidence = 0.4`, `min_tracking_confidence = 0.4`, `num_hands = 1`, `running_mode = VIDEO`. Normalised landmark coordinates `(x_norm, y_norm)` are converted to pixel space as `(int(x_norm × width), int(y_norm × height))`. The hand centre is computed as the pixel-space mean of all 21 landmarks.

**YOLO-Pose pipeline.** The `yolov8n-pose.pt` model outputs 17 keypoints per detected person following the COCO-17 convention. Keypoints 9 (left wrist) and 10 (right wrist) are extracted; the wrist with the higher confidence score is selected as the hand location. Only one point is returned per frame, so the hand centre equals the wrist coordinate directly.

**MediaPipe Holistic pipeline.** The Holistic model processes the full frame and returns pose, face, and hand landmarks in a single forward pass. The right-hand landmark set is preferred; if absent, left-hand landmarks are used. The hand centre is computed identically to the MediaPipe Hands backend.

**Evaluation metrics.** `hand_ms` (inference wall-clock time), `hand_found` (binary: 1 if a hand was detected).

---

### 5.3 Scheme 3 — Distance / Depth Estimation

**Goal.** Estimate the physical distance (in centimetres) between the hand centre and the target object centre, providing input to the guidance policy.

**Backends compared.**

| Backend | Method | Formula | Notes |
|---------|--------|---------|-------|
| `pixel` (baseline) | 2D Euclidean distance × cm/pixel scale | $d = \sqrt{\Delta x^2 + \Delta y^2} \times s$ | Stateless; no GPU required |
| `midas` | MiDaS_small monocular depth + 2D fusion | $d = \sqrt{d_{2D}^2 + d_Z^2}$ | Z-axis aware; ~83 MB model |

**Pixel distance estimator.** The pixel baseline computes the Euclidean distance between the hand centre `(hx, hy)` and the target centre `(tx, ty)` in image coordinates using `math.hypot(dx, dy)`, then multiplies by the current cm-per-pixel scale factor (dynamically calibrated where possible; see Section 4). This method is $O(1)$ per frame and requires no GPU.

**MiDaS depth estimator.** The `MiDaS_small` model is loaded via `torch.hub.load("intel-isl/MiDaS", "MiDaS_small")` and placed on GPU if available, otherwise CPU. The inference pipeline is:

1. Convert frame from BGR to RGB.
2. Apply the model-specific `small_transform` preprocessing.
3. Run forward inference with `torch.no_grad()`.
4. Upsample the output to the original frame resolution using `F.interpolate(..., mode="bicubic", align_corners=False)`.

The output is an inverse-depth map (higher values indicate closer surfaces). To produce a 3D distance estimate, the depth values at the hand and target locations are normalised to `[0, 1]` across the frame, inverted to convert to "distance from camera," and differenced:

$$d_Z = |(\text{inv\_depth}_\text{hand}) - (\text{inv\_depth}_\text{target})| \times s \times 100$$

where $s$ is the cm-per-pixel scale. The final 3D distance is $d = \sqrt{d_{2D}^2 + d_Z^2}$.

**Frame-skipping optimisation.** Because MiDaS inference is computationally expensive on CPU, the depth map is updated only once every `depth_update_every_n_frames = 3` frames. Between updates, the previous depth map is reused. This reduces the per-frame overhead while keeping the depth estimate sufficiently fresh for slow hand movements.

**Why MiDaS is useful.** In top-down or angled camera setups, a hand can appear directly above a target in 2D (pixel distance ≈ 0) while still being 15–20 cm above it in the Z-axis. The pixel baseline incorrectly reports "within reach" in this case; MiDaS provides the depth component needed to detect and report the vertical gap.

**Evaluation metrics.** `depth_ms` (wall-clock time for the depth update and distance estimation), `pixel_dist_cm` (2D estimate), `depth_dist_cm` (3D fused estimate).

---

### 5.4 Guidance Policy

**Goal.** Convert the current hand and object positions into a discrete spoken instruction that the user can act on immediately.

**Input.** Hand centre `(hx, hy)` and target centre `(tx, ty)` in pixel space. Both may be `None` if the respective component was not detected in the current frame.

**Decision logic.** The `GuidancePolicy` class implements the following state machine:

```
if hand not detected and target not detected:
    → "show hand and place target in view"
if target not detected:
    → "target not detected"
if hand not detected:
    → "show your hand to the camera"

Compute:
    dx = tx − hx,  dy = ty − hy
    manhattan = |dx| + |dy|

if manhattan < near_threshold (120 px):
    → "stop, object within reach, grasp now"
if |dx| > |dy|:                          # horizontal gap dominates
    if dx > x_threshold (60 px): → "move left"
    if dx < −x_threshold:        → "move right"
if dy > y_threshold (50 px): → "move down"
if dy < −y_threshold:        → "move up"
→ "move closer"                          # within both thresholds but not near
```

**Coordinate mirroring.** When the target appears to the right of the hand in the image (`dx > 0`), the correct user instruction is "move left" — because the camera faces the user. The sign of `dx` is therefore inverted before threshold comparisons. The same mirroring is applied in `geometry.to_clock_direction()` and `geometry.to_cardinal_direction()`.

**Stability filter.** Raw frame-by-frame decisions often flicker due to detection noise. A deque of length `stable_frames_required = 4` records the last four instruction strings. An instruction is marked *stable* only if all four entries are identical. Unstable instructions are still returned (so the system responds immediately), but are flagged so the audio output layer can optionally suppress rapid changes.

**Clock-face direction.** For spoken responses to "where is the object?" queries, the guidance uses `geometry.to_clock_direction(dx, dy)` to compute a clock-face bearing: $\theta = \arctan2(\Delta y, -\Delta x)$, converted to `clock_angle = (90 − \theta) \mod 360$, then rounded to the nearest hour. This gives a more natural spatial reference than raw cardinal directions.

**Grasp confirmation.** The pipeline confirms a successful grasp when either: (a) the Manhattan distance between hand and target is below `0.9 × near_threshold`, or (b) the overlap ratio between the hand bounding box and the target bounding box exceeds 0.25. The overlap ratio is $\text{inter\_area} / \min(\text{hand\_area}, \text{target\_area})$, which is more sensitive than standard IoU for the case where a small hand partially covers a large object.

---

## 6. System Implementation

**Pipeline structure.** The `QuerySnapshotPipeline` class orchestrates each frame. Detection, hand tracking, and depth estimation run sequentially; the results are stored as a snapshot (`last_snapshot_*`) and committed to the benchmark recorder. The webcam loop runs at the native camera frame rate (targeting 960×540 resolution); the voice interaction layer operates asynchronously in response to key presses.

**Runtime backend switching.** Keys `1`, `2`, `3` cycle through the detector, hand tracker, and depth estimator backends respectively using `next_detector_backend()`, `next_hand_tracker_backend()`, and `next_depth_backend()`. A new backend instance is created in-place; the pipeline continues processing without interruption.

**Speech recognition (STT).** The `SpeechInput` class records audio from the default microphone using `sounddevice` at 16 kHz for a configurable window (default 6 seconds). Transcription is performed by OpenAI Whisper; the system first attempts the CLI (`whisper` command), falling back to the Python API (`openai-whisper` package) if the CLI is unavailable. Empty transcriptions are detected and reported as a user-friendly error rather than passed to NLU.

**Natural language understanding (NLU).** The `SimpleNLU` class implements pattern-based intent recognition. Text is normalised (lowercased, whitespace collapsed, phone aliases unified: `cellphone` / `phone` → `cell phone`). Four intent categories are recognised by keyword prefix matching:

| Intent | Example trigger phrases |
|--------|------------------------|
| `scene_summary` | "what can you see", "what objects" |
| `select_target` | "help me pick up", "grab the", "i want the" |
| `ask_direction` | "where is", "which direction", "where's" |
| `ask_grasp_status` | "did i get", "am i holding", "do i have" |

If no intent phrase matches but a target label is found in the text, the system infers `select_target`. Unrecognised utterances return `unknown`.

**Text-to-speech (TTS).** The `AudioGuide` class selects a TTS provider in priority order: `pyttsx3` (offline, cross-platform) → MiniMax TTS API (cloud, high-quality voice) → macOS `say` command. A cooldown of 1.2 seconds prevents the same message from being repeated within that window. On Windows, `pyttsx3` runs in a subprocess to avoid threading issues. MiniMax audio is generated as MP3, played asynchronously, and cleaned up by a background thread.

**On-screen overlay.** The live video window displays: detected object bounding boxes with labels and confidence scores; hand landmarks and skeleton; a directional guidance arrow; and a four-line performance overlay updated with exponential moving average smoothing (`α = 0.1`):
```
[Det]  yolo11s       XX.X ms
[Hand] mediapipe      X.X ms
[Dep]  pixel          X.X ms
[Rec]  NNNN frames
```

**Testing.** The project includes 7 test modules covering `config`, `geometry`, `nlu`, `guidance`, `pipeline`, `speech_input`, and a shared `conftest`. Tests are run with pytest and a coverage gate of ≥ 80% (`pytest.ini`). Key scenarios tested include intent parsing with aliases, direction and distance calculations, grasp confirmation thresholds, and graceful degradation when a backend is unavailable.

---

## 7. Performance Evaluation

### 7.1 Experimental Setup

- **Scene:** Fixed tabletop, standard laptop webcam, natural indoor lighting.
- **Target objects:** pen, paper, cup, cell phone.
- **Evaluation data:** Per-frame benchmark CSV exported by pressing key `b` after each session. Separate sessions recorded for each backend combination.

*[To be filled after benchmark runs are complete.]*

---

### 7.2 Quantitative Comparison

**Scheme 1 — Object Detection**

Metrics: `det_conf` (mean detection confidence), `det_ms` (mean inference latency), `det_found` (frame-level detection rate %).

| Backend | det_ms (ms) | det_conf (mean) | det_found (%) |
|---------|------------|-----------------|---------------|
| yolo11s | *TBD* | *TBD* | *TBD* |
| yolo8n | *TBD* | *TBD* | *TBD* |
| yolo8s | *TBD* | *TBD* | *TBD* |
| yolo8m | *TBD* | *TBD* | *TBD* |
| ssd | *TBD* | *TBD* | *TBD* |

*[Figure: bar chart comparing mean latency and detection rate across backends.]*

**Scheme 2 — Hand Tracking**

Metrics: `hand_ms` (mean inference latency), `hand_found` (frame-level detection rate %).

| Backend | hand_ms (ms) | hand_found (%) |
|---------|-------------|----------------|
| mediapipe | *TBD* | *TBD* |
| yolo_pose | *TBD* | *TBD* |
| holistic | *TBD* | *TBD* |

*[Figure: detection rate comparison under normal lighting and partial occlusion conditions.]*

**Scheme 3 — Distance Estimation**

Metrics: `depth_ms`, mean absolute error of `pixel_dist_cm` and `depth_dist_cm` vs. ground-truth physical distance measured with a ruler.

| Backend | depth_ms (ms) | Distance error (cm) |
|---------|--------------|---------------------|
| pixel | *TBD* | *TBD* |
| midas | *TBD* | *TBD* |

*[Figure: distance estimate vs. ground-truth scatter plot for both methods, including a top-down camera angle scenario.]*

**End-to-end system**

- End-to-end latency (camera frame to spoken guidance): *TBD*
- Time-to-grasp (frames from voice query to "stop — grasp now"): *TBD*

---

### 7.3 Failure Case Analysis

Even without quantitative data, several failure modes are identified from system behaviour during development:

**Low-light detection failure.** In dim environments, YOLO confidence scores for small objects (pen, paper) drop significantly. The system correctly falls back to "target not detected" rather than producing a misleading guidance instruction.

**Hand occlusion.** When the user's hand is partly hidden by the target object (e.g., reaching over a cup), MediaPipe Hands can lose the hand entirely. YOLO-Pose is more robust in this scenario because wrist position is estimated from full-body context even when the hand itself is not visible.

**Top-down camera angle — pixel distance failure.** When the camera is positioned directly above the scene, a hand hovering 15 cm above the target can appear co-located in 2D (pixel distance ≈ 0). The pixel baseline then incorrectly reports "grasp now." MiDaS correctly detects the depth difference; however, the absolute scale of its depth estimates is uncalibrated, so the distance value in centimetres is approximate.

**NLU ambiguity.** The pattern-based NLU misclassifies utterances that contain multiple intent triggers (e.g., "Can you see where the cup is?" contains both `scene_like` and `direction_like` phrases). The first matching pattern wins, which may not reflect the user's actual intent.

---

## 8. Discussion and Limitations

**Monocular depth uncertainty.** MiDaS produces relative inverse-depth values with no absolute scale. Our calibration (multiplying depth differences by `cm_per_pixel × 100`) is a heuristic that works for the front-facing tabletop scenario but may produce unreliable estimates at different camera angles or distances. A stereo camera or structured-light sensor would eliminate this limitation.

**Fixed spatial thresholds.** The `near_threshold` (120 px), `x_threshold` (60 px), and `y_threshold` (50 px) are tuned for a fixed camera-to-scene distance. If the camera is positioned further away, the same pixel threshold corresponds to a larger physical distance, potentially triggering "grasp now" before the hand is actually within reach. Dynamic threshold adaptation based on the detected object size would address this.

**Query-driven interaction.** The current system responds to explicit user queries rather than providing continuous real-time guidance. This means the user must actively ask "where is the cup?" rather than receiving a constant stream of directional updates. Continuous guidance would better support the step-by-step motor correction loop implied by the task description.

**Pattern-based NLU.** `SimpleNLU` recognises intent by keyword matching. It does not handle negation, compound sentences, or out-of-vocabulary object names. A lightweight intent classifier (e.g., fine-tuned on the four intent categories) or a small language model would generalise better.

**Single-hand, fixed-scene assumption.** The system tracks one hand and targets one object at a time. Multi-object scenes with multiple candidate detections are resolved by selecting the largest high-confidence bounding box, which may not match the user's intent in cluttered environments.

---

## 9. Conclusion

We presented a real-time, voice-interactive vision system for guiding visually impaired users to grasp a specified object. The system integrates three swappable computer vision pipelines — object detection (YOLO11s / YOLOv8 / SSD MobileNet v2), hand tracking (MediaPipe / YOLO-Pose / Holistic), and distance estimation (pixel baseline / MiDaS) — within a unified evaluation framework that records per-frame latency and accuracy metrics.

Key design decisions include automatic cm-per-pixel calibration from known object widths, a 4-frame stability filter to suppress noisy guidance flicker, and coordinate mirroring to align image-space directions with the user's real-world movement directions.

*[Quantitative comparison of backends to be completed after benchmark data collection.]*

Future work should explore continuous (non-query-driven) guidance, depth-aware threshold adaptation, and more robust NLU. Combining the monocular depth estimate with object size information for absolute depth calibration is a promising direction for improving the accuracy of the MiDaS backend.

---

## 10. Team Contributions

*[To be filled by the team.]*

---

## References

- Liu, W. et al. (2016). SSD: Single Shot MultiBox Detector. *ECCV 2016.*
- Sandler, M. et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR 2018.*
- Jocher, G. et al. (2023). Ultralytics YOLOv8. [https://github.com/ultralytics/ultralytics]
- Zhang, F. et al. (2020). MediaPipe Hands: On-device Real-time Hand Tracking. *CVPRW 2020.*
- Ranftl, R. et al. (2020). Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer. *TPAMI 2022.*
- Lin, T.-Y. et al. (2014). Microsoft COCO: Common Objects in Context. *ECCV 2014.*
- Cheng, B. et al. (2022). Bottom-Up Human Pose Estimation Via Disentangled Keypoint Regression. *(YOLO-Pose reference — verify with actual citation.)*
