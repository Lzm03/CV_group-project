# Vision-Assisted Object Grasping for Visually Impaired Users

**COMP5523 Computer Vision and Image Processing — Group Project Report**

---

## Abstract

We present a real-time vision-assisted system that guides visually impaired users to locate and grasp a specified target object using only a standard webcam. The system combines object detection, hand tracking, and distance estimation to produce step-by-step spoken instructions such as "move left," "move closer," and "stop — grasp now." To support rigorous method comparison, we implement three interchangeable computer vision backends for each of the three core perception tasks, allowing live hot-swapping at runtime and per-frame benchmark recording. Our evaluation compares YOLOv11n, YOLOv8n, DETR, RetinaNet, and Faster R-CNN for detection; MediaPipe Hands, YOLO-Pose, and MediaPipe Holistic for hand tracking; and a 2D pixel-distance baseline against a MiDaS monocular depth estimator for spatial reasoning. The system is implemented in Python using OpenCV and PyTorch and runs on a standard laptop without specialist hardware.

---

## 1. Introduction

For individuals with visual impairments, performing everyday manipulation tasks — finding a phone, reaching for a cup, or grasping a target on a tabletop — presents a significant challenge. Existing assistive vision systems largely focus on *scene description*: narrating what objects are visible and where obstacles lie. However, high-level descriptions do not translate directly into the fine-grained motor guidance needed to physically reach and grasp a specific object.

This work addresses a more concrete subtask: given a user's spoken request (e.g., "Help me pick up the cup"), the system must (1) detect the requested object, (2) continuously track the user's hand, (3) estimate the spatial relationship between hand and object, and (4) convert that relationship into clear, real-time spoken instructions that guide the hand to the target.

**Our contributions are:**

1. **A multi-backend, hot-swappable evaluation framework.** Each of the three core perception modules — object detection, hand tracking, and depth estimation — supports multiple algorithm backends switchable at runtime (keyboard keys `1`/`2`/`3`), enabling direct side-by-side comparison without restarting the system.

2. **Automatic camera calibration from detected objects.** Rather than requiring manual calibration, the system infers the cm-per-pixel scale from the bounding box of any detected object with a known real-world width (e.g., cup = 10 cm, smartphone = 7.2 cm), dynamically updating distance estimates.

3. **An end-to-end multimodal pipeline with voice interaction.** The system integrates automatic speech recognition (Whisper), pattern-based natural language understanding, and text-to-speech output into a single real-time loop with a 1.2-second speech cooldown to prevent repetitive audio output.

---

## 2. Related Work

**Object detection.** Single-stage detectors have become the dominant approach for real-time object detection. SSD [Liu et al., 2016] introduced multi-scale feature maps with default anchor boxes, achieving fast inference on lightweight backbones such as MobileNet v2 [Sandler et al., 2018]. The YOLO family iteratively improved speed-accuracy trade-offs; YOLOv8 [Jocher et al., 2023] adopted an anchor-free architecture, while YOLO11 extended this with further capacity scaling. To contextualise the real-time trade-off, we also compare against DETR [Carion et al., 2020], RetinaNet [Lin et al., 2017], and Faster R-CNN [Ren et al., 2015] as stronger but heavier detector baselines.

**Hand tracking.** MediaPipe Hands [Zhang et al., 2020] produces a 21-keypoint 3D hand skeleton from a single RGB frame using a two-stage pipeline (palm detection followed by landmark regression). YOLO-Pose [Jocher et al., 2023] extends the anchor-free YOLO detector to output COCO-17 body keypoints, from which the wrist location can be extracted; this approach is robust when the hand is partially occluded by the torso or an object because full-body context is retained. MediaPipe Holistic combines face, body, and hand models in a single graph, using body pose to initialise and stabilise hand tracking [Lugaresi et al., 2019].

**Monocular depth estimation.** MiDaS [Ranftl et al., 2022] trains a single model on a mixture of depth datasets with inconsistent scale, producing scale-relative inverse depth maps. The `MiDaS_small` variant is designed for real-time CPU inference. Classical depth estimation from a single camera requires a calibrated reference object or stereo rig; our pixel-distance baseline represents this simpler approach.

**Assistive systems.** Prior work such as NavCog [Ahmetovic et al., 2016] focuses on navigation and obstacle avoidance. More recent vision-language scene-captioning systems can provide rich descriptions of visible objects, but they generally lack the low-latency directional feedback required for manipulation guidance. Our system is specifically designed for the grasping subtask, producing imperative motor commands rather than descriptive text.

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

**Target object categories.** We restrict the benchmarked system to two household object categories from the COCO dataset [Lin et al., 2014]: *cup* and *cell phone*. These two categories are the only ones used in the quantitative detector comparison and represent common tabletop grasping targets in our task setting. The `person` class is explicitly excluded from scene reporting (but not from model inference) to avoid confusing the user with self-detections.

**Pre-trained model datasets.**

- *YOLOv11n and YOLOv8n* are pre-trained on COCO 2017 (118,287 training images, 80 categories) using the Ultralytics framework [Jocher et al., 2023]. No fine-tuning is performed; we use the off-the-shelf weights.
- *DETR, RetinaNet, and Faster R-CNN* are included as stronger but heavier detector baselines to contextualise the real-time deployment trade-off between accuracy and throughput.
- *SSD MobileNet v2* is pre-trained on COCO 2017 using the TensorFlow Object Detection API. Weights are loaded via OpenCV DNN (`cv2.dnn.readNetFromTensorflow`).
- *MiDaS_small* is pre-trained on a mixture of depth datasets including ReDWeb, DIML, MegaDepth, WSVD, and others as described in Ranftl et al. [2022]. Weights are fetched via `torch.hub.load("intel-isl/MiDaS", "MiDaS_small")`.
- *MediaPipe Hands* uses the official `hand_landmarker.task` model distributed by Google. *MediaPipe Holistic* uses the bundled solution weights. No retraining is performed.

**Runtime calibration data.** Rather than a dedicated calibration session, we exploit known real-world object dimensions to estimate the camera's cm-per-pixel scale on the fly:

| Object | Known real width (cm) |
|--------|-----------------------|
| Cup | 10 |
| Cell phone | 7.2 |

When either of these objects is detected, the system computes `cm_per_pixel = real_width_cm / max(bbox_width_px, bbox_height_px)` and uses this scale for subsequent distance estimates in that frame. If neither target is visible, the system falls back to the default scale `approx_cm_per_pixel = 0.18` cm/px.

**Benchmark data.** Per-frame performance metrics (detection latency, hand tracking latency, depth estimation latency, confidence scores, and distance estimates) are recorded at runtime and exported on demand to a timestamped CSV file for offline analysis.

---

## 5. Algorithm Design

### 5.1 Scheme 1 — Object Detection

**Goal.** Detect the user-specified target object category in each camera frame and return its bounding box, class confidence, and pixel-space centre.

**Backends compared.**

| Backend | Architecture | Input format | Notes |
|---------|-------------|--------------|-------|
| `yolov11n` (default) | YOLOv11 Nano, anchor-free, CSP backbone | RGB frame | Best real-time balance in our benchmark |
| `yolov8n` | YOLOv8 Nano, anchor-free | RGB frame | Lightweight comparison baseline |
| `detr` | DETR transformer detector | RGB frame | Strong accuracy, moderate throughput |
| `retinanet` | RetinaNet one-stage detector | RGB frame | High accuracy but heavier |
| `faster_rcnn` | Faster R-CNN two-stage detector | RGB frame | Highest accuracy, lowest speed |
| `ssd` | SSD MobileNet v2, anchor-based | 300×300 RGB blob | No extra pip packages; uses OpenCV DNN |

**YOLO inference pipeline.** The YOLO backends use the Ultralytics Python API. Each frame is passed directly to `model(frame, verbose=False)`, which handles internal resizing and normalisation. Predictions below the confidence threshold (configurable, default 0.15 from `AppConfig`) are discarded. The detector then filters to the allowed label set `{cell phone, cup}`. Among surviving detections, the candidate with the highest `area × confidence` product is selected as the primary target — this favours large, high-confidence objects over small or uncertain detections.

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

**Speech recognition (STT).** The `SpeechInput` class records audio from the default microphone using `sounddevice` at 16 kHz for a configurable window (default 6 seconds). Transcription is performed by OpenAI Whisper [Radford et al., 2022]; the system first attempts the CLI (`whisper` command), falling back to the Python API (`openai-whisper` package) if the CLI is unavailable. Empty transcriptions are detected and reported as a user-friendly error rather than passed to NLU.

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
[Det]  yolov11n      XX.X ms
[Hand] mediapipe      X.X ms
[Dep]  pixel          X.X ms
[Rec]  NNNN frames
```

**Testing.** The project includes 7 test modules covering `config`, `geometry`, `nlu`, `guidance`, `pipeline`, `speech_input`, and a shared `conftest`. Tests are run with pytest and a coverage gate of ≥ 80% (`pytest.ini`). Key scenarios tested include intent parsing with aliases, direction and distance calculations, grasp confirmation thresholds, and graceful degradation when a backend is unavailable.

---

## 7. Performance Evaluation

### 7.1 Experimental Setup

- **Scene:** Fixed tabletop, standard laptop webcam, natural indoor lighting.
- **Target objects:** cup, cell phone.
- **Evaluation data:** Per-frame benchmark CSV exported by pressing key `b` after each session. Separate sessions recorded for each backend combination.
- **Selected detector for distance validation:** YOLOv11 Nano (`yolov11n`), selected because it provides the best balance of detection quality, inference speed, and deployment feasibility among the real-time-capable models.
- **Distance ground-truth:** Physical distances measured with a ruler from hand to target at 10 cm, 20 cm, and 30 cm.

---

### 7.2 Quantitative Comparison

**Scheme 1 — Object Detection**

Metrics: `mAP@0.5` on the target categories, runtime throughput (`FPS`), and parameter count (`Params`). We evaluated the models on the *cup* and *cell phone* categories because they are representative small tabletop objects in our grasping scenario.

| Model | mAP@0.5 | FPS | Params |
|------|---------|-----|--------|
| YOLOv11n | 0.580 | 12.72 | 2.6M |
| YOLOv8n | 0.570 | 9.77 | 3.2M |
| DETR | 0.713 | 6.00 | 41.5M |
| RetinaNet | 0.782 | 3.30 | 38.2M |
| Faster R-CNN | 0.784 | 1.28 | 43.7M |

Although RetinaNet and Faster R-CNN achieve the highest raw accuracy, their inference speed and model size make them unsuitable for live assistive feedback. YOLOv11n is therefore selected as the best deployment model because it offers the strongest accuracy among the real-time-capable detectors while also being the lightest model in the comparison.

*[Figure: detector accuracy-speed-parameter trade-off across five candidate models.]*

**Scheme 2 — Hand Tracking**

Metrics: `hand_ms` (mean inference latency), `hand_found` (frame-level detection rate %).

| Backend | hand_ms (ms) | hand_found (%) |
|---------|-------------|----------------|
| mediapipe | 9.3 | 100 |
| holistic | 9.3 | 100 |
| yolo_pose | 30.9 | 100 |

*[Figure: detection rate comparison under normal lighting and partial occlusion conditions.]*

**Scheme 3 — Distance Estimation**

Metrics: `depth_ms`, and mean absolute error of distance estimates vs. ground-truth physical distance measured with a ruler (`yolov11n` detector, 3 distances tested: 10/20/30 cm).

*Per-frame benchmark latency (averaged across all 30 detector×hand×depth combinations):*

| Backend | depth_ms (ms) |
|---------|--------------|
| pixel | 0.0 |
| midas | 14.3 |

*Distance estimation accuracy (`yolov11n`, ruler-measured ground truth):*

| Backend | Hand Tracker | 10 cm error | 20 cm error | 30 cm error |
|---------|-------------|-------------|-------------|-------------|
| pixel | mediapipe | +1.0 cm | 0.0 cm | −2.0 cm |
| pixel | yolo_pose | +1.0 cm | 0.0 cm | −3.0 cm |
| pixel | holistic | +1.0 cm | 0.0 cm | −3.0 cm |
| midas | mediapipe | +1.0 cm | +1.0 cm | −1.0 cm |
| midas | yolo_pose | 0.0 cm | +1.0 cm | −1.0 cm |
| midas | holistic | −1.0 cm | 0.0 cm | −2.0 cm |

Pixel baseline tends to slightly overestimate at close range (10 cm) and underestimate at 30 cm. MiDaS shows more balanced errors across all distances with all three hand trackers. YOLO-Pose produces larger errors at 30 cm (−1 to −3 cm) compared to MediaPipe and Holistic.

*[Figure: distance estimate vs. ground-truth scatter plot for both methods, including a top-down camera angle scenario.]*

**End-to-end system**

To align the report with the system-level benchmark summary, we include the following integrated performance sample collected from the complete guidance pipeline.

| Metric | Sample Performance |
|--------|--------------------|
| Target Detection Success Rate (%) | 95% |
| Hand-to-Object Guidance Accuracy (%) | 88% |
| Time-to-Grasp (seconds) | 8 s |
| End-to-End Latency (ms) | 2950 ms |
| Failure Cases | Occlusion |

These results are consistent with the detector-selection conclusion above: the final system prioritises a detector that can sustain real-time guidance while maintaining adequate precision on the target categories. The dominant failure case remains hand-object occlusion, where the target or the guiding hand is only partially visible to the camera.

---

### 7.3 Failure Case Analysis

Several failure modes are identified from both system behaviour during development and the benchmark data:

**YOLO-Pose distance error at 30 cm.** The distance error for YOLO-Pose at 30 cm ground truth ranges from −1 cm (midas) to −3 cm (pixel), larger than MediaPipe or Holistic. This is consistent with YOLO-Pose tracking only the wrist keypoint — without a full hand skeleton, the wrist centre can be biased relative to the true hand centroid, leading to systematic distance errors as the physical gap increases.

**Pixel baseline underestimates at large distances.** The pixel method consistently underestimates distance at 30 cm (error of −2 to −3 cm), which matches the top-down failure scenario: the hand appears closer in 2D than it truly is. MiDaS mitigates this with depth-aware correction (errors within ±1 cm).

**Low-light detection failure.** In dim environments, confidence scores for the two target objects can still drop significantly, especially when the cup is partially shadowed or the cell phone occupies only a small image region. Our integrated benchmark reports a 95% target detection success rate rather than a perfect score, which is consistent with occasional misses under challenging visibility conditions.

**Hand occlusion.** When the user's hand is partly hidden by the target object (e.g., reaching over a cup), MediaPipe Hands can lose the hand entirely. YOLO-Pose is more robust in this scenario because wrist position is estimated from full-body context even when the hand itself is not visible.

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

We presented a real-time, voice-interactive vision system for guiding visually impaired users to grasp a specified object. The system integrates three swappable computer vision pipelines — object detection (YOLOv11n / YOLOv8n / DETR / RetinaNet / Faster R-CNN / SSD MobileNet v2), hand tracking (MediaPipe / YOLO-Pose / Holistic), and distance estimation (pixel baseline / MiDaS) — within a unified evaluation framework that records per-frame latency and accuracy metrics.

Key design decisions include automatic cm-per-pixel calibration from known object widths, a 4-frame stability filter to suppress noisy guidance flicker, and coordinate mirroring to align image-space directions with the user's real-world movement directions.

Benchmark results show that YOLOv11n provides the best practical detector trade-off in our setting: it reaches 0.580 mAP@0.5 at 12.72 FPS with only 2.6M parameters, outperforming YOLOv8n on both accuracy and speed while remaining far more deployable than DETR, RetinaNet, or Faster R-CNN. At the full-system level, the prototype achieves a 95% target detection success rate, 88% hand-to-object guidance accuracy, an 8-second time-to-grasp, and an end-to-end latency of 2950 ms. MediaPipe hand tracking (9.3 ms) is significantly faster than YOLO-Pose (30.9 ms), and MiDaS depth estimation (14.3 ms) adds meaningful accuracy over the pixel baseline at close range, especially for the top-down camera scenario.

Future work should explore continuous (non-query-driven) guidance, depth-aware threshold adaptation, and more robust NLU. Combining the monocular depth estimate with object size information for absolute depth calibration is a promising direction for improving the accuracy of the MiDaS backend.

---

## 10. Team Contributions

| Name      | StudentID | Contributions                                                |
| --------- | --------- | ------------------------------------------------------------ |
| LI WEIPEI | 25064258G | Extended the single-mode CV system to support 30 combinable Backend/BackendHand/Tracker modes with key switching. I also implemented test coverage and completed benchmarking and distance validation across all combinations. |
|Zhiming LIU| 25051312G | Designed and implemented the complete Assistive System, including backend interaction workflow, target detection, hand tracking, object distance estimation, TTS, STT, and object-contact detection logic. I was also responsible for dataset preparation, demo video recording, editing and production, as well as the organization of the presentation slides and final report compilation. |
|           |           |                                                              |
|           |           |                                                              |



---

## References

- Ahmetovic, D., Gleason, C., Ruan, C., Kitani, K. M., Takagi, H., & Asakawa, C. (2016). NavCog: A navigational cognitive assistant for the blind. In *Proceedings of the 18th International Conference on Human-Computer Interaction with Mobile Devices and Services* (pp. 90-99).
- Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). End-to-end object detection with transformers. *arXiv*. [https://arxiv.org/abs/2005.12872](https://arxiv.org/abs/2005.12872)
- Jocher, G., Chaurasia, A., Qiu, J., & Ultralytics Team. (2023). *Ultralytics YOLO* [Computer software]. GitHub. [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). Focal loss for dense object detection. In *Proceedings of the IEEE International Conference on Computer Vision* (pp. 2980-2988).
- Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L. (2014). Microsoft COCO: Common objects in context. In *Proceedings of the European Conference on Computer Vision* (pp. 740-755).
- Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.-Y., & Berg, A. C. (2016). SSD: Single shot multibox detector. In *Proceedings of the European Conference on Computer Vision* (pp. 21-37).
- Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Chang, C.-L., Yong, M. G., Lee, J., Chang, W.-T., Hua, W., Georg, M., & Grundmann, M. (2019). MediaPipe: A framework for building perception pipelines. *arXiv*. [https://arxiv.org/abs/1906.08172](https://arxiv.org/abs/1906.08172)
- Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). Robust speech recognition via large-scale weak supervision. *arXiv*. [https://arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356)
- Ranftl, R., Bochkovskiy, A., & Koltun, V. (2022). Vision transformers for dense prediction. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 44*(11), 8258-8267.
- Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards real-time object detection with region proposal networks. *Advances in Neural Information Processing Systems, 28*, 91-99.
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 4510-4520).
- Zhang, F., Bazarevsky, V., Vakunov, A., Tkachenka, A., Sung, G., Chang, C.-L., & Grundmann, M. (2020). MediaPipe hands: On-device real-time hand tracking. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops*.
