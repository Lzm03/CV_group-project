import logging
import time

import cv2

from audio import AudioGuide
from benchmark import BenchmarkRecorder
from depth_estimator import create_depth_estimator, next_depth_backend
from detector import create_detector, next_detector_backend
from geometry import (
    calibrate_cm_per_pixel,
    distance_phrase,
    estimate_distance_cm,
    movement_instruction,
    overlap_ratio,
    to_cardinal_direction,
    to_clock_direction,
)
from hand_tracker import create_hand_tracker, next_hand_tracker_backend
from nlu import SimpleNLU
from phrases import (
    CHECKING,
    DIRECTION_NO_TARGET,
    DIRECTION_REPLY,
    FOLLOW_UP,
    GRASP_NO,
    GRASP_YES,
    LISTENING,
    SCENE_EMPTY,
    SCENE_MANY,
    SCENE_ONE,
    SHOW_HAND,
    UNKNOWN,
    pick,
)
from speech_input import SpeechInput


logger = logging.getLogger(__name__)

MISSING_TARGET_PROMPT = "Please tell me which object you want first."
GRASP_CONFIRMATION_UNAVAILABLE = (
    "I cannot confirm whether you are holding the {target} yet. Please keep it in view and ask me again."
)
GRASP_DISTANCE_RATIO = 0.9
GRASP_OVERLAP_THRESHOLD = 0.25
RAW_DETECTION_LIMIT = 8
HEARD_TEXT_MAX_LEN = 52
ANSWER_TEXT_MAX_LEN = 68
HAND_TRACKER_UNAVAILABLE = (
    "Hand tracking is unavailable right now, so I cannot guide your hand position yet. "
    "Please check the MediaPipe setup and try again."
)


class QuerySnapshotPipeline:
    def __init__(self, config):
        self.config = config
        self.current_target_label = None
        self.detector = create_detector(
            config.detector_backend, config.target_labels, config.detection_confidence
        )
        self.hand_tracker = create_hand_tracker(
            config.hand_tracker_backend, config.hand_landmarker_model
        )
        self.depth_estimator = create_depth_estimator(
            config.depth_estimator_backend,
            config.approx_cm_per_pixel,
            config.depth_update_every_n_frames,
        )
        self.benchmark = BenchmarkRecorder()
        self.audio = AudioGuide(
            config.speech_cooldown_sec,
            enabled=config.use_tts,
            provider=config.tts_provider,
            minimax_voice_id=config.minimax_voice_id,
            minimax_model=config.minimax_model,
        )
        self.speech_input = SpeechInput(seconds=config.speech_input_seconds)
        self.nlu = SimpleNLU(config)
        self.last_scene_objects = []
        self.last_heard_text = ""
        self.last_voice_status = "idle"
        self.last_answer = ""
        self.last_snapshot = None
        self.last_snapshot_detections = []
        self.last_snapshot_hand_center = None
        self.last_snapshot_hand_box = None
        self.last_snapshot_hand_landmarks = None
        self.last_raw_detection_text = ""
        self.live_detections = []
        self.live_hand_center = None
        self.live_hand_box = None
        self.live_hand_landmarks = None

    def _respond(self, message: str):
        self.last_answer = message
        self.audio.speak(message)
        return message

    def _set_current_target(self, target_label=None):
        if target_label:
            self.current_target_label = self.config.normalize_target(target_label)
        return self.current_target_label

    def _require_target_label(self, target_label=None):
        current_target = self._set_current_target(target_label)
        if current_target:
            return current_target
        self._respond(MISSING_TARGET_PROMPT)
        return None

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return f"{text[:limit]}..."

    @staticmethod
    def _unique_scene_labels(detections):
        labels = []
        seen = set()
        for detection in detections:
            label = detection["label"]
            if label in seen:
                continue
            seen.add(label)
            labels.append(label)
        return labels

    @staticmethod
    def _format_raw_detections(detections):
        if not detections:
            return "none"
        return ", ".join(
            f"{detection['label']}({detection['confidence']:.2f})"
            for detection in detections[:RAW_DETECTION_LIMIT]
        )

    def _effective_cm_per_pixel(self, target: dict) -> float:
        """Return a dynamically calibrated cm/pixel scale if the target has a known real size,
        otherwise fall back to the configured constant."""
        label = target.get("label", "")
        real_width = self.config.known_object_widths_cm.get(label)
        if real_width:
            scale = calibrate_cm_per_pixel(target["bbox"], real_width)
            if scale:
                return scale
        return self.config.approx_cm_per_pixel

    def _direction_context(self, target, hand_center):
        tx, ty = target["center"]
        hx, hy = hand_center
        dx, dy = tx - hx, ty - hy
        manhattan = abs(dx) + abs(dy)
        cm_per_px = self._effective_cm_per_pixel(target)
        return {
            "dx": dx,
            "dy": dy,
            "manhattan": manhattan,
            "clock": to_clock_direction(dx, dy),
            "cardinal": to_cardinal_direction(dx, dy, self.config.x_threshold, self.config.y_threshold),
            "movement": movement_instruction(dx, dy, self.config.x_threshold, self.config.y_threshold),
            "approx_cm": estimate_distance_cm(dx, dy, cm_per_px),
            "distance": distance_phrase(manhattan, self.config.near_threshold),
        }

    def _hand_tracking_unavailable(self):
        return not getattr(self.hand_tracker, "available", False)

    def _hand_tracking_message(self):
        detail = getattr(self.hand_tracker, "error_message", "")
        if detail:
            logger.warning("Hand tracking unavailable: %s", detail)
        return HAND_TRACKER_UNAVAILABLE

    def _hand_box_from_landmarks(self, frame, hand_landmarks):
        if not hand_landmarks:
            return None
        pts = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
        h, w = frame.shape[:2]
        xs = [int(x * w) for x, _ in pts]
        ys = [int(y * h) for _, y in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    def update_live_state(self, frame):
        # --- Object detection with timing ---
        t0 = time.perf_counter()
        detections = self.detector.detect_all(frame, restrict_to_allowed=True)
        det_ms = (time.perf_counter() - t0) * 1000
        detections = [
            detection
            for detection in detections
            if detection["label"] not in self.config.ignored_scene_labels
        ]
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        best_conf = detections[0]["confidence"] if detections else 0.0
        self.benchmark.record_detection(
            self.detector.backend_name, det_ms, bool(detections), best_conf
        )

        # --- Hand tracking with timing ---
        t0 = time.perf_counter()
        hand_center, hand_landmarks = self.hand_tracker.detect(frame)
        hand_ms = (time.perf_counter() - t0) * 1000
        hand_box = self._hand_box_from_landmarks(frame, hand_landmarks)
        self.benchmark.record_hand_tracking(
            self.hand_tracker.backend, hand_ms, hand_center is not None
        )

        # --- Depth estimation with timing ---
        t0 = time.perf_counter()
        self.depth_estimator.update(frame)
        depth_ms = (time.perf_counter() - t0) * 1000
        pixel_cm = 0.0
        depth_cm = 0.0
        if hand_center and detections:
            target = self._best_target(detections, self.current_target_label) or detections[0]
            pixel_cm = estimate_distance_cm(
                target["center"][0] - hand_center[0],
                target["center"][1] - hand_center[1],
                self.config.approx_cm_per_pixel,
            )
            depth_cm = self.depth_estimator.estimate_distance_cm(
                hand_center, target["center"], frame.shape
            )
        self.benchmark.record_depth(
            self.depth_estimator.backend_name, depth_ms, pixel_cm, depth_cm
        )
        self.benchmark.commit_frame()

        self.live_detections = detections
        self.live_hand_center = hand_center
        self.live_hand_landmarks = hand_landmarks
        self.live_hand_box = hand_box
        self.last_snapshot = frame
        self.last_snapshot_detections = detections
        self.last_snapshot_hand_center = hand_center
        self.last_snapshot_hand_box = hand_box
        self.last_snapshot_hand_landmarks = hand_landmarks

        self.last_scene_objects = self._unique_scene_labels(detections)
        self.last_raw_detection_text = self._format_raw_detections(detections)

    def _best_target(self, detections, label):
        if not label:
            return None
        candidates = [d for d in detections if d["label"] == label]
        if not candidates:
            return None
        return max(candidates, key=lambda x: (x["area"], x["confidence"]))

    def answer_scene_summary(self):
        labels = self.last_scene_objects
        if not labels:
            msg = pick(SCENE_EMPTY)
        elif len(labels) == 1:
            msg = pick(SCENE_ONE).format(objects=labels[0])
        else:
            objects_text = ', '.join(labels[:-1]) + f", and {labels[-1]}"
            msg = pick(SCENE_MANY).format(objects=objects_text)
        return self._respond(msg)

    def answer_direction(self, target_label=None):
        current_target = self._require_target_label(target_label)
        if current_target is None:
            return self.last_answer

        target = self._best_target(self.live_detections, current_target)
        if target is None:
            return self._respond(pick(DIRECTION_NO_TARGET).format(target=current_target))
        if self._hand_tracking_unavailable():
            return self._respond(self._hand_tracking_message())
        if self.live_hand_center is None:
            return self._respond(pick(SHOW_HAND).format(target=current_target))

        context = self._direction_context(target, self.live_hand_center)
        msg = pick(DIRECTION_REPLY).format(
            target=current_target,
            clock=context["clock"],
            cardinal=context["cardinal"],
            movement=context["movement"],
            approx_cm=context["approx_cm"],
            distance=context["distance"],
            follow=pick(FOLLOW_UP),
        )
        return self._respond(msg)

    def answer_grasp_status(self, target_label=None):
        current_target = self._require_target_label(target_label)
        if current_target is None:
            return self.last_answer

        target = self._best_target(self.live_detections, current_target)
        if target is None or self.live_hand_center is None or self.live_hand_box is None:
            if self._hand_tracking_unavailable():
                return self._respond(self._hand_tracking_message())
            return self._respond(GRASP_CONFIRMATION_UNAVAILABLE.format(target=current_target))

        context = self._direction_context(target, self.live_hand_center)
        overlap = overlap_ratio(self.live_hand_box, target["bbox"])
        if (
            context["manhattan"] < self.config.near_threshold * GRASP_DISTANCE_RATIO
            or overlap > GRASP_OVERLAP_THRESHOLD
        ):
            return self._respond(
                pick(GRASP_YES).format(target=current_target, follow=pick(FOLLOW_UP))
            )

        msg = pick(GRASP_NO).format(
            target=current_target,
            clock=context["clock"],
            cardinal=context["cardinal"],
            movement=context["movement"],
            approx_cm=context["approx_cm"],
        )
        return self._respond(msg)

    def handle_voice_text(self, text):
        self.last_heard_text = text or ""
        self.last_voice_status = "heard"
        logger.info("Heard: %s", self.last_heard_text)
        result = self.nlu.parse(text)
        self.audio.speak(pick(CHECKING))
        if result.intent == "scene_summary":
            self.last_voice_status = "understood: scene_summary"
            return self.answer_scene_summary()
        if result.intent == "select_target":
            self.last_voice_status = f"understood: select_target ({result.target})"
            return self.answer_direction(result.target)
        if result.intent == "ask_direction":
            self.last_voice_status = f"understood: ask_direction ({result.target})"
            return self.answer_direction(result.target)
        if result.intent == "ask_grasp_status":
            self.last_voice_status = f"understood: ask_grasp_status ({result.target})"
            return self.answer_grasp_status(result.target)
        self.last_voice_status = "not understood"
        return self._respond(pick(UNKNOWN))

    def listen_for_command(self):
        self.audio.speak(pick(LISTENING))
        text, err = self.speech_input.listen_once()
        if err:
            self.last_voice_status = f"voice error: {err}"
            self.last_heard_text = ""
            logger.warning("Voice input failed: %s", err)
            if not self.speech_input.is_available():
                return None, err
            msg = f"Voice input failed: {err}"
            return None, self._respond(msg)
        return text, None

    def draw_status(self, frame):
        self.update_live_state(frame)
        display = frame.copy()

        for det in self.live_detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(display, (x1, y1), (x2, y2), (90, 90, 90), 1)
            cv2.putText(display, det['label'], (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        if self.live_hand_landmarks:
            self._draw_hand_landmarks(display, self.live_hand_landmarks)
        if self.live_hand_center:
            cv2.circle(display, self.live_hand_center, 12, (255, 0, 0), -1)
            cv2.circle(display, self.live_hand_center, 18, (255, 200, 0), 2)
            cv2.putText(display, "hand", (self.live_hand_center[0] + 14, self.live_hand_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.rectangle(display, (10, 10), (920, 108), (25, 25, 25), -1)
        target_text = self.current_target_label if self.current_target_label else "none"
        scene_text = ", ".join(self.last_scene_objects) if self.last_scene_objects else "none"
        heard = self._truncate_text(self.last_heard_text, HEARD_TEXT_MAX_LEN)
        answer = self._truncate_text(self.last_answer, ANSWER_TEXT_MAX_LEN)
        if self._hand_tracking_unavailable():
            hand_status = "UNAVAILABLE"
        else:
            hand_status = "OK" if self.live_hand_center is not None else "MISSING"
        cv2.putText(display, f"Hand: {hand_status}   Target: {target_text}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, f"Objects: {scene_text}", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
        cv2.putText(display, f"Heard: {heard if heard else 'none'}", (20, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 120), 1)
        cv2.putText(display, f"Answer: {answer if answer else 'none'}", (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 220, 255), 1)
        self._draw_benchmark_overlay(display)
        return display

    def _draw_hand_landmarks(self, frame, hand_landmarks):
        points = [
            (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
            for landmark in hand_landmarks.landmark
        ]
        for point in points:
            cv2.circle(frame, point, 3, (0, 255, 255), -1)

        for connection in self.hand_tracker.hand_connections or ():
            start_index = getattr(connection, "start", None)
            end_index = getattr(connection, "end", None)
            if start_index is None or end_index is None:
                start_index, end_index = connection
            if start_index >= len(points) or end_index >= len(points):
                continue
            cv2.line(frame, points[start_index], points[end_index], (0, 180, 255), 2)

    # ------------------------------------------------------------------
    # Runtime backend switching
    # ------------------------------------------------------------------

    def cycle_detector(self) -> str:
        """Switch to the next object detector backend and return its name."""
        self.hand_tracker.close()  # not relevant but keep symmetry
        next_backend = next_detector_backend(self.config.detector_backend)
        self.config.detector_backend = next_backend
        self.detector = create_detector(
            next_backend, self.config.target_labels, self.config.detection_confidence
        )
        logger.info("Detector switched → %s", next_backend)
        return next_backend

    def cycle_hand_tracker(self) -> str:
        """Switch to the next hand tracker backend and return its name."""
        self.hand_tracker.close()
        next_backend = next_hand_tracker_backend(self.config.hand_tracker_backend)
        self.config.hand_tracker_backend = next_backend
        self.hand_tracker = create_hand_tracker(next_backend, self.config.hand_landmarker_model)
        logger.info("Hand tracker switched → %s", next_backend)
        return next_backend

    def cycle_depth_estimator(self) -> str:
        """Switch to the next depth estimator backend and return its name."""
        next_backend = next_depth_backend(self.config.depth_estimator_backend)
        self.config.depth_estimator_backend = next_backend
        self.depth_estimator = create_depth_estimator(
            next_backend, self.config.approx_cm_per_pixel, self.config.depth_update_every_n_frames
        )
        logger.info("Depth estimator switched → %s", next_backend)
        return next_backend

    def save_benchmark(self, directory: str = "logs") -> str:
        """Flush benchmark records to CSV and return the file path."""
        return self.benchmark.save_csv(directory)

    # ------------------------------------------------------------------
    # Benchmark overlay drawing
    # ------------------------------------------------------------------

    def _draw_benchmark_overlay(self, frame):
        lines = self.benchmark.overlay_lines()
        x, y0 = 10, frame.shape[0] - 10 - len(lines) * 18
        cv2.rectangle(frame, (x - 4, y0 - 14), (x + 300, y0 + len(lines) * 18), (20, 20, 20), -1)
        for i, line in enumerate(lines):
            cv2.putText(
                frame, line, (x, y0 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1
            )

    def close(self):
        self.hand_tracker.close()
        self.audio.close()
