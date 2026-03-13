import cv2

from audio import AudioGuide
from detector import TargetDetector
from geometry import (
    distance_phrase,
    estimate_distance_cm,
    movement_instruction,
    overlap_ratio,
    to_cardinal_direction,
    to_clock_direction,
)
from hand_tracker import HandTracker
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


class QuerySnapshotPipeline:
    def __init__(self, config):
        self.config = config
        self.current_target_label = None
        self.detector = TargetDetector(config.yolo_model, config.target_labels, config.detection_confidence)
        self.hand_tracker = HandTracker()
        self.audio = AudioGuide(
            config.speech_cooldown_sec,
            enabled=config.use_tts,
            provider=config.tts_provider,
            minimax_voice_id=config.minimax_voice_id,
            minimax_model=config.minimax_model,
        )
        self.speech_input = SpeechInput()
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

    def _hand_box_from_landmarks(self, frame, hand_landmarks):
        if not hand_landmarks:
            return None
        pts = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
        h, w = frame.shape[:2]
        xs = [int(x * w) for x, _ in pts]
        ys = [int(y * h) for _, y in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    def update_live_state(self, frame):
        detections = self.detector.detect_all(frame, restrict_to_allowed=True)
        detections = [d for d in detections if d['label'] not in self.config.ignored_scene_labels]
        hand_center, hand_landmarks = self.hand_tracker.detect(frame)
        hand_box = self._hand_box_from_landmarks(frame, hand_landmarks)

        self.live_detections = detections
        self.live_hand_center = hand_center
        self.live_hand_landmarks = hand_landmarks
        self.live_hand_box = hand_box
        self.last_snapshot = frame.copy()
        self.last_snapshot_detections = detections
        self.last_snapshot_hand_center = hand_center
        self.last_snapshot_hand_box = hand_box
        self.last_snapshot_hand_landmarks = hand_landmarks

        labels = []
        seen = set()
        for d in sorted(detections, key=lambda x: x['confidence'], reverse=True):
            if d['label'] not in seen:
                seen.add(d['label'])
                labels.append(d['label'])
        self.last_scene_objects = labels
        self.last_raw_detection_text = ', '.join(
            [f"{d['label']}({d['confidence']:.2f})" for d in sorted(detections, key=lambda x: x['confidence'], reverse=True)[:8]]
        ) if detections else 'none'

    def _best_target(self, detections, label):
        if not label:
            return None
        candidates = [d for d in detections if d['label'] == label]
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x['area'], x['confidence']), reverse=True)
        return candidates[0]

    def answer_scene_summary(self):
        labels = self.last_scene_objects
        if not labels:
            msg = pick(SCENE_EMPTY)
        elif len(labels) == 1:
            msg = pick(SCENE_ONE).format(objects=labels[0])
        else:
            objects_text = ', '.join(labels[:-1]) + f", and {labels[-1]}"
            msg = pick(SCENE_MANY).format(objects=objects_text)
        self.last_answer = msg
        self.audio.speak(msg)
        return msg

    def answer_direction(self, target_label=None):
        detections = self.live_detections
        hand_center = self.live_hand_center
        if target_label:
            self.current_target_label = self.config.normalize_target(target_label)
        if not self.current_target_label:
            msg = "Please tell me which object you want first."
            self.last_answer = msg
            self.audio.speak(msg)
            return msg
        target = self._best_target(detections, self.current_target_label)
        if target is None:
            msg = pick(DIRECTION_NO_TARGET).format(target=self.current_target_label)
            self.last_answer = msg
            self.audio.speak(msg)
            return msg
        if hand_center is None:
            msg = pick(SHOW_HAND).format(target=self.current_target_label)
            self.last_answer = msg
            self.audio.speak(msg)
            return msg

        tx, ty = target['center']
        hx, hy = hand_center
        dx, dy = tx - hx, ty - hy
        dist = abs(dx) + abs(dy)
        clock = to_clock_direction(dx, dy)
        cardinal = to_cardinal_direction(dx, dy, self.config.x_threshold, self.config.y_threshold)
        movement = movement_instruction(dx, dy, self.config.x_threshold, self.config.y_threshold)
        approx_cm = estimate_distance_cm(dx, dy, self.config.approx_cm_per_pixel)
        msg = pick(DIRECTION_REPLY).format(
            target=self.current_target_label,
            clock=clock,
            cardinal=cardinal,
            movement=movement,
            approx_cm=approx_cm,
            distance=distance_phrase(dist, self.config.near_threshold),
            follow=pick(FOLLOW_UP),
        )
        self.last_answer = msg
        self.audio.speak(msg)
        return msg

    def answer_grasp_status(self, target_label=None):
        detections = self.live_detections
        hand_center = self.live_hand_center
        hand_box = self.live_hand_box
        if target_label:
            self.current_target_label = self.config.normalize_target(target_label)
        if not self.current_target_label:
            msg = "Please tell me which object you want first."
            self.last_answer = msg
            self.audio.speak(msg)
            return msg
        target = self._best_target(detections, self.current_target_label)
        if target is None or hand_center is None or hand_box is None:
            msg = f"I cannot confirm whether you are holding the {self.current_target_label} yet. Please keep it in view and ask me again."
            self.last_answer = msg
            self.audio.speak(msg)
            return msg

        tx, ty = target['center']
        hx, hy = hand_center
        dist = abs(tx - hx) + abs(ty - hy)
        ov = overlap_ratio(hand_box, target['bbox'])
        if dist < self.config.near_threshold * 0.9 or ov > 0.25:
            msg = pick(GRASP_YES).format(target=self.current_target_label, follow=pick(FOLLOW_UP))
        else:
            dx, dy = tx - hx, ty - hy
            clock = to_clock_direction(dx, dy)
            cardinal = to_cardinal_direction(dx, dy, self.config.x_threshold, self.config.y_threshold)
            movement = movement_instruction(dx, dy, self.config.x_threshold, self.config.y_threshold)
            approx_cm = estimate_distance_cm(dx, dy, self.config.approx_cm_per_pixel)
            msg = pick(GRASP_NO).format(
                target=self.current_target_label,
                clock=clock,
                cardinal=cardinal,
                movement=movement,
                approx_cm=approx_cm,
            )
        self.last_answer = msg
        self.audio.speak(msg)
        return msg

    def handle_voice_text(self, text):
        self.last_heard_text = text or ""
        self.last_voice_status = "heard"
        print(f"[HEARD] {self.last_heard_text}")
        result = self.nlu.parse(text)
        self.audio.speak(pick(CHECKING))
        if result.intent == 'scene_summary':
            self.last_voice_status = 'understood: scene_summary'
            return self.answer_scene_summary()
        if result.intent == 'select_target':
            self.last_voice_status = f"understood: select_target ({result.target})"
            return self.answer_direction(result.target)
        if result.intent == 'ask_direction':
            self.last_voice_status = f"understood: ask_direction ({result.target})"
            return self.answer_direction(result.target)
        if result.intent == 'ask_grasp_status':
            self.last_voice_status = f"understood: ask_grasp_status ({result.target})"
            return self.answer_grasp_status(result.target)
        self.last_voice_status = 'not understood'
        msg = pick(UNKNOWN)
        self.last_answer = msg
        self.audio.speak(msg)
        return msg

    def listen_for_command(self):
        self.audio.speak(pick(LISTENING))
        text, err = self.speech_input.listen_once()
        if err:
            self.last_voice_status = f"voice error: {err}"
            self.last_heard_text = ""
            msg = f"Voice input failed: {err}"
            print(f"[VOICE ERROR] {err}")
            self.audio.speak(msg)
            return None, msg
        return text, None

    def draw_status(self, frame):
        self.update_live_state(frame)
        display = frame.copy()

        for det in self.live_detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(display, (x1, y1), (x2, y2), (90, 90, 90), 1)
            cv2.putText(display, det['label'], (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        if self.live_hand_landmarks and self.hand_tracker.drawer:
            self.hand_tracker.drawer.draw_landmarks(display, self.live_hand_landmarks, self.hand_tracker.hand_connections)
        if self.live_hand_center:
            cv2.circle(display, self.live_hand_center, 12, (255, 0, 0), -1)
            cv2.circle(display, self.live_hand_center, 18, (255, 200, 0), 2)
            cv2.putText(display, 'hand', (self.live_hand_center[0] + 14, self.live_hand_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        cv2.rectangle(display, (10, 10), (920, 108), (25, 25, 25), -1)
        target_text = self.current_target_label if self.current_target_label else 'none'
        scene_text = ', '.join(self.last_scene_objects) if self.last_scene_objects else 'none'
        heard = (self.last_heard_text[:52] + '...') if len(self.last_heard_text) > 52 else self.last_heard_text
        answer = (self.last_answer[:68] + '...') if len(self.last_answer) > 68 else self.last_answer
        hand_status = 'OK' if self.live_hand_center is not None else 'MISSING'
        cv2.putText(display, f"Hand: {hand_status}   Target: {target_text}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(display, f"Objects: {scene_text}", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,200,200), 1)
        cv2.putText(display, f"Heard: {heard if heard else 'none'}", (20, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220,220,120), 1)
        cv2.putText(display, f"Answer: {answer if answer else 'none'}", (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80,220,255), 1)
        return display
