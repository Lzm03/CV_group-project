class HandTracker:
    def __init__(self):
        self.hands = None
        self.drawer = None
        self.hand_connections = None
        try:
            import mediapipe as mp
            self.mp = mp
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )
            self.drawer = mp.solutions.drawing_utils
            self.hand_connections = mp.solutions.hands.HAND_CONNECTIONS
        except Exception as e:
            print(f"[WARN] MediaPipe not ready: {e}")

    def detect(self, frame_bgr):
        if self.hands is None:
            return None, None
        try:
            frame_rgb = frame_bgr[:, :, ::-1]
            results = self.hands.process(frame_rgb)
            if not results.multi_hand_landmarks:
                return None, None
            hand_landmarks = results.multi_hand_landmarks[0]
            h, w = frame_bgr.shape[:2]
            coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            center_x = sum(p[0] for p in coords) // len(coords)
            center_y = sum(p[1] for p in coords) // len(coords)
            return (center_x, center_y), hand_landmarks
        except Exception as e:
            print(f"[WARN] hand tracking failed: {e}")
            return None, None
