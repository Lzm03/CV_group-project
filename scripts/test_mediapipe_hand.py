import cv2
import mediapipe as mp


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    if not cap.isOpened():
        raise RuntimeError('Could not open camera')

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    print('MediaPipe hand test running...')
    print('Show one hand clearly to the camera.')
    print('Controls: q=quit')

    while True:
        ok, frame = cap.read()
        if not ok:
            print('Failed to read frame')
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        detected = False
        if results.multi_hand_landmarks:
            detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                h, w = frame.shape[:2]
                xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
                ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
                cx = sum(xs) // len(xs)
                cy = sum(ys) // len(ys)
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                cv2.putText(frame, f'hand center: ({cx}, {cy})', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        status = 'HAND DETECTED' if detected else 'NO HAND DETECTED'
        color = (0, 255, 0) if detected else (0, 0, 255)
        cv2.putText(frame, status, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow('MediaPipe Hand Test', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
