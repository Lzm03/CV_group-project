# Group Project MVP

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
- pyttsx3 or simple TTS wrapper for audio output

## MVP scope
- Fixed tabletop scene
- Single camera (laptop webcam or phone webcam)
- Supported scene/grasp targets: pen, paper, phone only
- One hand only
- Rule-based guidance policy

## Project structure
- `src/main.py` — app entry point
- `src/config.py` — app config
- `src/detector.py` — target detection wrapper
- `src/hand_tracker.py` — hand tracking wrapper
- `src/guidance.py` — hand-to-object guidance logic
- `src/audio.py` — speech output with anti-spam cooldown
- `src/pipeline.py` — end-to-end frame processing
- `docs/plan.md` — implementation roadmap
- `requirements.txt` — Python dependencies

## Run plan
1. Install dependencies
2. Run webcam demo
3. Select target label
4. Observe overlay + spoken directions

