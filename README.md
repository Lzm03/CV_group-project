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
- pyttsx3 or MiniMax TTS for audio output

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
1. Create a Python 3.10 or 3.11 virtual environment
2. Install dependencies with `pip install -r requirements.txt`
3. Run `python run.py` from the project root
4. Press `k` to type a command or `v` to use voice input
5. Observe overlay + spoken directions

## Windows notes
- Windows local testing should use the default `auto` TTS provider, which falls back to `pyttsx3`
- Voice input supports either the `whisper` CLI or the `openai-whisper` Python package
- `ffmpeg-python` alone does not provide the `ffmpeg` executable; when `ffmpeg.exe` is unavailable, the app now prefers the Python Whisper backend for recorded WAV input
- Voice input now records a 6-second window by default to make short pauses less likely to produce empty transcripts
- Empty transcripts now return a friendly retry prompt instead of surfacing raw backend errors
- For `mediapipe>=0.10.32`, download the official `hand_landmarker.task` model and place it in the project root before running the app
- The YOLO model path is now resolved from the `src/` folder, so you can launch from the repository root

## Recent optimizations
- Audio output now reuses HTTP sessions, cleans up temporary MP3 files, and closes TTS resources on exit
- Logging is standardized across runtime modules, and production defaults now use `debug=False`
- The pipeline has been simplified with helper methods and early returns, while avoiding unnecessary frame copies
- Runtime dependencies are version-pinned, and developer test dependencies are tracked separately
- Automated tests now cover the core config, geometry, NLU, pipeline, and speech-input behavior with an 80%+ coverage gate
- Voice input can use either the Whisper CLI or the Python `openai-whisper` backend, and now handles missing `ffmpeg.exe` more gracefully
- Hand tracking now supports newer MediaPipe task-based environments by loading `hand_landmarker.task` from the project root

