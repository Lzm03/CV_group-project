import cv2
from pathlib import Path
from dotenv import load_dotenv

from config import AppConfig
from phrases import OPENINGS, pick
from pipeline import QuerySnapshotPipeline


def main():
    load_dotenv(Path(__file__).resolve().parent.parent / '.env')
    config = AppConfig()
    cap = cv2.VideoCapture(config.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    pipeline = QuerySnapshotPipeline(config)
    pipeline.audio.speak(pick(OPENINGS))
    print("Controls: v=speak one natural English question, q=quit")
    print("Examples: 'What objects are in front of me?', 'I want to pick up the cup', 'Where is the cup?', 'Did I get the cup?'")

    last_frame = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        last_frame = frame.copy()
        output = pipeline.draw_status(frame)
        cv2.imshow(config.window_name, output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('v'):
            text, err = pipeline.listen_for_command()
            if err:
                continue
            pipeline.handle_voice_text(text)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
