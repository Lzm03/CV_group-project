import logging
import cv2
from pathlib import Path
from dotenv import load_dotenv

from config import AppConfig
from phrases import OPENINGS, pick
from pipeline import QuerySnapshotPipeline


logger = logging.getLogger(__name__)
VOICE_FALLBACK_PROMPT = "Voice input is unavailable. Type a command and press Enter: "


def configure_logging(debug: bool):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="[%(levelname)s] %(message)s",
        force=True,
    )


def main():
    load_dotenv(Path(__file__).resolve().parent.parent / '.env')
    config = AppConfig()
    configure_logging(config.debug)
    cap = cv2.VideoCapture(config.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    pipeline = QuerySnapshotPipeline(config)
    pipeline.audio.speak(pick(OPENINGS))
    logger.info("Controls: v=voice question, k=type question, q=quit")
    logger.info("Examples: 'What objects are in front of me?', 'I want to pick up the cup', 'Where is the cup?', 'Did I get the cup?'")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            output = pipeline.draw_status(frame)
            cv2.imshow(config.window_name, output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v'):
                text, err = pipeline.listen_for_command()
                if err:
                    if not pipeline.speech_input.is_available():
                        typed_text = input(VOICE_FALLBACK_PROMPT).strip()
                        if typed_text:
                            pipeline.handle_voice_text(typed_text)
                    continue
                pipeline.handle_voice_text(text)
            elif key == ord('k'):
                text = input("Type a command and press Enter: ").strip()
                if text:
                    pipeline.handle_voice_text(text)
    finally:
        pipeline.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
