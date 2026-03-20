import logging
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

if sys.version_info >= (3, 12):
    logging.warning("Python 3.12+ may not work well with MediaPipe on Windows. Python 3.10 or 3.11 is recommended.")

from main import main


if __name__ == "__main__":
    main()
