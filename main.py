import logging
import re
import sys

import cv2

from app.core.config import AppConfig
from app.core.processor import VideoProcessor


def has_gui_support() -> bool:
    """Headless OpenCV builds (the default dependency) cannot open windows."""
    return not re.search(r"^\s*GUI:\s*NONE\s*$", cv2.getBuildInformation(), re.MULTILINE)


def main():
    if not has_gui_support():
        print(
            "This legacy mode needs an OpenCV build with GUI support, but the installed one is headless.\n"
            "Use the web dashboard instead: python run_api.py\n"
            "Or swap the OpenCV build: pip uninstall opencv-contrib-python-headless && pip install opencv-contrib-python"
        )
        return 1

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Use default config
    config = AppConfig()

    # Override with command line if needed (simplified for this wrapper)
    if len(sys.argv) > 1:
        config.video_source = sys.argv[1]

    print("--- EmotionAI Legacy Wrapper ---")
    print("Starting process with OpenCV fallback display...")

    processor = VideoProcessor(config)
    try:
        processor.run()  # This runs without callback, using internal cv2.imshow
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
