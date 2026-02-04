import sys
import logging
from app.core.config import AppConfig
from app.core.processor import VideoProcessor

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Use default config
    config = AppConfig()
    
    # Override with command line if needed (simplified for this wrapper)
    if len(sys.argv) > 1:
        config.video_source = sys.argv[1]

    print("--- EmotionAI Legacy Wrapper ---")
    print("Starting process with OpenCV fallback display...")
    
    processor = VideoProcessor(config)
    try:
        processor.run() # This runs without callback, using internal cv2.imshow
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
