# Face Emotion Recognition - Realtime

A real-time facial emotion recognition system using Computer Vision and Deep Learning.

## Features
- **Real-time Face Detection**: Uses OpenCV Haar Cascades for lightweight and compatible face tracking.
- **Emotion Recognition**: leverages `DeepFace` (using FER+ or other pre-trained models) to classify emotions (happy, sad, neutral, angry, etc.).
- **Live Visualization**: Displays bounding boxes and emotion labels with confidence scores on the webcam feed.

## Prerequisites
- Python 3.8+ (Tested on 3.13)
- Webcam

## Installation

1.  Clone the repository or download the files.
2.  Create a virtual environment:
    ```bash
    python -m venv .venv
    ```
3.  Activate the virtual environment:
    -   Windows: `.\.venv\Scripts\Activate`
    -   Linux/Mac: `source .venv/bin/activate`
4.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main script:

```bash
python main.py
```

-   The first run might take a moment to download the pre-trained emotion detection model.
-   Press `Esc` to quit the application.

## Technologies
-   [OpenCV](https://opencv.org/)
-   [MediaPipe](https://developers.google.com/mediapipe)
-   [DeepFace](https://github.com/serengil/deepface)
