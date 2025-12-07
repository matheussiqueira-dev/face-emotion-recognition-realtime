# Face Emotion Recognition Realtime

**Developed by Matheus Siqueira**

## Project Overview

This project implements a real-time facial emotion recognition system using Python. It leverages Computer Vision techniques and Deep Learning models to capture video from a webcam, detect faces, and classify emotional expressions (such as happiness, sadness, neutrality, and anger) with high accuracy.

The system is designed to be robust and easy to deploy, featuring automatic environment setup and video recording capabilities.

## Key Features

- **Real-time Face Detection**: Utilizes OpenCV Haar Cascades for efficient face tracking.
- **Emotion Classification**: Integrates the DeepFace library to analyze facial attributes and predict emotions.
- **Live Visualization**: Overlays bounding boxes, emotion labels, and confidence percentages on the video feed.
- **Session Recording**: Automatically records the session and saves it as `output_preview.avi`.
- **Fault Tolerance**: Includes fallback mechanisms for model loading and detection resources.

## Technologies Used

- **Language**: Python 3.11
- **Computer Vision**: OpenCV (cv2)
- **Deep Learning**: DeepFace, TensorFlow/Keras
- **Automation**: Batch scripting for environment management

## Prerequisites

- Python 3.11 is recommended for optimal compatibility with OpenCV and TensorFlow on Windows systems.
- A functional webcam.

## Installation and Setup

### Automated Setup (Recommended)

1. Navigate to the project folder.
2. Double-click the `run.bat` script.
   - This script will automatically create the virtual environment, install all dependencies, and launch the application.

### Manual Installation

If you prefer to configure the environment manually:

1. Create a virtual environment using Python 3.11:
   ```powershell
   py -3.11 -m venv .venv
   ```

2. Activate the virtual environment:
   ```powershell
   .\.venv\Scripts\Activate
   ```

3. Install the required dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Usage

To run the application via terminal:

```powershell
python main.py
```

### Controls

- **Esc**: Press the 'Esc' key to close the application window and save the recording.

## Output

The application generates a video file named `output_preview.avi` in the project root directory, containing the recorded session with all visual overlays.
