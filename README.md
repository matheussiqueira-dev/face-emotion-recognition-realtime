# EmotionAI: Real-time Face Emotion Recognition

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg?style=for-the-badge&logo=opencv&logoColor=white)
![DeepFace](https://img.shields.io/badge/DeepFace-AI%20Analysis-orange.svg?style=for-the-badge)

**EmotionAI** is a high-performance, real-time facial emotion recognition system. It combines a robust Python backend leveraging OpenCV and DeepFace with a state-of-the-art web dashboard for real-time analytics and visualization.

---

## 🌟 Features

- **Real-time Detection & Tracking**: Advanced face detection using Haar Cascades with persistent ID tracking across frames.
- **Deep Emotion Analysis**: Powered by DeepFace, detecting 7 core emotions: *Happy, Sad, Angry, Surprise, Neutral, Fear, and Disgust*.
- **Modern Web Dashboard**: A premium, glassmorphism-inspired UI for live monitoring.
- **Real-time Analytics**: Dynamic charts showing emotion distribution and sentiment timeline.
- **High Performance**: Asynchronous processing with FastAPI and WebSockets for low-latency streaming.
- **Session Management**: Automated recording and session analysis (configurable).

## 🛠️ Tech Stack

- **Backend**: Python 3.8+, FastAPI, OpenCV, NumPy, DeepFace, TensorFlow/Keras.
- **Frontend**: Vanilla HTML5, Modern CSS (Glassmorphism), JavaScript (ES6+), Chart.js.
- **Communication**: WebSockets (Bi-directional real-time data).

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher.
- A functional webcam.

### Installation & Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/matheussiqueira-dev/face-emotion-recognition.git
   cd face-emotion-recognition
   ```

2. **Run the application**:
   Simply execute the provided batch file (Windows):
   ```bash
   run.bat
   ```
   Or manually:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   python run_api.py
   ```

3. **Access the Dashboard**:
   Open your browser and navigate to: `http://127.0.0.1:8000`

---

## 📂 Project Structure

```text
├── app/
│   ├── backend/          # FastAPI API and WebSocket logic
│   ├── core/             # Core processing (detection, tracking, analysis)
│   └── frontend/         # Web dashboard (HTML, CSS, JS)
├── run.bat               # One-click startup script
├── run_api.py            # Entry point for the web server
└── requirements.txt      # Project dependencies
```

---

## 🛠️ Configuration

You can customize the application behavior in `app/core/config.py`:
- `video_source`: Change between camera index (0, 1...) or a video file path.
- `emotion_interval`: Frequency of emotion analysis (seconds).
- `detect_scale`: Resolution scaling for faster detection.

## 📈 Future Improvements

- [ ] Support for multiple detection backends (MediaPipe, MTCNN).
- [ ] Export session data to PDF/CSV reports.
- [ ] Multi-camera support.
- [ ] User authentication and cloud sync.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

Autoria: Matheus Siqueira  
Website: [https://www.matheussiqueira.dev/](https://www.matheussiqueira.dev/)
