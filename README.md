# EmotionAI — Real-time Face Emotion Recognition Platform

EmotionAI is a real-time facial emotion recognition platform that blends a high-performance Python backend with a modern web dashboard for live monitoring. The system targets product teams, researchers, and operators who need fast visual insights from camera feeds while maintaining a clean, scalable architecture.

## 🎯 Purpose & Business Goals
- **Human insight at scale:** understand mood trends from live or recorded video streams.
- **Operational clarity:** provide a single dashboard to monitor emotion metrics, tracking performance, and system health.
- **Extensibility:** allow new detectors, emotion models, and data destinations to be integrated without rewrites.

## ✅ Key Features
- **Live detection and tracking** with stable IDs across frames.
- **Deep emotion analysis** using DeepFace (7 core emotions).
- **Real-time dashboard** with sentiment timeline and emotion distribution.
- **WebSocket streaming** for low-latency visual output.
- **Configurable pipeline** for detection scale, tracking TTL, and inference cadence.
- **API endpoints** for health and configuration telemetry.

---

## 🧠 Architecture Overview
EmotionAI follows a modular, clean architecture layout to separate concerns and make future evolution straightforward.

```
app/
  backend/         # API + WebSocket streaming (FastAPI)
  core/            # Detection, tracking, emotion analysis, and configuration
  frontend/        # Dashboard UI (HTML, CSS, JS)
```

### Core Flow
1. **Video capture** → OpenCV stream reads frames.
2. **Face detection** → Haar cascade detection in `FaceDetector`.
3. **Tracking** → IoU-based tracking assigns stable IDs.
4. **Emotion inference** → DeepFace analysis on the cropped face.
5. **Streaming** → Encoded frames + metadata via WebSocket to the frontend.

### Technical Decisions
- **FastAPI + WebSockets**: async-friendly and production-ready for real-time streaming.
- **Modular Core**: decoupled classes so each pipeline component can evolve independently.
- **Minimal frontend stack**: fast load times and easier deployment with static assets.

---

## 🛠️ Stack
- **Backend**: Python 3.8+, FastAPI, OpenCV, NumPy, DeepFace, TensorFlow/Keras
- **Frontend**: HTML5, Modern CSS, JavaScript (ES6+), Chart.js
- **Communication**: WebSockets

---

## 📦 Installation & Run

### 1. Clone & install
```bash
git clone https://github.com/matheussiqueira-dev/face-emotion-recognition.git
cd face-emotion-recognition
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Run the server
```bash
python run_api.py
```

### 3. Open dashboard
Visit: `http://127.0.0.1:8000`

---

## 🔌 API Endpoints
- `GET /api/health` → health check
- `GET /api/config` → current runtime config
- `WS /ws/stream` → video + emotion stream

---

## ⚙️ Configuration
All main runtime settings are defined in `app/core/config.py`:
- `video_source`: camera index or video file path
- `emotion_interval`: time between emotion inferences per track
- `detect_scale`: detection resolution downscale
- `track_ttl`: how long a face remains tracked without seeing it

---

## 🔒 Security & Reliability
- WebSocket transmission is isolated and only enabled on known origins by default.
- Graceful shutdown ensures video resources are released correctly.
- Emotion inference is disabled if DeepFace is unavailable to prevent runtime crashes.

---

## 🧪 Testing Strategy (Suggested)
To evolve into production-grade quality, the following test layers are recommended:
- **Unit tests** for detector/tracker/analyzer logic
- **Integration tests** for websocket payload contract
- **E2E tests** for dashboard rendering and responsiveness

---

## 🚀 Future Improvements
- MediaPipe and MTCNN detector support
- Multi-camera fusion dashboard
- Emotion history export (CSV, PDF)
- Persistent session storage
- Auth with role-based access control

---

Autoria: Matheus Siqueira  
Website: https://www.matheussiqueira.dev/
