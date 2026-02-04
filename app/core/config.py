from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

@dataclass
class AppConfig:
    """Configuration for the Face Emotion Recognition application."""
    # Video Input/Output
    video_source: str = "0"  # Camera index or path to video file
    width: Optional[int] = 1280
    height: Optional[int] = 720
    fps: Optional[float] = 30.0
    output_path: Path = Path("output/recordings")
    record_session: bool = True
    
    # Detection Settings
    detect_scale: float = 0.5
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_face_size: int = 30
    cascade_path: Optional[Path] = None
    
    # Analysis Settings
    emotion_interval: float = 0.5  # Seconds between analyses per face
    min_emotion_score: float = 0.0
    
    # Tracking Settings
    track_ttl: float = 1.5
    iou_threshold: float = 0.3
    
    # Performance
    max_fps: int = 30
    enable_gpu: bool = True
    
    # UI/Visuals
    draw_landmarks: bool = True
    show_fps: bool = True
    theme_color: str = "#6366f1"  # Indigo-500
