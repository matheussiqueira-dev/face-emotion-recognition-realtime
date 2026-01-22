from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

try:
    from deepface import DeepFace  # type: ignore
    _DEEPFACE_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - runtime dependent
    DeepFace = None  # type: ignore[assignment]
    _DEEPFACE_ERROR = exc

LOGGER = logging.getLogger("face_emotion")


@dataclass
class AppConfig:
    camera_index: int = 0
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    output_path: Optional[Path] = Path("output_preview.avi")
    record: bool = True
    display: bool = True
    mirror: bool = True
    max_frames: int = 0
    cascade_path: Optional[Path] = None
    detect_scale: float = 0.5
    scale_factor: float = 1.1
    min_neighbors: int = 4
    min_face_size: int = 30
    emotion_min_size: int = 40
    emotion_interval: float = 0.5
    track_ttl: float = 1.25
    iou_threshold: float = 0.3
    draw_fps: bool = True
    log_level: str = "INFO"
    output_codec: str = "XVID"


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def center(self) -> Tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0


@dataclass
class EmotionResult:
    dominant: str
    scores: Dict[str, float]
    timestamp: float


@dataclass
class Track:
    track_id: int
    box: FaceBox
    last_seen: float
    last_analyzed: float = 0.0
    emotion: Optional[EmotionResult] = None


class FPSMeter:
    def __init__(self, smoothing: float = 0.9) -> None:
        self._last_time: Optional[float] = None
        self._fps: float = 0.0
        self._smoothing = smoothing

    def update(self, now: float) -> float:
        if self._last_time is None:
            self._last_time = now
            return self._fps
        elapsed = now - self._last_time
        self._last_time = now
        if elapsed <= 0:
            return self._fps
        instant = 1.0 / elapsed
        if self._fps == 0.0:
            self._fps = instant
        else:
            self._fps = (self._fps * self._smoothing) + (instant * (1.0 - self._smoothing))
        return self._fps


class FaceDetector:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        cascade_path = self._resolve_cascade_path(config.cascade_path)
        LOGGER.info("Using cascade: %s", cascade_path)
        self._classifier = cv2.CascadeClassifier(str(cascade_path))
        if self._classifier.empty():
            raise RuntimeError(f"Failed to load Haar Cascade from {cascade_path}")

    def detect(self, frame: np.ndarray) -> List[FaceBox]:
        scale = self._config.detect_scale
        if not (0.2 <= scale <= 1.0):
            raise ValueError("detect_scale must be between 0.2 and 1.0")

        if scale != 1.0:
            small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        else:
            small = frame

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        min_size = max(1, int(self._config.min_face_size * scale))
        faces = self._classifier.detectMultiScale(
            gray,
            scaleFactor=self._config.scale_factor,
            minNeighbors=self._config.min_neighbors,
            minSize=(min_size, min_size),
        )

        if scale == 1.0:
            return [FaceBox(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

        inv_scale = 1.0 / scale
        boxes: List[FaceBox] = []
        for (x, y, w, h) in faces:
            boxes.append(
                FaceBox(
                    int(x * inv_scale),
                    int(y * inv_scale),
                    int(w * inv_scale),
                    int(h * inv_scale),
                )
            )
        return boxes

    @staticmethod
    def _resolve_cascade_path(custom_path: Optional[Path]) -> Path:
        candidates: List[Path] = []
        if custom_path:
            candidates.append(custom_path)
        candidates.append(Path.cwd() / "haarcascade_frontalface_default.xml")
        candidates.append(Path(__file__).with_name("haarcascade_frontalface_default.xml"))
        if hasattr(cv2, "data"):
            candidates.append(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        for path in candidates:
            if path and path.exists():
                return path
        joined = ", ".join(str(p) for p in candidates if p)
        raise FileNotFoundError(
            "Haar Cascade XML not found. Provide --cascade or place "
            "haarcascade_frontalface_default.xml in the project directory. "
            f"Tried: {joined}"
        )


class EmotionAnalyzer:
    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled and DeepFace is not None
        if enabled and DeepFace is None:
            LOGGER.warning("DeepFace unavailable, emotion analysis disabled: %s", _DEEPFACE_ERROR)
        if self._enabled:
            self._warm_up()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def analyze(self, face_img: np.ndarray) -> Optional[EmotionResult]:
        if not self._enabled:
            return None
        try:
            result = DeepFace.analyze(  # type: ignore[misc]
                img_path=face_img,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="skip",
            )
        except Exception as exc:  # pragma: no cover - runtime dependent
            LOGGER.debug("Emotion analysis failed: %s", exc)
            return None
        if isinstance(result, list) and result:
            result = result[0]
        if not isinstance(result, dict):
            return None
        dominant = result.get("dominant_emotion")
        scores = result.get("emotion")
        if not dominant or not isinstance(scores, dict):
            return None
        return EmotionResult(dominant=dominant, scores=scores, timestamp=time.time())

    @staticmethod
    def _warm_up() -> None:
        dummy = np.zeros((48, 48, 3), dtype=np.uint8)
        try:
            DeepFace.analyze(  # type: ignore[misc]
                img_path=dummy,
                actions=["emotion"],
                enforce_detection=False,
            )
        except Exception as exc:  # pragma: no cover - runtime dependent
            LOGGER.warning("DeepFace warm-up failed: %s", exc)


class FaceTracker:
    def __init__(self, ttl_seconds: float, iou_threshold: float) -> None:
        self._ttl = ttl_seconds
        self._iou_threshold = iou_threshold
        self._tracks: List[Track] = []
        self._next_id = 1

    def update(self, detections: Iterable[FaceBox], now: float) -> List[Track]:
        detections_list = list(detections)
        assigned: List[Track] = []
        used_ids: set[int] = set()

        for det in detections_list:
            best_track: Optional[Track] = None
            best_iou = 0.0
            for track in self._tracks:
                if track.track_id in used_ids:
                    continue
                score = _iou(track.box, det)
                if score > best_iou:
                    best_iou = score
                    best_track = track
            if best_track and best_iou >= self._iou_threshold:
                best_track.box = det
                best_track.last_seen = now
                assigned.append(best_track)
                used_ids.add(best_track.track_id)
            else:
                new_track = Track(track_id=self._next_id, box=det, last_seen=now)
                self._next_id += 1
                self._tracks.append(new_track)
                assigned.append(new_track)
                used_ids.add(new_track.track_id)

        self._tracks = [t for t in self._tracks if (now - t.last_seen) <= self._ttl]
        return assigned


class VideoProcessor:
    def __init__(self, config: AppConfig, detector: FaceDetector, analyzer: EmotionAnalyzer) -> None:
        self._config = config
        self._detector = detector
        self._analyzer = analyzer
        self._tracker = FaceTracker(config.track_ttl, config.iou_threshold)
        self._fps_meter = FPSMeter()

    def run(self) -> int:
        cap = cv2.VideoCapture(self._config.camera_index)
        if not cap.isOpened():
            LOGGER.error("Unable to open webcam index %s", self._config.camera_index)
            return 2

        if self._config.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        if self._config.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        if self._config.fps:
            cap.set(cv2.CAP_PROP_FPS, self._config.fps)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or (self._config.width or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or (self._config.height or 480)
        output_fps = cap.get(cv2.CAP_PROP_FPS) or (self._config.fps or 20.0)
        frame_size = (width, height)

        writer = self._create_writer(frame_size, output_fps) if self._config.record else None

        LOGGER.info("Running capture %sx%s at %.2f fps", width, height, output_fps)
        if self._config.display:
            LOGGER.info("Press Esc to stop.")

        frame_count = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    LOGGER.warning("Empty frame received.")
                    continue

                if self._config.mirror:
                    frame = cv2.flip(frame, 1)
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, frame_size)

                now = time.time()
                boxes = self._detector.detect(frame)
                tracks = self._tracker.update(boxes, now)

                for track in tracks:
                    box = track.box
                    _draw_box(frame, box)
                    label = self._label_for_track(track, frame, now)
                    _draw_label(frame, box, label)

                if self._config.draw_fps:
                    fps = self._fps_meter.update(now)
                    _draw_fps(frame, fps)

                if writer is not None:
                    writer.write(frame)

                if self._config.display:
                    cv2.imshow("Face Emotion Recognition", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                frame_count += 1
                if self._config.max_frames and frame_count >= self._config.max_frames:
                    LOGGER.info("Max frames reached (%s).", self._config.max_frames)
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if self._config.display:
                cv2.destroyAllWindows()
        return 0

    def _label_for_track(self, track: Track, frame: np.ndarray, now: float) -> str:
        if not self._analyzer.enabled:
            return "Emotion: N/A"
        if track.box.w < self._config.emotion_min_size or track.box.h < self._config.emotion_min_size:
            return "Face"

        if (now - track.last_analyzed) >= self._config.emotion_interval:
            face_roi = _safe_crop(frame, track.box)
            if face_roi.size:
                result = self._analyzer.analyze(face_roi)
                if result:
                    track.emotion = result
                    track.last_analyzed = now

        if track.emotion:
            score = track.emotion.scores.get(track.emotion.dominant, 0.0)
            return f"{track.emotion.dominant} ({score:.0f}%)"
        return "Analyzing..."

    def _create_writer(self, frame_size: Tuple[int, int], fps: float) -> Optional[cv2.VideoWriter]:
        if not self._config.output_path:
            return None
        output_path = self._config.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self._config.output_codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), frame_size)
        if not writer.isOpened():
            LOGGER.warning("Failed to open video writer at %s", output_path)
            return None
        LOGGER.info("Recording to %s", output_path)
        return writer


def _iou(a: FaceBox, b: FaceBox) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def _safe_crop(frame: np.ndarray, box: FaceBox) -> np.ndarray:
    x1 = max(0, box.x)
    y1 = max(0, box.y)
    x2 = min(frame.shape[1], box.x + box.w)
    y2 = min(frame.shape[0], box.y + box.h)
    return frame[y1:y2, x1:x2]


def _draw_box(frame: np.ndarray, box: FaceBox) -> None:
    cv2.rectangle(frame, (box.x, box.y), (box.x + box.w, box.y + box.h), (0, 255, 0), 2)


def _draw_label(frame: np.ndarray, box: FaceBox, text: str) -> None:
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = box.x
    text_y = max(0, box.y - 10)
    bg_top = max(0, text_y - text_size[1] - 6)
    bg_right = min(frame.shape[1], text_x + text_size[0] + 10)
    cv2.rectangle(frame, (text_x, bg_top), (bg_right, text_y + 4), (0, 255, 0), -1)
    cv2.putText(frame, text, (text_x + 4, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def _draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Real-time face emotion recognition.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0).")
    parser.add_argument("--width", type=int, help="Capture width.")
    parser.add_argument("--height", type=int, help="Capture height.")
    parser.add_argument("--fps", type=float, help="Preferred capture fps.")
    parser.add_argument("--cascade", type=Path, help="Path to Haar Cascade XML file.")
    parser.add_argument("--detect-scale", type=float, default=0.5, help="Downscale factor for detection.")
    parser.add_argument("--scale-factor", type=float, default=1.1, help="Haar cascade scale factor.")
    parser.add_argument("--min-neighbors", type=int, default=4, help="Haar cascade min neighbors.")
    parser.add_argument("--min-face-size", type=int, default=30, help="Minimum face size (px).")
    parser.add_argument("--emotion-min-size", type=int, default=40, help="Min face size for emotion analysis.")
    parser.add_argument("--emotion-interval", type=float, default=0.5, help="Seconds between analyses per track.")
    parser.add_argument("--track-ttl", type=float, default=1.25, help="Seconds to keep face tracks alive.")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="IOU threshold for tracking.")
    parser.add_argument("--no-record", action="store_true", help="Disable video recording.")
    parser.add_argument("--output", type=Path, default=Path("output_preview.avi"), help="Output video path.")
    parser.add_argument("--codec", type=str, default="XVID", help="FourCC codec (default: XVID).")
    parser.add_argument("--no-display", action="store_true", help="Disable live window (headless).")
    parser.add_argument("--no-mirror", action="store_true", help="Disable horizontal mirroring.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = infinite).")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")

    args = parser.parse_args()

    return AppConfig(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        output_path=args.output,
        record=not args.no_record,
        display=not args.no_display,
        mirror=not args.no_mirror,
        max_frames=args.max_frames,
        cascade_path=args.cascade,
        detect_scale=args.detect_scale,
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors,
        min_face_size=args.min_face_size,
        emotion_min_size=args.emotion_min_size,
        emotion_interval=args.emotion_interval,
        track_ttl=args.track_ttl,
        iou_threshold=args.iou_threshold,
        log_level=args.log_level,
        output_codec=args.codec,
    )


def main() -> int:
    config = _parse_args()
    _configure_logging(config.log_level)
    LOGGER.info("Starting Face Emotion Recognition.")
    analyzer = EmotionAnalyzer(enabled=True)
    detector = FaceDetector(config)
    processor = VideoProcessor(config, detector, analyzer)
    return processor.run()


if __name__ == "__main__":
    raise SystemExit(main())
