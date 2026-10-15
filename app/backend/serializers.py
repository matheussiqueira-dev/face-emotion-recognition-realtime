from __future__ import annotations

from typing import Any


def serialize_track(track: Any) -> dict[str, Any]:
    """Wire format of a tracked face, shared by the server-stream and browser-camera websockets."""
    serialized: dict[str, Any] = {
        "id": track.id,
        "box": {"x": track.box.x, "y": track.box.y, "w": track.box.w, "h": track.box.h},
        "emotion": None,
    }
    if track.emotion:
        serialized["emotion"] = {
            "dominant": track.emotion.dominant,
            "scores": track.emotion.scores,
        }
    return serialized
