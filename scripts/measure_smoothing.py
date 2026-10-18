"""Opt-in: count label changes per face with and without temporal smoothing on the real demo video.

Both series come from the same run: the server reports the smoothed label ("dominant") and the model's
label for the latest inference ("raw_dominant"). Results depend on the video, model and settings; they are
a sanity check of the smoothing, not an accuracy benchmark.

Usage: python scripts/measure_smoothing.py [frames] [smoothing]
"""

import json
import os
import sys
from collections import defaultdict

from demo_server import ROOT, demo_server
from websockets.sync.client import connect


def count_changes(labels: list[str]) -> int:
    return sum(1 for previous, current in zip(labels, labels[1:]) if previous != current)


def main() -> None:
    frames = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    smoothing = sys.argv[2] if len(sys.argv) > 2 else "0.5"
    os.chdir(ROOT)
    raw: dict[int, list[str]] = defaultdict(list)
    smoothed: dict[int, list[str]] = defaultdict(list)
    with demo_server({"EMOTION_SMOOTHING": smoothing}) as (url, token, output):
        with connect(url.replace("http://", "ws://") + "/api/v1/ws/stream", max_size=4_000_000) as ws:
            ws.send(json.dumps({"type": "auth", "token": token}))
            for _ in range(frames):
                payload = json.loads(ws.recv(timeout=60))
                for track in payload["data"]["tracks"]:
                    emotion = track["emotion"]
                    if emotion:
                        raw[track["id"]].append(emotion["raw_dominant"])
                        smoothed[track["id"]].append(emotion["dominant"])

    summary = {
        "frames": frames,
        "emotion_smoothing": float(smoothing),
        "tracks_with_labels": len(raw),
        "labelled_observations": sum(len(labels) for labels in raw.values()),
        "raw_label_changes": sum(count_changes(labels) for labels in raw.values()),
        "smoothed_label_changes": sum(count_changes(labels) for labels in smoothed.values()),
    }
    (output / "smoothing-results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
