"""Opt-in real inference/HTTP/WebSocket smoke check. Downloads model weights only if missing."""

import json
import os
from urllib.request import Request, urlopen

import cv2
from demo_server import ROOT, demo_server, demo_video
from websockets.sync.client import connect


def main() -> None:
    os.chdir(ROOT)
    with demo_server() as (url, token, output):
        ws_url = url.replace("http://", "ws://")
        with connect(ws_url + "/api/v1/ws/stream", max_size=4_000_000) as ws:
            ws.send(json.dumps({"type": "auth", "token": token}))
            frames = [json.loads(ws.recv(timeout=60)) for _ in range(3)]
            assert all(frame["frame"] and frame["data"]["tracks"] for frame in frames)
            assert any(track["emotion"] for track in frames[-1]["data"]["tracks"])

        cap = cv2.VideoCapture(str(demo_video()))
        try:
            ok, frame = cap.read()
            assert ok, "Demo frame could not be read"
        finally:
            cap.release()
        ok, jpeg = cv2.imencode(".jpg", frame)
        assert ok
        with connect(ws_url + "/api/v1/ws/analyze") as ws:
            ws.send(json.dumps({"type": "auth", "token": token}))
            ready = json.loads(ws.recv(timeout=60))
            assert ready["type"] == "ready" and ready["analysis_enabled"]
            ws.send(jpeg.tobytes())
            result = json.loads(ws.recv(timeout=30))
            assert result["type"] == "analysis" and result["analysis_enabled"]
            assert any(track["emotion"] for track in result["tracks"])
        request = Request(url + "/api/v1/metrics", headers={"Authorization": f"Bearer {token}"})
        with urlopen(request, timeout=5) as response:
            metrics = json.load(response)
        assert metrics["expression_observations"] == metrics["analyzed_faces"]
        summary = {
            "server_frames": len(frames),
            "browser_protocol": "passed",
            "sample_latency_ms": result["latency_ms"],
            "metrics_contract": "passed",
        }
        (output / "demo-results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
