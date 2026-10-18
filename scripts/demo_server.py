"""Isolated loopback server for opt-in real-model checks. Never opens a physical camera."""

from __future__ import annotations

import os
import secrets
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def demo_server(extra_env: dict[str, str] | None = None) -> Iterator[tuple[str, str, Path]]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    output = ROOT / ".tmp" / "verification"
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / f"server-{port}.log"
    token = secrets.token_urlsafe(32)
    env = {
        **os.environ,
        "PYTHONUTF8": "1",
        "EMOTION_VIDEO_SOURCE": "Face-emotions.mp4",
        "EMOTION_API_READ_TOKEN": token,
        "EMOTION_API_ADMIN_TOKEN": secrets.token_urlsafe(32),
        "EMOTION_CORS_ORIGINS": "",
        **(extra_env or {}),
    }
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-X", "utf8", "run_api.py", "--host", "127.0.0.1", "--port", str(port)],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            url = f"http://127.0.0.1:{port}"
            for _ in range(80):
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited; inspect {log_path}")
                try:
                    with urlopen(url + "/api/v1/health", timeout=1):
                        break
                except URLError:
                    time.sleep(0.25)
            else:
                raise TimeoutError(f"Server startup timed out; inspect {log_path}")
            yield url, token, output
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
