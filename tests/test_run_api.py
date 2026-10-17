import io
import sys

import run_api


def test_launcher_starts_with_windows_log_encoding(monkeypatch):
    """Redirected Windows consoles can use CP1252 instead of UTF-8."""
    output = io.BytesIO()
    stdout = io.TextIOWrapper(output, encoding="cp1252")
    calls = []
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "argv", ["run_api.py", "--port", "8765"])
    monkeypatch.setattr(run_api.uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    run_api.main()
    stdout.flush()

    assert "http://127.0.0.1:8765" in output.getvalue().decode("cp1252")
    assert calls == [
        (
            ("app.backend.api:app",),
            {"host": "127.0.0.1", "port": 8765, "reload": False, "ws_max_size": 2_004_096, "ws_max_queue": 2},
        )
    ]
