import pytest

cv2 = pytest.importorskip("cv2")

import main  # noqa: E402


@pytest.mark.parametrize(
    ("gui_line", "expected"),
    [
        ("  GUI:                           NONE", False),
        ("  GUI:                           GTK3", True),
        ("  GUI:                           WIN32UI", True),
    ],
)
def test_has_gui_support_reads_build_information(monkeypatch, gui_line, expected):
    build_info = f"General configuration for OpenCV:\n{gui_line}\n  Media I/O:\n"
    monkeypatch.setattr(cv2, "getBuildInformation", lambda: build_info)
    assert main.has_gui_support() is expected


def test_main_exits_with_guidance_on_headless_build(monkeypatch, capsys):
    monkeypatch.setattr(main, "has_gui_support", lambda: False)
    assert main.main() == 1
    assert "run_api.py" in capsys.readouterr().out
