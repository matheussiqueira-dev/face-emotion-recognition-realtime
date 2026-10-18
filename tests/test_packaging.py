import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _requirements(path: Path) -> list[str]:
    lines = (line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines())
    return sorted(line for line in lines if line and not line.startswith("-"))


def test_pyproject_dependencies_match_requirements_in():
    """Vercel installs from pyproject.toml; pip, CI and Docker from the requirements files. Keep them equal."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert sorted(project["dependencies"]) == _requirements(ROOT / "requirements.in")
