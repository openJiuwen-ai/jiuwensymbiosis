"""Installed CLI and workbench resources must work outside the source checkout."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_installed_wheel_entries_and_workbench_resources(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    source = tmp_path / "source"
    source.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(repo / name, source / name)
    for name in ("jiuwensymbiosis", "jiuwensymbiosis_gui", "examples", "scripts"):
        shutil.copytree(repo / name, source / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    def run(args, *, cwd=tmp_path):
        result = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--no-index",
            "--wheel-dir",
            str(tmp_path / "dist"),
            str(source),
        ]
    )
    (wheel,) = (tmp_path / "dist").glob("*.whl")
    installed = tmp_path / "installed"
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--ignore-installed",
            "--target",
            str(installed),
            str(wheel),
        ]
    )
    # Remove the build tree too, so package-data fallback cannot hide omissions.
    shutil.rmtree(source)
    probe = """
import importlib.metadata
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
root = Path(sys.argv[1])
user_dir = root.parent / "user"
Path.home = classmethod(lambda cls: user_dir)
import examples
import jiuwensymbiosis
import jiuwensymbiosis_gui
from jiuwensymbiosis_gui.workbench import registry
for module in (examples, jiuwensymbiosis, jiuwensymbiosis_gui):
    assert Path(module.__file__).is_relative_to(root), module.__file__
gui = root / "jiuwensymbiosis_gui" / "workbench"
for filename in ("bodies.yaml", "tasks.yaml", "calibration_profiles.yaml"):
    assert (gui / "data" / filename).is_file(), filename
for filename in ("favicon.svg", "app_icon.png"):
    assert (gui / "assets" / filename).stat().st_size > 0, filename
for body in registry.list_bodies():
    config = body.config_path()
    assert config.is_relative_to(user_dir), config
    packaged = gui / "data" / "configs" / body.key / config.name
    assert config.read_bytes() == packaged.read_bytes()
dist = importlib.metadata.distribution("jiuwensymbiosis")
entry = next(e for e in dist.entry_points if e.name == sys.argv[2])
sys.argv = [entry.name, "--help"]
entry.load()()
"""
    for entry in ("jiuwensymbiosis-run", "jiuwensymbiosis-gui"):
        assert "--config" in run([sys.executable, "-I", "-c", probe, str(installed), entry])
    env["PYTHONPATH"] = str(installed)
    assert "--gui" in run([sys.executable, "-m", "jiuwensymbiosis.gui", "--help"])
