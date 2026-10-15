"""Check standalone backend selection without importing Isaac or starting a server."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("shadowed", [False, True])
def test_launcher_loads_only_the_requested_checkout(tmp_path, shadowed):
    repo = tmp_path / "backend"
    package = repo / "sim_service"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "main.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "def create_app(**kwargs):\n"
        "    assert Path.cwd() == Path(__file__).resolve().parents[1]\n"
        "    assert kwargs == dict(headless=True, streaming=False, shutdown_sim_on_lifespan=False)\n"
        "    raise SystemExit(0)\n"
        "def run_service_loop(*args):\n    raise AssertionError('must not start Isaac')\n",
        encoding="utf-8",
    )
    dependencies = tmp_path / "dependencies"
    dependencies.mkdir()
    (dependencies / "uvicorn.py").write_text("", encoding="utf-8")
    if shadowed:
        shadow = dependencies / "sim_service"
        shadow.mkdir()
        (shadow / "__init__.py").write_text("raise AssertionError('shadow imported')", encoding="utf-8")
    launcher = Path(__file__).resolve().parents[4] / "scripts" / "run_robosim_service.py"
    result = subprocess.run(
        [sys.executable, str(launcher), "--repo", str(repo), "--headless"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(dependencies)},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if shadowed:
        assert result.returncode == 2
        assert "sim_service resolves outside --repo" in result.stderr
        assert "shadow imported" not in result.stderr
    else:
        assert result.returncode == 0, result.stderr
