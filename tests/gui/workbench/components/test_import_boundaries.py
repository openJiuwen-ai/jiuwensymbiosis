# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workbench pages must not load the calibration subsystem just by being imported."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_importing_workbench_calibration_pages_keeps_calibration_lazy() -> None:
    """The calibration tool reports missing [calib] dependencies on use, not app import."""
    probe = """
import sys
import jiuwensymbiosis_gui.workbench
import jiuwensymbiosis_gui.workbench.pages.tools_view
import jiuwensymbiosis_gui.workbench.pages.calibration_view
offenders = sorted(name for name in sys.modules if name == 'jiuwensymbiosis.calibration' or name.startswith('jiuwensymbiosis.calibration.'))
assert not offenders, offenders
assert 'cv2' not in sys.modules, 'importing the workbench must not pull in OpenCV'
"""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(value for value in (str(_REPO_ROOT), existing) if value)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
