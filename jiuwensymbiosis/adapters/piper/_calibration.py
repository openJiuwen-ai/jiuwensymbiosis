# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Piper calibration JSON loader.

Thin wrapper over ``perception.calibration.load_calibration`` with
Piper-specific defaults: the camera pose lives in ``tf_flange_cam`` (6-DoF
eye-in-hand on the wrist).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jiuwensymbiosis.perception.calibration import (
    CURRENT_SCHEMA_VERSION,
    LegacyCalibrationError,
)
from jiuwensymbiosis.perception.calibration import (
    load_calibration as _generic_load_calibration,
)

__all__ = ["load_calibration", "LegacyCalibrationError", "CURRENT_SCHEMA_VERSION"]


def load_calibration(path: str | Path) -> dict[str, Any]:
    """Load a Piper calibration JSON. See ``perception.calibration`` for the schema."""
    return _generic_load_calibration(
        path,
        frame_field="T_flange_cam",
        legacy_field="T_TCP_cam",
        env_var="JIUWEN_PIPER_ALLOW_LEGACY_CALIB",
    )
