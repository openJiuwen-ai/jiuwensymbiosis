# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters._common.calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jiuwensymbiosis.adapters._common.calibration import (
    LegacyCalibrationError,
    load_calibration,
)


@pytest.fixture
def calib_dir(tmp_path):
    return tmp_path


def _write_calib(path: Path, *, frame_field: str = "T_J3link_cam", legacy: bool = False, version: int = 2):
    data = {
        "schema_version": version,
        "intrinsics": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        "object": {"xyz_base_mm": [100, 200, 300]},
    }
    field_name = "T_TCP_cam" if legacy else frame_field
    data[field_name] = {"matrix_4x4": np.eye(4).tolist()}
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


class TestLoadCalibration:
    def test_valid_new_schema(self, calib_dir):
        p = calib_dir / "calib.json"
        _write_calib(p)
        result = load_calibration(str(p))
        assert "T_J3link_cam" in result
        assert isinstance(result["T_J3link_cam"]["matrix_4x4"], np.ndarray)
        assert result["intrinsics"].shape == (3, 3)

    def test_legacy_without_env_var_raises(self, calib_dir):
        p = calib_dir / "legacy.json"
        _write_calib(p, legacy=True, version=1)
        with pytest.raises(LegacyCalibrationError):
            load_calibration(str(p))

    def test_legacy_with_env_var(self, calib_dir, monkeypatch):
        monkeypatch.setenv("JIUWEN_ALLOW_LEGACY_CALIB", "1")
        p = calib_dir / "legacy.json"
        _write_calib(p, legacy=True, version=1)
        result = load_calibration(str(p))
        assert "T_J3link_cam" in result
        assert result["T_J3link_cam"].get("_legacy_remap_from") == "T_TCP_cam"

    def test_missing_file_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            load_calibration("/nonexistent/path.json")

    def test_malformed_raises(self, calib_dir):
        p = calib_dir / "bad.json"
        p.write_text("not json at all", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_calibration(str(p))

    def test_missing_both_fields_raises(self, calib_dir):
        p = calib_dir / "empty.json"
        data = {
            "schema_version": 2,
            "intrinsics": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            "object": {"xyz_base_mm": [100, 200, 300]},
        }
        p.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="malformed"):
            load_calibration(str(p))
