# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for cruzr camera calibration loader."""

from __future__ import annotations

import json

import numpy as np

from jiuwensymbiosis.adapters.cruzr._calibration import load_cruzr_camera_calib


def test_loads_both_fields(tmp_path):
    p = tmp_path / "calib.json"
    p.write_text(json.dumps({
        "intrinsics": [[100, 0, 4], [0, 100, 4], [0, 0, 1]],
        "tf_base_cam": np.eye(4).tolist(),
    }))
    out = load_cruzr_camera_calib(p)
    assert out["intrinsics"].shape == (3, 3)
    assert out["tf_base_cam"].shape == (4, 4)


def test_missing_fields_are_none(tmp_path):
    p = tmp_path / "calib.json"
    p.write_text(json.dumps({"intrinsics": [[100, 0, 4], [0, 100, 4], [0, 0, 1]]}))
    out = load_cruzr_camera_calib(p)
    assert out["intrinsics"].shape == (3, 3)
    assert out["tf_base_cam"] is None
