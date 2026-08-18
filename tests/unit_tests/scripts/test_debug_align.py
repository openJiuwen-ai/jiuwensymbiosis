# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Offline render test for the head RGB↔mask↔cloud alignment debug view. Feeds a synthetic
organized cloud + masks straight into ``render_alignment`` (no ROS, no detector, no GUI) and checks
it returns a 3-panel BGR image. Also covers the flat-cloud reshape and the no-cloud placeholder."""

import numpy as np
import pytest

pytest.importorskip("cv2")   # opencv ships in the [full] / [calib] extras, not [dev]

from scripts.cruzr.debug_align import render_alignment  # noqa: E402


def _scene():
    h = w = 30
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    cloud = np.full((h, w, 3), np.nan, dtype=np.float64)
    cloud[10:20, 10:20] = (1.0, 0.0, 0.75)          # a textured patch with valid points
    tgt = np.zeros((h, w), dtype=bool)
    tgt[10:20, 10:20] = True                         # target mask over the valid patch
    ref = np.zeros((h, w), dtype=bool)
    ref[10:20, 10:20] = True
    tf = np.eye(4, dtype=np.float64)
    dets = [{"role": "target", "name": "white box", "mask": tgt},
            {"role": "reference", "name": "brown box", "mask": ref}]
    return rgb, cloud, tf, dets


def test_render_alignment_three_panels():
    rgb, cloud, tf, dets = _scene()
    img = render_alignment(rgb, cloud, tf, dets=dets, min_valid=30)
    assert img.dtype == np.uint8
    assert img.shape == (30, 30 * 3, 3)              # three panels side by side


def test_render_alignment_flat_cloud_reshaped():
    rgb, cloud, tf, dets = _scene()
    flat = cloud.reshape(1, 30 * 30, 3)              # unorganized but count matches → reshaped
    img = render_alignment(rgb, flat, tf, dets=dets, min_valid=30)
    assert img.shape == (30, 30 * 3, 3)


def test_render_alignment_no_cloud_placeholder():
    rgb, _cloud, _tf, dets = _scene()
    img = render_alignment(rgb, None, None, dets=dets, min_valid=30)
    assert img.shape == (30, 30 * 3, 3)              # still 3 panels; panel 2 is the NO CLOUD placeholder
