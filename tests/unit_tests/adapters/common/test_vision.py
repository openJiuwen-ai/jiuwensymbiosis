# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters._common.vision."""

from __future__ import annotations

import numpy as np

from jiuwensymbiosis.adapters._common.vision import (
    _mask_centroid,
    _median_depth_window,
    apply_xy_correction,
)


class TestMaskCentroid:
    def test_center_mask(self):
        mask = np.zeros((100, 200), dtype=bool)
        mask[45:55, 95:105] = True
        result = _mask_centroid({"mask": mask}, img_w=200, img_h=100, log_prefix="[test]")
        assert abs(result["u"] - 100) < 2
        assert abs(result["v"] - 50) < 2

    def test_corner_mask(self):
        mask = np.zeros((100, 200), dtype=bool)
        mask[0:10, 0:10] = True
        result = _mask_centroid({"mask": mask}, img_w=200, img_h=100, log_prefix="[test]")
        assert result["u"] < 15
        assert result["v"] < 10


class TestMedianDepthWindow:
    def test_valid_depth(self):
        depth = np.ones((480, 640), dtype=np.float32) * 0.5
        result = _median_depth_window(depth, 320.0, 240.0, "[test]")
        assert result is not None
        assert abs(result - 0.5) < 0.01

    def test_zero_depth_returns_none(self):
        depth = np.zeros((480, 640), dtype=np.float32)
        result = _median_depth_window(depth, 320.0, 240.0, "[test]")
        assert result is None


class TestApplyXyCorrection:
    def test_no_correction(self):
        xyz = np.array([100.0, 200.0, 300.0])
        result, desc = apply_xy_correction(xyz)
        np.testing.assert_array_almost_equal(result, xyz)
        assert desc == "none"

    def test_translation_correction(self):
        xyz = np.array([100.0, 200.0, 300.0])
        result, desc = apply_xy_correction(xyz, xy_correction_mm=[5.0, -10.0])
        np.testing.assert_array_almost_equal(result[0], 105.0)
        np.testing.assert_array_almost_equal(result[1], 190.0)
        np.testing.assert_array_almost_equal(result[2], 300.0)

    def test_affine_correction(self):
        xyz = np.array([100.0, 200.0, 300.0])
        A = np.eye(2)
        b = np.array([5.0, -10.0])
        result, desc = apply_xy_correction(
            xyz,
            xy_transform={"A": A.tolist(), "b": b.tolist(), "method": "affine", "n_samples": 3, "rms_residual_mm": 1.2},
        )
        np.testing.assert_array_almost_equal(result[0], 105.0)
        np.testing.assert_array_almost_equal(result[1], 190.0)

    def test_affine_priority_over_translation(self):
        xyz = np.array([100.0, 200.0, 300.0])
        A = np.array([[1.1, 0.0], [0.0, 0.9]])
        b = np.array([5.0, -10.0])
        result, desc = apply_xy_correction(
            xyz,
            xy_transform={"A": A.tolist(), "b": b.tolist(), "method": "affine", "n_samples": 3, "rms_residual_mm": 1.0},
            xy_correction_mm=[999.0, 999.0],
        )
        assert "xy_transform" in desc
        assert "affine" in desc
