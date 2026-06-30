# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters.piper.geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jiuwensymbiosis.adapters._common.geometry import make_transform
from jiuwensymbiosis.adapters.piper.geometry import (
    FlangePose,
    pixel_and_depth_to_base_xyz,
    rpy_deg_to_rot,
)


class TestFlangePose:
    def test_construction(self):
        p = FlangePose(1, 2, 3, 0, 0, 0)
        assert p.x_mm == 1
        assert p.y_mm == 2
        assert p.z_mm == 3

    def test_frozen(self):
        p = FlangePose(1, 2, 3, 0, 0, 0)
        with pytest.raises(AttributeError):
            p.x_mm = 10

    def test_to_tf_base_flange_translation(self):
        p = FlangePose(100, 200, 300, 0, 0, 0)
        T = p.to_tf_base_flange()
        np.testing.assert_array_almost_equal(T[:3, 3], [100, 200, 300])
        np.testing.assert_array_almost_equal(T[:3, :3], np.eye(3), decimal=10)

    def test_to_tf_base_flange_rotation(self):
        p = FlangePose(0, 0, 0, 0, 90, 0)
        T = p.to_tf_base_flange()
        R = T[:3, :3]
        assert abs(abs(R[2, 0]) - 1.0) < 1e-6


class TestRpyDegToRot:
    def test_identity(self):
        R = rpy_deg_to_rot(0, 0, 0)
        np.testing.assert_array_almost_equal(R, np.eye(3), decimal=10)

    def test_90yaw(self):
        R = rpy_deg_to_rot(0, 0, 90)
        assert abs(R[0, 1] + 1) < 1e-6 or abs(R[1, 0] - 1) < 1e-6

    def test_consistent_with_scipy(self):
        rx, ry, rz = 10.0, 20.0, 30.0
        R = rpy_deg_to_rot(rx, ry, rz)
        R_sp = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
        np.testing.assert_array_almost_equal(R, R_sp, decimal=10)


class TestPixelAndDepthToBaseXyz:
    def test_known_projection(self):
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        flange = FlangePose(200.0, 0.0, 400.0, 0.0, 90.0, 0.0)
        tf_flange_cam = make_transform(np.eye(3), np.array([0.0, 0.0, 0.0]))
        result = pixel_and_depth_to_base_xyz(
            uv=(320.0, 240.0),
            depth_m=1.0,
            flange_pose=flange,
            tf_flange_cam=tf_flange_cam,
            intrinsics=K,
        )
        assert result.shape == (3,)
        assert all(math.isfinite(v) for v in result)
