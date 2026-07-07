# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters._common.geometry."""

from __future__ import annotations

import numpy as np
import pytest

from jiuwensymbiosis.adapters._common.geometry import (
    _rot_z,
    apply_transform,
    invert_transform,
    make_transform,
    pixel_and_depth_to_camera_xyz,
)


class TestMakeTransform:
    def test_identity(self):
        T = make_transform(np.eye(3), np.zeros(3))
        assert np.allclose(T, np.eye(4))

    def test_pure_translation(self):
        t = np.array([10.0, 20.0, 30.0])
        T = make_transform(np.eye(3), t)
        np.testing.assert_array_almost_equal(T[:3, 3], t)
        np.testing.assert_array_almost_equal(T[:3, :3], np.eye(3))

    def test_rotation(self):
        R = _rot_z(90.0)
        T = make_transform(R, np.zeros(3))
        expected_R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        np.testing.assert_array_almost_equal(T[:3, :3], expected_R)


class TestApplyTransform:
    def test_origin_gets_translation(self):
        t = np.array([5.0, 6.0, 7.0])
        T = make_transform(np.eye(3), t)
        p = apply_transform(T, np.zeros(3))
        np.testing.assert_array_almost_equal(p, t)

    def test_roundtrip_with_inverse(self):
        R = _rot_z(45.0)
        t = np.array([100.0, 200.0, 300.0])
        T = make_transform(R, t)
        T_inv = invert_transform(T)
        p = np.array([10.0, 20.0, 30.0])
        p_back = apply_transform(T_inv, apply_transform(T, p))
        np.testing.assert_array_almost_equal(p_back, p, decimal=10)


class TestInvertTransform:
    def test_self_inverse(self):
        T = make_transform(_rot_z(30.0), np.array([1.0, 2.0, 3.0]))
        T_inv = invert_transform(T)
        identity = T @ T_inv
        np.testing.assert_array_almost_equal(identity, np.eye(4), decimal=10)


class TestRotZ:
    @pytest.mark.parametrize(
        ("deg", "expected"),
        [
            (0.0, np.eye(3)),
            (90.0, np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)),
            (180.0, np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)),
        ],
        ids=["zero", "90", "180"],
    )
    def test_known_angles(self, deg, expected):
        np.testing.assert_array_almost_equal(_rot_z(deg), expected, decimal=10)


class TestPixelDepthToCameraXyz:
    def test_principal_point_depth_one(self):
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        result = pixel_and_depth_to_camera_xyz((320.0, 240.0), 1.0, K)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 1000.0])

    def test_offset_pixel(self):
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        result = pixel_and_depth_to_camera_xyz((420.0, 240.0), 1.0, K)
        expected_x = (420.0 - 320.0) * 1000.0 / 500.0
        assert abs(result[0] - expected_x) < 1e-6
