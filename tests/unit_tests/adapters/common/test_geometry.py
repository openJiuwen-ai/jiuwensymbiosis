# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters._common.geometry."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.adapters._common.geometry import (
    CartesianPose,
    matrix_to_pose,
    orientation_error_deg,
    pose_to_matrix,
    position_error_mm,
    read_pose_xyzrpy,
)


class TestReadPose:
    def test_reads_rx_ry_rz(self):
        obj = SimpleNamespace(x=1.0, y=2.0, z=3.0, rx=10.0, ry=20.0, rz=30.0)
        assert read_pose_xyzrpy(obj) == (1.0, 2.0, 3.0, 10.0, 20.0, 30.0)

    def test_falls_back_to_r_for_rz(self):
        obj = SimpleNamespace(x=1.0, y=2.0, z=3.0, r=45.0)
        assert read_pose_xyzrpy(obj) == (1.0, 2.0, 3.0, 0.0, 0.0, 45.0)


class TestPoseMatrixRoundtrip:
    def test_translation_is_metres(self):
        m = pose_to_matrix(CartesianPose(100.0, -200.0, 300.0, 0.0, 0.0, 0.0))
        assert m[:3, 3] == pytest.approx([0.1, -0.2, 0.3])

    def test_roundtrip_generic_pose(self):
        pose = CartesianPose(120.0, -30.0, 415.0, 10.0, 20.0, 30.0)
        back = matrix_to_pose(pose_to_matrix(pose))
        assert back.as_tuple() == pytest.approx(pose.as_tuple(), abs=1e-6)

    def test_roundtrip_zero_rotation(self):
        pose = CartesianPose(0.0, 0.0, 250.0, 0.0, 0.0, 0.0)
        back = matrix_to_pose(pose_to_matrix(pose))
        assert back.as_tuple() == pytest.approx(pose.as_tuple(), abs=1e-6)

    def test_roundtrip_near_180(self):
        pose = CartesianPose(50.0, 60.0, 70.0, 179.0, 0.0, 0.0)
        back = matrix_to_pose(pose_to_matrix(pose))
        assert pose_to_matrix(back) == pytest.approx(pose_to_matrix(pose), abs=1e-9)

    def test_bad_matrix_shape_raises(self):
        with pytest.raises(ValueError):
            matrix_to_pose(np.eye(3))


class TestErrorFunctions:
    def test_position_error_mm(self):
        a = pose_to_matrix(CartesianPose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        b = pose_to_matrix(CartesianPose(30.0, 40.0, 0.0, 0.0, 0.0, 0.0))
        assert position_error_mm(a, b) == pytest.approx(50.0)

    def test_orientation_error_deg(self):
        a = pose_to_matrix(CartesianPose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        b = pose_to_matrix(CartesianPose(0.0, 0.0, 0.0, 0.0, 0.0, 90.0))
        assert orientation_error_deg(a, b) == pytest.approx(90.0)

    def test_zero_errors(self):
        m = pose_to_matrix(CartesianPose(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        assert position_error_mm(m, m) == pytest.approx(0.0)
        assert orientation_error_deg(m, m) == pytest.approx(0.0, abs=1e-9)
        assert math.isfinite(orientation_error_deg(m, m))
