# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters.so101.geometry.

Covers: unit round-trip (mm<->m, deg), Euler zero rotation, intrinsic-XYZ Euler
vs scipy, multi-axis Euler != rotvec (contract check), cross-+-180deg,
non-finite rejection, matrix shape rejection, and the two error metrics.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jiuwensymbiosis.adapters.so101.geometry import (
    So101Pose,
    matrix_m_to_pose_mm_deg,
    orientation_error_deg,
    pose_mm_deg_to_matrix_m,
    position_error_mm,
)


class TestUnitRoundTrip:
    def test_identity_pose_round_trip(self):
        p = So101Pose(0, 0, 0, 0, 0, 0)
        m = pose_mm_deg_to_matrix_m(p)
        np.testing.assert_array_almost_equal(m, np.eye(4), decimal=10)
        back = matrix_m_to_pose_mm_deg(m)
        assert (back.x, back.y, back.z) == pytest.approx((p.x, p.y, p.z), abs=1e-6)
        assert (back.rx, back.ry, back.rz) == pytest.approx((p.rx, p.ry, p.rz), abs=1e-6)

    def test_translation_mm_to_m(self):
        # 1000 mm should become 1.0 m in the matrix.
        p = So101Pose(1000, 2000, 3000, 0, 0, 0)
        m = pose_mm_deg_to_matrix_m(p)
        np.testing.assert_array_almost_equal(m[:3, 3], [1.0, 2.0, 3.0], decimal=10)
        np.testing.assert_array_almost_equal(m[:3, :3], np.eye(3), decimal=10)

    def test_translation_m_to_mm(self):
        m = np.eye(4)
        m[:3, 3] = [0.5, -0.25, 1.25]  # metres
        p = matrix_m_to_pose_mm_deg(m)
        assert p.x == pytest.approx(500.0, abs=1e-6)
        assert p.y == pytest.approx(-250.0, abs=1e-6)
        assert p.z == pytest.approx(1250.0, abs=1e-6)
        assert (p.rx, p.ry, p.rz) == pytest.approx((0, 0, 0), abs=1e-6)

    def test_full_pose_round_trip(self):
        p = So101Pose(123.4, -56.7, 789.0, 10.0, -20.0, 30.0)
        m = pose_mm_deg_to_matrix_m(p)
        back = matrix_m_to_pose_mm_deg(m)
        assert back.x == pytest.approx(p.x, abs=1e-6)
        assert back.y == pytest.approx(p.y, abs=1e-6)
        assert back.z == pytest.approx(p.z, abs=1e-6)
        assert back.rx == pytest.approx(p.rx, abs=1e-6)
        assert back.ry == pytest.approx(p.ry, abs=1e-6)
        assert back.rz == pytest.approx(p.rz, abs=1e-6)


class TestEulerRotation:
    def test_zero_rotation_is_identity(self):
        p = So101Pose(0, 0, 0, 0, 0, 0)
        m = pose_mm_deg_to_matrix_m(p)
        np.testing.assert_array_almost_equal(m[:3, :3], np.eye(3), decimal=10)

    def test_euler_matches_scipy_intrinsic_xyz(self):
        euler_deg = np.array([15.0, -30.0, 45.0])
        p = So101Pose(0, 0, 0, *euler_deg.tolist())
        m = pose_mm_deg_to_matrix_m(p)
        expected = Rotation.from_euler("xyz", euler_deg, degrees=True).as_matrix()
        np.testing.assert_array_almost_equal(m[:3, :3], expected, decimal=10)

    def test_multi_axis_euler_is_not_rotvec(self):
        """Contract check: rx/ry/rz are XYZ Euler, NOT a rotation vector.

        A multi-axis (30, 45, 60)deg triple must match scipy's intrinsic-XYZ
        Euler and must NOT match the rotation-vector of the same numbers —
        the two differ substantially for multi-axis inputs.
        """
        euler_deg = np.array([30.0, 45.0, 60.0])
        p = So101Pose(0, 0, 0, *euler_deg.tolist())
        m_euler = pose_mm_deg_to_matrix_m(p)[:3, :3]
        m_rotvec = Rotation.from_rotvec(euler_deg, degrees=True).as_matrix()
        # Euler and rotvec disagree for multi-axis inputs.
        assert not np.allclose(m_euler, m_rotvec, atol=1e-3)
        # ... and the Euler version matches scipy's intrinsic-XYZ.
        expected = Rotation.from_euler("xyz", euler_deg, degrees=True).as_matrix()
        np.testing.assert_array_almost_equal(m_euler, expected, decimal=10)

    def test_cross_plus_minus_180_single_axis(self):
        # A 180deg rotation about +Z and about -Z yield the same orientation;
        # the round trip through the matrix must recover an equivalent Euler
        # triple (scipy normalises each Euler angle to [-180, 180]).
        p_plus = So101Pose(0, 0, 0, 0, 0, 180.0)
        p_minus = So101Pose(0, 0, 0, 0, 0, -180.0)
        m_plus = pose_mm_deg_to_matrix_m(p_plus)
        m_minus = pose_mm_deg_to_matrix_m(p_minus)
        np.testing.assert_array_almost_equal(m_plus[:3, :3], m_minus[:3, :3], decimal=6)
        # Either Euler representation is valid; the Z angle magnitude is 180.
        back = matrix_m_to_pose_mm_deg(m_plus)
        assert abs(back.rz) == pytest.approx(180.0, abs=1e-6)

    def test_single_axis_175_round_trips(self):
        # A pure 175deg about X must round-trip to within tolerance.
        p = So101Pose(10.0, 20.0, 30.0, 175.0, 0, 0)
        back = matrix_m_to_pose_mm_deg(pose_mm_deg_to_matrix_m(p))
        assert (back.x, back.y, back.z) == pytest.approx((p.x, p.y, p.z), abs=1e-6)
        assert (back.rx, back.ry, back.rz) == pytest.approx((p.rx, p.ry, p.rz), abs=1e-6)


class TestNonFiniteAndShape:
    @pytest.mark.parametrize(
        "bad_field,bad_val",
        [
            ("x", float("nan")),
            ("y", float("inf")),
            ("z", float("-inf")),
            ("rx", float("nan")),
            ("ry", float("inf")),
            ("rz", float("-inf")),
        ],
    )
    def test_non_finite_pose_rejected(self, bad_field, bad_val):
        kwargs = {"x": 1, "y": 2, "z": 3, "rx": 0, "ry": 0, "rz": 0}
        kwargs[bad_field] = bad_val
        p = So101Pose(**kwargs)
        with pytest.raises(ValueError):
            pose_mm_deg_to_matrix_m(p)

    def test_non_finite_matrix_rejected(self):
        m = np.eye(4)
        m[0, 3] = float("nan")
        with pytest.raises(ValueError):
            matrix_m_to_pose_mm_deg(m)

    def test_wrong_shape_matrix_rejected(self):
        with pytest.raises(ValueError):
            matrix_m_to_pose_mm_deg(np.zeros((3, 3)))
        with pytest.raises(ValueError):
            matrix_m_to_pose_mm_deg(np.zeros((4, 4, 2)))

    def test_non_pose_type_rejected(self):
        with pytest.raises(TypeError):
            pose_mm_deg_to_matrix_m((1, 2, 3, 0, 0, 0))

    def test_matrix_output_shape_and_dtype(self):
        p = So101Pose(1, 2, 3, 0, 0, 0)
        m = pose_mm_deg_to_matrix_m(p)
        assert m.shape == (4, 4)
        assert m.dtype == np.float64


class TestPositionError:
    def test_zero_when_identical(self):
        p = So101Pose(100, 200, 300, 5, 6, 7)
        assert position_error_mm(p, p) == pytest.approx(0.0, abs=1e-9)

    def test_pure_translation_distance(self):
        a = So101Pose(0, 0, 0, 0, 0, 0)
        b = So101Pose(300, 400, 0, 0, 0, 0)  # 3-4-5 triangle in XY plane
        assert position_error_mm(a, b) == pytest.approx(500.0, abs=1e-6)

    def test_ignores_rotation(self):
        a = So101Pose(0, 0, 0, 0, 0, 0)
        b = So101Pose(0, 0, 0, 0, 0, 90)
        assert position_error_mm(a, b) == pytest.approx(0.0, abs=1e-9)

    def test_non_finite_rejected(self):
        a = So101Pose(float("nan"), 0, 0, 0, 0, 0)
        b = So101Pose(0, 0, 0, 0, 0, 0)
        with pytest.raises(ValueError):
            position_error_mm(a, b)


class TestOrientationError:
    def test_zero_when_identical(self):
        p = So101Pose(100, 200, 300, 5, 6, 7)
        assert orientation_error_deg(p, p) == pytest.approx(0.0, abs=1e-9)

    def test_known_90deg_about_z(self):
        a = So101Pose(0, 0, 0, 0, 0, 0)
        b = So101Pose(0, 0, 0, 0, 0, 90)
        assert orientation_error_deg(a, b) == pytest.approx(90.0, abs=1e-6)

    def test_ignores_translation(self):
        a = So101Pose(0, 0, 0, 0, 0, 0)
        b = So101Pose(999, -999, 999, 0, 0, 0)
        assert orientation_error_deg(a, b) == pytest.approx(0.0, abs=1e-9)

    def test_symmetric(self):
        a = So101Pose(0, 0, 0, 10, -20, 30)
        b = So101Pose(0, 0, 0, -5, 15, 40)
        assert orientation_error_deg(a, b) == pytest.approx(orientation_error_deg(b, a), abs=1e-9)

    def test_clamped_at_180(self):
        # Antipodal rotations give the maximum geodesic distance, 180deg.
        a = So101Pose(0, 0, 0, 0, 0, 0)
        b = So101Pose(0, 0, 0, 180.0, 0, 0)
        assert orientation_error_deg(a, b) == pytest.approx(180.0, abs=1e-6)

    def test_non_finite_rejected(self):
        a = So101Pose(0, 0, 0, float("inf"), 0, 0)
        b = So101Pose(0, 0, 0, 0, 0, 0)
        with pytest.raises(ValueError):
            orientation_error_deg(a, b)
