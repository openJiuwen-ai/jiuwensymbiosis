# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters._common.kinematics."""

from __future__ import annotations

import pytest

from jiuwensymbiosis.adapters._common.geometry import CartesianPose, pose_to_matrix
from jiuwensymbiosis.adapters._common.kinematics import (
    MotionLimits,
    interpolate_joint_waypoints,
    interpolate_se3,
    plan_cartesian_move,
    plan_joint_move,
)

from ._kinematic_fakes import GantryBackend

_NAMES = ("jx", "jy", "jz")


def _limits(**over) -> MotionLimits:
    # Gantry joints ARE mm, so a 50mm cartesian step is a 50-unit joint step;
    # keep the branch-flip guard above that (real arms use deg vs mm).
    base = {
        "joint_names": _NAMES,
        "joint_limits": ((-500.0, 500.0), (-500.0, 500.0), (0.0, 500.0)),
        "max_joint_step_deg": 50.0,
        "max_cartesian_step_mm": 50.0,
        "max_ik_jump_deg": 1000.0,
    }
    base.update(over)
    return MotionLimits(**base)


class TestMotionLimits:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            MotionLimits(joint_names=_NAMES, joint_limits=((-1.0, 1.0),))


class TestJointInterpolation:
    def test_step_count_and_endpoint(self):
        wps = interpolate_joint_waypoints([0.0, 0.0, 0.0], [0.0, 0.0, 120.0], max_step_deg=50.0)
        assert len(wps) == 3  # ceil(120/50)
        assert wps[-1] == pytest.approx([0.0, 0.0, 120.0])

    def test_zero_move_single_waypoint(self):
        wps = interpolate_joint_waypoints([5.0, 5.0, 5.0], [5.0, 5.0, 5.0], max_step_deg=50.0)
        assert len(wps) == 1


class TestPlanJointMove:
    def test_valid_target(self):
        wps = plan_joint_move([0.0, 0.0, 0.0], [10.0, 20.0, 30.0], _limits())
        assert wps[-1] == pytest.approx([10.0, 20.0, 30.0])

    def test_non_finite_rejected(self):
        with pytest.raises(ValueError):
            plan_joint_move([0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0], _limits())

    def test_out_of_limit_rejected(self):
        with pytest.raises(ValueError):
            plan_joint_move([0.0, 0.0, 0.0], [0.0, 0.0, 999.0], _limits())

    def test_wrong_length_rejected(self):
        with pytest.raises(ValueError):
            plan_joint_move([0.0, 0.0], [0.0, 0.0], _limits())


class TestSe3Interpolation:
    def test_endpoint_and_count(self):
        start = pose_to_matrix(CartesianPose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        end = pose_to_matrix(CartesianPose(100.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        path = interpolate_se3(start, end, 4)
        assert len(path) == 4
        assert path[-1] == pytest.approx(end)


class TestPlanCartesianMove:
    def test_reachable_zero_residual(self):
        backend = GantryBackend()
        target = pose_to_matrix(CartesianPose(100.0, 0.0, 300.0, 0.0, 0.0, 0.0))
        plan = plan_cartesian_move(backend, [0.0, 0.0, 100.0], target, _limits())
        assert plan[-1].q == pytest.approx([100.0, 0.0, 300.0])
        assert plan[-1].position_error_mm == pytest.approx(0.0, abs=1e-6)

    def test_unreachable_rejected(self):
        backend = GantryBackend(max_reach_mm=200.0)
        target = pose_to_matrix(CartesianPose(500.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        with pytest.raises(ValueError, match="unreachable"):
            plan_cartesian_move(backend, [0.0, 0.0, 100.0], target, _limits())

    def test_orientation_recorded_not_rejected_when_no_tolerance(self):
        backend = GantryBackend()
        target = pose_to_matrix(CartesianPose(50.0, 0.0, 100.0, 90.0, 0.0, 0.0))
        plan = plan_cartesian_move(backend, [0.0, 0.0, 100.0], target, _limits())
        assert plan[-1].orientation_error_deg == pytest.approx(90.0, abs=1e-6)

    def test_orientation_rejected_when_tolerance_set(self):
        backend = GantryBackend()
        target = pose_to_matrix(CartesianPose(50.0, 0.0, 100.0, 90.0, 0.0, 0.0))
        with pytest.raises(ValueError, match="orientation"):
            plan_cartesian_move(backend, [0.0, 0.0, 100.0], target, _limits(ik_orientation_tolerance_deg=10.0))

    def test_ik_branch_jump_rejected(self):
        backend = GantryBackend()
        target = pose_to_matrix(CartesianPose(100.0, 0.0, 100.0, 0.0, 0.0, 0.0))
        with pytest.raises(ValueError, match="jump"):
            plan_cartesian_move(backend, [0.0, 0.0, 100.0], target, _limits(max_ik_jump_deg=1.0))
