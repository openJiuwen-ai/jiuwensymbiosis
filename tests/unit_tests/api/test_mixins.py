# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.api.mixins."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.api.mixins import (
    JointMotionMixin,
    MotionMixin,
    ParallelGripperMixin,
    SuctionMixin,
    VisionMixin,
)


class TestMixinCapabilities:
    def test_motion_mixin(self):
        assert MotionMixin.capability == "motion.cartesian"
        for name in ("home", "get_pose", "get_home_pose", "goto_xyzr", "move_direction"):
            method = getattr(MotionMixin, name, None)
            assert method is not None
            assert hasattr(method, "__robot_tool__")

    def test_joint_motion_mixin(self):
        assert JointMotionMixin.capability == "motion.joint"
        assert hasattr(JointMotionMixin.move_joint, "__robot_tool__")

    def test_suction_mixin(self):
        assert SuctionMixin.capability == "grasp.suction"
        for name in ("activate_suction", "deactivate_suction"):
            assert hasattr(getattr(SuctionMixin, name), "__robot_tool__")

    def test_parallel_gripper_mixin(self):
        assert ParallelGripperMixin.capability == "grasp.parallel"
        for name in ("open_gripper", "close_gripper"):
            assert hasattr(getattr(ParallelGripperMixin, name), "__robot_tool__")

    def test_vision_mixin(self):
        assert VisionMixin.capability == "vision.detection"
        for name in ("get_grasp_info_simple", "pixel_to_base_xyz", "get_image", "analyze_scene"):
            assert hasattr(getattr(VisionMixin, name), "__robot_tool__")

    def test_mixin_tool_meta_propagates(self):
        meta = MotionMixin.home.__robot_tool__
        assert meta.name == "home"
        assert "motion" in meta.tags


class _FakeMotionEnv:
    """Minimal env exposing just what MotionMixin.move_direction needs."""

    z_min_safe = 20.0
    workspace_bounds = (-300.0, -300.0, 300.0, 300.0)

    def __init__(self):
        self._pose = SimpleNamespace(x=100.0, y=0.0, z=200.0, rx=180.0, ry=0.0, rz=0.0)
        self.moved_to = None

    def get_flange_pose(self):
        return self._pose

    def move_to_flange(self, pose):
        self.moved_to = pose
        self._pose = pose


class _GenericArm(MotionMixin, BaseRobotApi):
    """Any robot that composes MotionMixin — NOT SO101-specific."""


class TestMoveDirectionIsGeneric:
    """move_direction lives on MotionMixin, so it works for any motion.cartesian robot."""

    def test_left_moves_plus_y_on_a_generic_arm(self):
        env = _FakeMotionEnv()
        api = _GenericArm(env)
        res = api.move_direction("left", 20)
        assert res["ok"] is True
        assert env.moved_to.y == 20.0  # left = +y
        assert env.moved_to.x == 100.0
        assert env.moved_to.z == 200.0

    def test_out_of_bounds_raises(self):
        api = _GenericArm(_FakeMotionEnv())
        with pytest.raises(ValueError, match="out of"):
            api.move_direction("right", 10_000)
