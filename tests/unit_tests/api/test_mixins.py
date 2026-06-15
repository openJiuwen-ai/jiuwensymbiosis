# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.api.mixins."""

from __future__ import annotations

from jiuwensymbiosis.api.mixins import (
    MotionMixin,
    JointMotionMixin,
    SuctionMixin,
    ParallelGripperMixin,
    VisionMixin,
)


class TestMixinCapabilities:
    def test_motion_mixin(self):
        assert MotionMixin.capability == "motion.cartesian"
        for name in ("home", "get_pose", "get_home_pose", "goto_xyzr"):
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
