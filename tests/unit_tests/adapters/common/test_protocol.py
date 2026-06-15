# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters._common.protocol — runtime_checkable."""

from __future__ import annotations

from jiuwensymbiosis.adapters._common.protocol import (
    RobotDriver,
    JointDriver,
    CameraDriver,
    SuctionDriver,
)

from tests.mocks.mock_driver import MockPiperDriver


class _MinimalRobotDriver:
    @property
    def home_pose(self):
        return None

    @property
    def z_min_safe(self):
        return 0.0

    @property
    def flange_z_min_safe(self):
        return 0.0

    @property
    def tool_offset_mm(self):
        return 0.0

    def close(self):
        pass

    def home(self):
        pass

    def get_pose(self):
        return None

    def move_to_pose_blocking(self, *a, **kw):
        pass


class _MinimalJointDriver:
    def get_angles(self):
        return None

    def move_joint_blocking(self, q, *, timeout_s=30.0):
        pass


class _MinimalCameraDriver:
    @property
    def intrinsics(self):
        return None

    def grab_frames(self):
        return None


class _MinimalSuctionDriver:
    @property
    def suction_state(self):
        return False

    @property
    def suction_di_last(self):
        return None

    def set_suction(self, on):
        pass


class TestProtocols:
    def test_robot_driver_protocol(self):
        assert isinstance(_MinimalRobotDriver(), RobotDriver)

    def test_joint_driver_protocol(self):
        assert isinstance(_MinimalJointDriver(), JointDriver)

    def test_camera_driver_protocol(self):
        assert isinstance(_MinimalCameraDriver(), CameraDriver)

    def test_suction_driver_protocol(self):
        assert isinstance(_MinimalSuctionDriver(), SuctionDriver)

    def test_mock_piper_driver_satisfies_robot_driver(self):
        assert isinstance(MockPiperDriver(), RobotDriver)
