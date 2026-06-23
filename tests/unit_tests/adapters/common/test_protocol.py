# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters._common.protocol — runtime_checkable."""

from __future__ import annotations

import inspect

from jiuwensymbiosis.adapters._common.protocol import (
    RobotDriver,
    JointDriver,
    CameraDriver,
    SuctionDriver,
    GripperDriver,
    VisionDriver,
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


class _MinimalGripperDriver:
    def set_gripper(self, on):
        pass

    @property
    def gripper_state(self):
        return False


class _MinimalVisionDriver:
    @property
    def tf_flange_cam(self):
        return None

    @property
    def calibration(self):
        return None


class TestMoveToPoseBlockingSignature:
    """The Protocol used to type move_to_pose_blocking as ``*args, **kwargs``,
    hiding a forgotten ``pose`` argument until runtime. The first positional
    parameter must be the structured ``pose`` object."""

    def test_first_param_is_pose(self):
        sig = inspect.signature(RobotDriver.move_to_pose_blocking)
        params = list(sig.parameters)
        assert params[0] == "self"
        assert params[1] == "pose", (
            "move_to_pose_blocking must declare `pose` as its first non-self "
            "parameter so a missing pose is a static error, not a runtime crash"
        )

    def test_vendor_kwargs_still_accepted(self):
        # Vendor extensions (sync_timeout_s, joint=...) ride in *args/**kwargs
        # after pose — changing the signature must not break Piper's override.
        sig = inspect.signature(RobotDriver.move_to_pose_blocking)
        params = sig.parameters
        assert any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params.values())
        assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


class TestProtocols:
    def test_robot_driver_protocol(self):
        assert isinstance(_MinimalRobotDriver(), RobotDriver)

    def test_joint_driver_protocol(self):
        assert isinstance(_MinimalJointDriver(), JointDriver)

    def test_camera_driver_protocol(self):
        assert isinstance(_MinimalCameraDriver(), CameraDriver)

    def test_suction_driver_protocol(self):
        assert isinstance(_MinimalSuctionDriver(), SuctionDriver)

    def test_gripper_driver_protocol(self):
        assert isinstance(_MinimalGripperDriver(), GripperDriver)

    def test_vision_driver_protocol(self):
        assert isinstance(_MinimalVisionDriver(), VisionDriver)

    def test_gripper_and_suction_are_distinct(self):
        # A suction-only driver must NOT structurally satisfy GripperDriver.
        assert not isinstance(_MinimalSuctionDriver(), GripperDriver)
        assert not isinstance(_MinimalGripperDriver(), SuctionDriver)

    def test_mock_piper_driver_satisfies_robot_driver(self):
        assert isinstance(MockPiperDriver(), RobotDriver)


class _FullMockDriver(
    _MinimalRobotDriver,
    _MinimalJointDriver,
    _MinimalCameraDriver,
    _MinimalGripperDriver,
    _MinimalVisionDriver,
):
    """A mock that satisfies all five vendor protocols simultaneously."""

    pass


class TestPiperFullDriver:
    """Tests for the composite PiperFullDriver Protocol."""

    def test_full_mock_satisfies_composite(self):
        from jiuwensymbiosis.adapters._common.protocol import PiperFullDriver

        assert isinstance(_FullMockDriver(), PiperFullDriver)

    def test_robot_only_does_not_satisfy_composite(self):
        from jiuwensymbiosis.adapters._common.protocol import PiperFullDriver

        # A driver implementing only RobotDriver must NOT satisfy the composite.
        assert not isinstance(_MinimalRobotDriver(), PiperFullDriver)
