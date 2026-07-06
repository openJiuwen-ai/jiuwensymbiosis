# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters._common.ros2_odom.

These tests run without rclpy/nav_msgs installed: the pose extractors are
pure-python and accept any object with the right attributes, and the
``start()`` degradation path is exercised by hiding ``rclpy`` in ``sys.modules``.
Mirrors ``test_ros2_camera.py``.
"""

from __future__ import annotations

import math
import sys
import types

import numpy as np
import pytest

from jiuwensymbiosis.adapters._common.ros2_odom import (
    Ros2Odom,
    _extract_odometry,
    _extract_pose_stamped,
    _extract_pose_with_covariance_stamped,
    _quat_to_yaw_deg,
)


def _pose(x=0.0, y=0.0, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
    """A geometry_msgs/Pose-shaped object (no rclpy needed)."""
    return types.SimpleNamespace(
        position=types.SimpleNamespace(x=x, y=y, z=z),
        orientation=types.SimpleNamespace(x=qx, y=qy, z=qz, w=qw),
    )


def _odom_msg(*pose_args, **pose_kwargs):
    """A nav_msgs/Odometry-shaped object: pose under .pose.pose."""
    return types.SimpleNamespace(pose=types.SimpleNamespace(pose=_pose(*pose_args, **pose_kwargs)))


def _pose_stamped_msg(*pose_args, **pose_kwargs):
    """A geometry_msgs/PoseStamped-shaped object: pose under .pose."""
    return types.SimpleNamespace(pose=_pose(*pose_args, **pose_kwargs))


def _pwc_msg(*pose_args, **pose_kwargs):
    """A geometry_msgs/PoseWithCovarianceStamped-shaped object: pose under .pose.pose."""
    return types.SimpleNamespace(pose=types.SimpleNamespace(pose=_pose(*pose_args, **pose_kwargs)))


class TestExtractors:
    def test_odometry_extracts_xyz_quat(self):
        xyz, quat = _extract_odometry(_odom_msg(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0))
        assert np.allclose(xyz, [1.0, 2.0, 3.0])
        assert np.allclose(quat, [0.0, 0.0, 0.0, 1.0])

    def test_pose_stamped_extracts_xyz_quat(self):
        xyz, quat = _extract_pose_stamped(_pose_stamped_msg(4.0, 5.0, 6.0, 0.0, 0.0, 1.0, 0.0))
        assert np.allclose(xyz, [4.0, 5.0, 6.0])
        assert np.allclose(quat, [0.0, 0.0, 1.0, 0.0])

    def test_pose_with_covariance_stamped_extracts_xyz_quat(self):
        xyz, quat = _extract_pose_with_covariance_stamped(_pwc_msg(7.0, 8.0, 9.0))
        assert np.allclose(xyz, [7.0, 8.0, 9.0])
        assert np.allclose(quat, [0.0, 0.0, 0.0, 1.0])

    def test_odometry_missing_pose_returns_none(self):
        # No .pose attribute → None, no raise.
        assert _extract_odometry(types.SimpleNamespace()) is None

    def test_pose_stamped_missing_pose_returns_none(self):
        assert _extract_pose_stamped(types.SimpleNamespace()) is None

    def test_missing_position_returns_none(self):
        # .pose exists but has no .position → None, no raise.
        msg = types.SimpleNamespace(pose=types.SimpleNamespace(orientation=_pose().orientation))
        assert _extract_pose_stamped(msg) is None

    def test_missing_orientation_returns_none(self):
        msg = types.SimpleNamespace(pose=types.SimpleNamespace(position=_pose().position))
        assert _extract_pose_stamped(msg) is None

    def test_non_numeric_position_returns_none(self):
        # Non-numeric position fields must not raise; degrade to None.
        pose = types.SimpleNamespace(
            position=types.SimpleNamespace(x="x", y=0.0, z=0.0),
            orientation=_pose().orientation,
        )
        assert _extract_pose_stamped(types.SimpleNamespace(pose=pose)) is None


class TestQuatToYaw:
    def test_identity_quaternion_is_zero_yaw(self):
        assert _quat_to_yaw_deg(0.0, 0.0, 0.0, 1.0) == pytest.approx(0.0)

    def test_90deg_yaw_about_z(self):
        # Quaternion for +90° rotation about Z: (x,y,z,w) = (0,0,sin45,cos45).
        z = math.sin(math.radians(45))
        w = math.cos(math.radians(45))
        assert _quat_to_yaw_deg(0.0, 0.0, z, w) == pytest.approx(90.0, abs=1e-6)

    def test_negative_90deg_yaw(self):
        z = math.sin(math.radians(-45))
        w = math.cos(math.radians(-45))
        assert _quat_to_yaw_deg(0.0, 0.0, z, w) == pytest.approx(-90.0, abs=1e-6)

    def test_180deg_yaw(self):
        # Pure 180° about Z: (0,0,1,0).
        assert _quat_to_yaw_deg(0.0, 0.0, 1.0, 0.0) == pytest.approx(180.0, abs=1e-6)


class TestRos2OdomConstructAndDegrade:
    def test_construction_never_raises(self):
        # Even if rclpy is absent, __init__ must succeed (Ros2Camera parity).
        odom = Ros2Odom(odom_topic="/odom")
        assert odom.is_running is False
        assert odom.grab_pose() is None

    def test_unknown_msg_kind_falls_back_without_raising(self):
        # "Construction never raises" — an unknown kind degrades to "odometry".
        odom = Ros2Odom(odom_topic="/odom", msg_kind="not-a-real-kind")
        assert odom.is_running is False
        assert odom.grab_pose() is None

    def test_start_returns_false_when_rclpy_missing(self, monkeypatch):
        # Simulate rclpy not importable: hide the module + the nav_msgs import
        # path inside Ros2Odom.start().
        for mod in (
            "rclpy",
            "rclpy.node",
            "rclpy.executors",
            "nav_msgs",
            "nav_msgs.msg",
            "geometry_msgs",
            "geometry_msgs.msg",
        ):
            monkeypatch.setitem(sys.modules, mod, None)

        odom = Ros2Odom(odom_topic="/odom")
        assert odom.start() is False
        assert odom.is_running is False
        assert odom.grab_pose() is None

    def test_grab_pose_none_until_first_message(self):
        odom = Ros2Odom(odom_topic="/odom")
        assert odom.grab_pose() is None
        # Simulate a callback arriving.
        odom._on_odom(_odom_msg(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0))
        pose = odom.grab_pose()
        assert pose is not None
        assert pose["x"] == 1.0
        assert pose["y"] == 2.0
        assert pose["z"] == 3.0
        assert pose["qx"] == 0.0
        assert pose["qy"] == 0.0
        assert pose["qz"] == 0.0
        assert pose["qw"] == 1.0
        assert pose["yaw_deg"] == pytest.approx(0.0)

    def test_callback_ignores_malformed_message(self):
        # A malformed message (missing pose) must be dropped silently — the
        # latest pose stays None. Provide a valid first message so the
        # assertion isn't vacuous (proves the second msg was genuinely dropped).
        odom = Ros2Odom(odom_topic="/odom")
        odom._on_odom(_odom_msg(1.0, 0.0, 0.0))  # valid
        assert odom.grab_pose() is not None
        odom._on_odom(types.SimpleNamespace())  # malformed — must not crash / overwrite
        pose = odom.grab_pose()
        assert pose is not None
        # The valid pose is retained (malformed message did not reset it).
        assert pose["x"] == 1.0

    def test_yaw_deg_in_grab_pose_matches_quaternion(self):
        odom = Ros2Odom(odom_topic="/odom")
        z = math.sin(math.radians(45))
        w = math.cos(math.radians(45))
        odom._on_odom(_odom_msg(0.0, 0.0, 0.0, 0.0, 0.0, z, w))
        pose = odom.grab_pose()
        assert pose is not None
        assert pose["yaw_deg"] == pytest.approx(90.0, abs=1e-6)

    def test_pose_stamped_msg_kind_routes_to_pose_extractor(self):
        # With msg_kind="pose_stamped", a PoseStamped-shaped message is read
        # (the pose is under .pose, not .pose.pose — wrong nesting would yield None).
        odom = Ros2Odom(odom_topic="/pose", msg_kind="pose_stamped")
        odom._on_odom(_pose_stamped_msg(5.0, 6.0, 7.0))
        pose = odom.grab_pose()
        assert pose is not None
        assert pose["x"] == 5.0
        assert pose["y"] == 6.0
        assert pose["z"] == 7.0
