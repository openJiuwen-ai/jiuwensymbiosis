# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters._common.ros2_cmd_vel.

These tests run without rclpy/geometry_msgs installed: the ``start()``
degradation path is exercised by hiding ``rclpy`` in ``sys.modules``, and the
``publish_twist()`` no-op-before-start path is covered directly. Mirrors
``test_ros2_scan.py`` / ``test_ros2_odom.py``.
"""

from __future__ import annotations

import sys

from jiuwensymbiosis.adapters._common.ros2_cmd_vel import Ros2CmdVel


class TestRos2CmdVelConstructAndDegrade:
    def test_construction_never_raises(self):
        # Even if rclpy is absent, __init__ must succeed (Ros2Odom parity).
        pub = Ros2CmdVel(cmd_vel_topic="/cmd_vel")
        assert pub.is_running is False
        # publish_twist before start() is a no-op returning False (no raise).
        assert pub.publish_twist(1.0, 0.0, 0.5) is False

    def test_unknown_msg_kind_falls_back_without_raising(self):
        # "Construction never raises" — an unknown kind degrades to "twist".
        pub = Ros2CmdVel(cmd_vel_topic="/cmd_vel", msg_kind="not-a-real-kind")
        assert pub.is_running is False
        assert pub.publish_twist(1.0, 0.0, 0.0) is False

    def test_start_returns_false_when_rclpy_missing(self, monkeypatch):
        # Simulate rclpy not importable: hide the module + the geometry_msgs
        # import path inside Ros2CmdVel.start().
        for mod in (
            "rclpy",
            "rclpy.node",
            "rclpy.executors",
            "geometry_msgs",
            "geometry_msgs.msg",
        ):
            monkeypatch.setitem(sys.modules, mod, None)

        pub = Ros2CmdVel(cmd_vel_topic="/cmd_vel")
        assert pub.start() is False
        assert pub.is_running is False
        assert pub.publish_twist(1.0, 0.0, 0.0) is False

    def test_start_idempotent_after_failure(self, monkeypatch):
        # Calling start() twice after a (simulated) failure must not raise and
        # must leave the publisher not running. Hide rclpy so start() fails.
        for mod in (
            "rclpy",
            "rclpy.node",
            "rclpy.executors",
            "geometry_msgs",
            "geometry_msgs.msg",
        ):
            monkeypatch.setitem(sys.modules, mod, None)
        pub = Ros2CmdVel(cmd_vel_topic="/cmd_vel")
        assert pub.start() is False
        pub.start()  # idempotent no-op
        assert pub.is_running is False

    def test_stop_is_safe_before_start(self):
        # stop() before start() must be a no-op, no raise.
        pub = Ros2CmdVel(cmd_vel_topic="/cmd_vel")
        pub.stop()
        assert pub.is_running is False

    def test_publish_twist_returns_false_before_start(self):
        # The nav loop relies on publish_twist returning False (not raising)
        # when the publisher isn't running, so it can treat "no publisher" the
        # same as "no motion" without a try/except.
        pub = Ros2CmdVel(cmd_vel_topic="/cmd_vel")
        assert pub.publish_twist(0.5, 0.1, 0.2) is False
        assert pub.publish_twist(0.0, 0.0, 0.0) is False
