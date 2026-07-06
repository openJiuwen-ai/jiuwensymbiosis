# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters.unitree_go2.env."""

from __future__ import annotations

from jiuwensymbiosis.adapters.unitree_go2.config import UnitreeGo2Config


class _MockLowLevel:
    """Mock UnitreeGo2Driver for hardware-free env tests."""

    home_pose = type("Pose", (), {"x": 0.0, "y": 0.0, "z": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0})()
    tool_offset_mm = 0.0
    _connected = False
    # Default odom pose (mirrors the real driver's self._odom-driven return);
    # tests override this attribute to simulate "no message yet".
    _odom_pose = {"x": 1.0, "y": 2.0, "z": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0, "yaw_deg": 30.0}

    def close(self):
        self._connected = False

    def get_pose(self):
        # Simulate an odom-derived base pose: x=1.0 m, y=2.0 m, yaw=30 deg.
        return type("P", (), {"x": 1.0, "y": 2.0, "z": 0.0, "rx": 0.0, "ry": 0.0, "rz": 30.0})()

    def move_to_pose_blocking(self, *a, **kw):
        pass

    def grab_frames(self):
        return None  # no camera frames in this mock

    def get_odom_pose(self):
        return self._odom_pose


class TestUnitreeGo2EnvConstruction:
    def test_default_config(self):
        from jiuwensymbiosis.adapters.unitree_go2.env import UnitreeGo2Env

        cfg = UnitreeGo2Config()
        env = UnitreeGo2Env(cfg)
        assert env.low_level is None
        assert "motion.cartesian" in env.capabilities
        assert "vision.camera" in env.capabilities
        assert "vision.depth" in env.capabilities

    def test_capabilities_exclude_arm_and_grasp(self):
        from jiuwensymbiosis.adapters.unitree_go2.env import UnitreeGo2Env

        env = UnitreeGo2Env(UnitreeGo2Config())
        # Mobile base form factor: no joint motion, no gripper/suction.
        assert "motion.joint" not in env.capabilities
        assert "grasp.parallel" not in env.capabilities
        assert "grasp.suction" not in env.capabilities


class TestUnitreeGo2EnvSafetyContract:
    def test_z_min_safe_fixed_zero_planar(self):
        from jiuwensymbiosis.adapters.unitree_go2.env import UnitreeGo2Env

        env = UnitreeGo2Env(UnitreeGo2Config())
        # Before connect: reads from config (0.0).
        assert env.z_min_safe == 0.0
        # With a live driver: still 0.0 (planar base, never triggers).
        env.low_level = _MockLowLevel()
        assert env.z_min_safe == 0.0

    def test_workspace_bounds_from_config(self):
        from jiuwensymbiosis.adapters.unitree_go2.env import UnitreeGo2Env

        cfg = UnitreeGo2Config(x_min_m=-5.0, y_min_m=-5.0, x_max_m=5.0, y_max_m=5.0)
        env = UnitreeGo2Env(cfg)
        assert env.workspace_bounds == (-5.0, -5.0, 5.0, 5.0)

    def test_workspace_bounds_none_when_unset(self):
        from jiuwensymbiosis.adapters.unitree_go2.env import UnitreeGo2Env

        cfg = UnitreeGo2Config(x_min_m=None)
        env = UnitreeGo2Env(cfg)
        assert env.workspace_bounds is None

    def test_low_level_property_round_trip(self):
        from jiuwensymbiosis.adapters.unitree_go2.env import UnitreeGo2Env

        env = UnitreeGo2Env(UnitreeGo2Config())
        assert env.low_level is None
        driver = _MockLowLevel()
        env.low_level = driver
        assert env.low_level is driver

    def test_tool_offset_mm_fixed_zero(self):
        from jiuwensymbiosis.adapters.unitree_go2.env import UnitreeGo2Env

        env = UnitreeGo2Env(UnitreeGo2Config())
        env.low_level = _MockLowLevel()
        assert env.tool_offset_mm == 0.0  # mobile base: no flange→tip offset


class TestUnitreeGo2EnvObservation:
    def test_get_observation_returns_empty_before_connect(self):
        from jiuwensymbiosis.adapters.unitree_go2.env import UnitreeGo2Env

        env = UnitreeGo2Env(UnitreeGo2Config())
        obs = env.get_observation()
        assert obs.pose is None
        assert obs.rgb is None
        assert obs.depth is None
        assert obs.extra == {}  # no low_level → empty extra

    def test_get_observation_surfaces_pose_and_odom(self):
        from jiuwensymbiosis.adapters.unitree_go2.env import UnitreeGo2Env

        cfg = UnitreeGo2Config(ros2_odom_topic="/odom")
        env = UnitreeGo2Env(cfg)
        env.low_level = _MockLowLevel()
        obs = env.get_observation()
        assert obs.pose == {"x": 1.0, "y": 2.0, "z": 0.0, "rx": 0.0, "ry": 0.0, "rz": 30.0}
        # odom surfaced into extra (raw ROS units + yaw_deg).
        assert obs.extra["odom"] == {
            "x": 1.0,
            "y": 2.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
            "yaw_deg": 30.0,
        }
        assert obs.extra["z_min_safe"] == 0.0

    def test_get_observation_odom_none_when_no_odom_topic(self):
        from jiuwensymbiosis.adapters.unitree_go2.env import UnitreeGo2Env

        # No ros2_odom_topic configured → extra["odom"] is None even with a driver.
        env = UnitreeGo2Env(UnitreeGo2Config())
        env.low_level = _MockLowLevel()
        obs = env.get_observation()
        assert obs.extra["odom"] is None

    def test_get_observation_odom_none_when_driver_returns_none(self):
        from jiuwensymbiosis.adapters.unitree_go2.env import UnitreeGo2Env

        cfg = UnitreeGo2Config(ros2_odom_topic="/odom")
        env = UnitreeGo2Env(cfg)
        env.low_level = _MockLowLevel()
        # Force the driver to report no odom message yet.
        env.low_level._odom_pose = None
        obs = env.get_observation()
        assert obs.extra["odom"] is None
        # Pose still falls back to the home pose (nominal origin).
        assert obs.pose is not None

    def test_home_pose_prop(self):
        from jiuwensymbiosis.adapters.unitree_go2.env import UnitreeGo2Env

        env = UnitreeGo2Env(UnitreeGo2Config())
        env.low_level = _MockLowLevel()
        hp = env.home_pose
        assert hp is not None
        assert hp.x == 0.0
