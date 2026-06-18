# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters.piper.env."""

from __future__ import annotations

import pytest

from jiuwensymbiosis.adapters.piper.config import PiperConfig


class _MockLowLevel:
    home_pose = type("Pose", (), {"x": 200, "y": 0, "z": 400, "rx": 0, "ry": 90, "rz": 0})()
    z_min_safe = 50.0
    flange_z_min_safe = 185.8
    tool_offset_mm = 135.8
    _connected = False

    def close(self):
        self._connected = False

    def home(self):
        self._pose = type(
            "P",
            (),
            {
                "x": self.home_pose.x,
                "y": self.home_pose.y,
                "z": self.home_pose.z,
                "rx": self.home_pose.rx,
                "ry": self.home_pose.ry,
                "rz": self.home_pose.rz,
            },
        )()

    def get_pose(self):
        return type("P", (), {"x": 200, "y": 0, "z": 400, "rx": 0, "ry": 90, "rz": 0})()

    def move_to_pose_blocking(self, *a, **kw):
        pass


class TestPiperEnvConstruction:
    def test_default_config(self):
        from jiuwensymbiosis.adapters.piper.env import PiperEnv

        cfg = PiperConfig()
        env = PiperEnv(cfg)
        assert env._inner is None
        assert "motion.cartesian" in env.capabilities

    def test_capabilities_include_joint(self):
        from jiuwensymbiosis.adapters.piper.env import PiperEnv

        cfg = PiperConfig()
        env = PiperEnv(cfg)
        assert "motion.joint" in env.capabilities


class TestPiperEnvSafetyContract:
    """Formalized BaseRobotEnv contract: z_min_safe / workspace_bounds / low_level."""

    def test_z_min_safe_from_config_before_connect(self):
        from jiuwensymbiosis.adapters.piper.env import PiperEnv

        cfg = PiperConfig(z_min_safe_mm=42.0)
        env = PiperEnv(cfg)
        assert env.z_min_safe == 42.0  # driver is None → falls back to config

    def test_z_min_safe_prefers_live_driver(self):
        from jiuwensymbiosis.adapters.piper.env import PiperEnv

        cfg = PiperConfig(z_min_safe_mm=42.0)
        env = PiperEnv(cfg)
        env._inner = _MockLowLevel()  # driver reports 50.0
        assert env.z_min_safe == 50.0

    def test_workspace_bounds_from_config(self):
        from jiuwensymbiosis.adapters.piper.env import PiperEnv

        cfg = PiperConfig(x_min_mm=0.0, y_min_mm=-500.0, x_max_mm=700.0, y_max_mm=500.0)
        env = PiperEnv(cfg)
        assert env.workspace_bounds == (0.0, -500.0, 700.0, 500.0)

    def test_workspace_bounds_none_when_unset(self):
        from jiuwensymbiosis.adapters.piper.env import PiperEnv

        cfg = PiperConfig(x_min_mm=None)
        env = PiperEnv(cfg)
        assert env.workspace_bounds is None

    def test_low_level_property_tracks_inner(self):
        from jiuwensymbiosis.adapters.piper.env import PiperEnv

        env = PiperEnv(PiperConfig())
        assert env.low_level is None
        driver = _MockLowLevel()
        env._inner = driver
        assert env.low_level is driver


class TestPiperEnvNoDriverForwarding:
    """Driver attrs that are NOT on the Env contract are not proxied."""

    def test_driver_attr_not_forwarded(self):
        """Attrs not on the Env contract still require env.low_level access."""
        from jiuwensymbiosis.adapters.piper.env import PiperEnv

        env = PiperEnv(PiperConfig())
        env._inner = _MockLowLevel()
        # tool_offset_mm IS on the Env contract now → should work via property
        assert env.tool_offset_mm == 135.8
        # grab_frames is NOT on the Env contract → must go through low_level
        with pytest.raises(AttributeError):
            _ = env.grab_frames

    def test_home_pose_prop(self):
        from jiuwensymbiosis.adapters.piper.env import PiperEnv

        env = PiperEnv(PiperConfig())
        env._inner = _MockLowLevel()
        hp = env.home_pose
        assert hp is not None
        assert hp.x == 200
