# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters.piper.env."""

from __future__ import annotations

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
