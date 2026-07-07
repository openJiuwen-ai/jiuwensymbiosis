# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters.piper.api — structural and config-level tests.

Hardware-dependent paths (actual motion, detection) are tested via mock injection.
"""

from __future__ import annotations


class TestPiperApiStructure:
    def test_api_has_robot_tool_methods(self):
        from jiuwensymbiosis.adapters.piper.api import PiperApi

        expected_methods = [
            "home",
            "get_pose",
            "get_home_pose",
            "goto_xyzr",
            "close_gripper",
            "open_gripper",
            "get_grasp_info_simple",
            "pixel_to_base_xyz",
            "get_image",
            "analyze_scene",
        ]
        for name in expected_methods:
            method = getattr(PiperApi, name, None)
            assert method is not None, f"PiperApi.{name} not found"
            assert hasattr(method, "__robot_tool__"), f"PiperApi.{name} missing @robot_tool"

    def test_api_capabilities(self):
        from jiuwensymbiosis.adapters.piper.api import PiperApi
        from jiuwensymbiosis.adapters.piper.config import PiperConfig
        from jiuwensymbiosis.adapters.piper.env import PiperEnv

        api = PiperApi(PiperEnv(PiperConfig()))
        assert api.capabilities == frozenset(
            {
                "motion.cartesian",
                "motion.joint",
                "grasp.parallel",
                "vision.detection",
            }
        )


class _SpyDriver:
    """Records driver calls; satisfies what PiperApi/PiperEnv delegate to."""

    def __init__(self):
        self.log: list = []
        self.home_pose = type("P", (), {"x": 200.0, "y": 0.0, "z": 400.0, "rx": 0.0, "ry": 90.0, "rz": 0.0})()
        self.tool_offset_mm = 135.8
        self.z_min_safe = 50.0

    def home(self):
        self.log.append("home")

    def get_pose(self):
        return type("P", (), {"x": 1.0, "y": 2.0, "z": 3.0, "rx": 0.0, "ry": 0.0, "rz": 7.0})()

    def move_to_pose_blocking(self, pose):
        self.log.append(("move", pose))

    def move_joint_blocking(self, q, *, timeout_s=30.0):
        self.log.append(("joint", list(q)))

    def set_gripper(self, on):
        self.log.append(("gripper", on))


class TestPiperApiDelegatesThroughEnv:
    """Motion/gripper route api -> env public method -> driver (not via _ll)."""

    def _build(self):
        from jiuwensymbiosis.adapters.piper.api import PiperApi
        from jiuwensymbiosis.adapters.piper.config import PiperConfig
        from jiuwensymbiosis.adapters.piper.env import PiperEnv

        env = PiperEnv(PiperConfig())
        driver = _SpyDriver()
        env._inner = driver
        return PiperApi(env), env, driver

    def test_home_reaches_driver_through_env(self):
        api, _env, driver = self._build()
        api.home()
        assert "home" in driver.log

    def test_move_joint_reaches_driver_through_env(self):
        api, _env, driver = self._build()
        api.move_joint([0, 1, 2, 3, 4, 5])
        assert ("joint", [0, 1, 2, 3, 4, 5]) in driver.log

    def test_goto_pose_reaches_driver_through_env(self):
        from jiuwensymbiosis.adapters.piper.geometry import FlangePose

        api, _env, driver = self._build()
        api.goto_pose(FlangePose(1, 2, 3, 180, 0, 0))
        assert any(c[0] == "move" for c in driver.log)

    def test_goto_xyzr_reaches_driver_through_env(self):
        api, _env, driver = self._build()
        api.goto_xyzr(100.0, 0.0, 200.0, 0.0)
        assert any(c[0] == "move" for c in driver.log)

    def test_close_gripper_calls_env_set_end_effector(self):
        from unittest.mock import MagicMock

        api, env, _driver = self._build()
        env.set_end_effector = MagicMock()
        api.close_gripper()
        env.set_end_effector.assert_called_once_with(True)

    def test_open_gripper_engages_driver_via_env(self):
        api, _env, driver = self._build()
        api.open_gripper()
        assert ("gripper", False) in driver.log

    def test_grasp_z_floor_reads_env_property(self):
        # env.z_min_safe (formal contract) is used, not getattr on the driver.
        api, env, _driver = self._build()
        assert env.z_min_safe == 50.0  # comes from the spy driver via PiperEnv property
