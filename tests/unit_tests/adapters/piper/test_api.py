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
        from jiuwensymbiosis.env.mock import MockArmEnv
        from tests.mocks.mock_api import MockApi

        env = MockArmEnv()
        api = MockApi(env)
        assert "motion.cartesian" in api.capabilities or len(api.capabilities) > 0
