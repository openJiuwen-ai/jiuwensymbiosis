# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.api.base."""

from __future__ import annotations

from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.api.decorators import robot_tool
from jiuwensymbiosis.api.mixins import MotionMixin, VisionMixin
from jiuwensymbiosis.env.mock import MockArmEnv


class SimpleApi(BaseRobotApi):
    @robot_tool
    def my_tool(self) -> dict:
        return {"ok": True}


class MotionVisionApi(MotionMixin, VisionMixin, BaseRobotApi):
    capability = {"motion.cartesian"}  # will be overridden by MRO

    @robot_tool(desc="home", tags=["motion"])
    def home(self) -> None:
        self.env.home()

    @robot_tool
    def get_pose(self) -> dict:
        return self.env.get_observation().pose or {}

    @robot_tool
    def get_home_pose(self) -> dict:
        return self.env.home_pose()

    @robot_tool(tags=["motion"])
    def goto_xyzr(self, x: float, y: float, z: float, r: float | None = None) -> None:
        self.env.move(x, y, z, r)

    @robot_tool
    def get_grasp_info_simple(self, object_name: str) -> dict:
        return {"ok": True}

    @robot_tool
    def pixel_to_base_xyz(self, u: float, v: float, depth_m: float) -> dict:
        return {"x": 0, "y": 0, "z": 0}

    @robot_tool
    def get_image(self):
        return None

    @robot_tool
    def analyze_scene(self, object_name: str | None = None) -> dict:
        return {"ok": True}


class TestBaseRobotApiCapabilities:
    def test_simple_api_capabilities(self):
        env = MockArmEnv()
        api = SimpleApi(env)
        assert api.capabilities == frozenset()

    def test_mixin_capabilities_union(self):
        env = MockArmEnv()
        api = MotionVisionApi(env)
        caps = api.capabilities
        assert "motion.cartesian" in caps
        assert "vision.detection" in caps

    def test_describe(self):
        env = MockArmEnv()
        api = SimpleApi(env)
        desc = api.describe()
        assert "name" in desc
        assert "env_capabilities" in desc
        assert "api_capabilities" in desc
