# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.api.base."""

from __future__ import annotations

from jiuwensymbiosis.api import defaults
from jiuwensymbiosis.api.actions import GET_IMAGE, GET_POSE, GOTO_XYZR, implements
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.api.decorators import robot_tool
from jiuwensymbiosis.env.mock import MockArmEnv


class SimpleApi(BaseRobotApi):
    @robot_tool
    def my_tool(self) -> dict:
        return {"ok": True}


class MotionVisionApi(BaseRobotApi):
    capability = {"planning.reachability"}  # a marker capability: no action of its own

    @implements(GET_POSE)
    def get_pose(self) -> dict:
        return defaults.get_pose(self)

    @implements(GOTO_XYZR)
    def goto_xyzr(self, x: float, y: float, z: float, r: float | None = None) -> None:
        self.env.move(x, y, z, r)

    @implements(GET_IMAGE)
    def get_image(self):
        return None


class TestBaseRobotApiCapabilities:
    def test_simple_api_capabilities(self):
        env = MockArmEnv()
        api = SimpleApi(env)
        assert api.capabilities == frozenset()

    def test_capabilities_come_from_the_actions_implemented(self):
        env = MockArmEnv()
        api = MotionVisionApi(env)
        caps = api.capabilities
        assert "motion.cartesian" in caps
        assert "vision.camera" in caps

    def test_a_marker_capability_still_counts(self):
        # planning.reachability has no action to be inferred from, so the class attr
        # is the only way to advertise it.
        assert "planning.reachability" in MotionVisionApi(MockArmEnv()).capabilities

    def test_describe(self):
        env = MockArmEnv()
        api = SimpleApi(env)
        desc = api.describe()
        assert "name" in desc
        assert "env_capabilities" in desc
        assert "api_capabilities" in desc
