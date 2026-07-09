# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Mock API for testing — mirrors examples/piper_pick_demo._MockPiperApi."""

from __future__ import annotations

from typing import Any

from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.api.decorators import robot_tool
from jiuwensymbiosis.api.mixins import (
    MotionMixin,
    ParallelGripperMixin,
    VisionMixin,
)
from jiuwensymbiosis.env.mock import MockArmEnv


class MockApi(MotionMixin, ParallelGripperMixin, VisionMixin, BaseRobotApi):
    """In-memory mock of a Piper-like API for tests."""

    # Tests always construct MockApi with a MockArmEnv, which exposes the
    # simplified move()/set_suction() helpers used below. Narrow the base's
    # BaseRobotEnv annotation so those calls type-check.
    env: MockArmEnv

    def __init__(self, env, *, detection_result: dict | None = None, pixel_result: dict | None = None) -> None:
        super().__init__(env)
        self._call_log: list[str] = []
        self._detection_result = detection_result or {
            "ok": True,
            "position": [230.0, 0.0, 50.0],
            "score": 0.9,
            "pixel_uv": [320, 240],
            "depth_m": 0.20,
        }
        self._pixel_result = pixel_result or {"x": 230.0, "y": 0.0, "z": 50.0}

    # -- motion --
    @robot_tool(desc="home", tags=["motion"])
    def home(self) -> None:
        self._call_log.append("home")
        self.env.home()

    @robot_tool
    def get_pose(self) -> dict:
        self._call_log.append("get_pose")
        return self.env.get_observation().pose or {}

    @robot_tool
    def get_home_pose(self) -> dict:
        hp = self.env.home_pose
        if isinstance(hp, dict):
            return hp
        return {
            "x": hp.x,
            "y": hp.y,
            "z": hp.z,
            "rx": getattr(hp, "rx", 0),
            "ry": getattr(hp, "ry", 0),
            "rz": getattr(hp, "rz", 0),
        }

    @robot_tool(tags=["motion"])
    def goto_xyzr(self, x: float, y: float, z: float, r: float | None = None) -> None:
        self._call_log.append(f"goto_xyzr({x},{y},{z},{r})")
        self.env.move(x, y, z, r)

    # -- gripper --
    # -- suction --
    def activate_suction(self) -> dict:
        self._call_log.append("activate_suction")
        self.env.set_suction(True)
        return {"ok": True, "state": "on"}

    def deactivate_suction(self) -> dict:
        self._call_log.append("deactivate_suction")
        self.env.set_suction(False)
        return {"ok": True, "state": "off"}

    @robot_tool(tags=["grasp"])
    def close_gripper(self, force_n: float | None = None) -> dict:
        self._call_log.append("close_gripper")
        self.env.set_suction(True)
        return {"ok": True, "state": "closed"}

    @robot_tool(tags=["grasp"])
    def open_gripper(self, width_mm: float = 70.0) -> dict:
        self._call_log.append("open_gripper")
        self.env.set_suction(False)
        return {"ok": True, "state": "open"}

    # -- vision --
    @robot_tool
    def get_grasp_info_simple(self, object_name: str) -> dict:
        self._call_log.append(f"get_grasp_info_simple({object_name!r})")
        return dict(self._detection_result)

    @robot_tool
    def pixel_to_base_xyz(self, u: float, v: float, depth_m: float) -> dict:
        return dict(self._pixel_result)

    @robot_tool
    def get_image(self) -> Any:
        obs = self.env.get_observation()
        return obs.rgb

    @robot_tool
    def analyze_scene(self, object_name: str | None = None) -> dict:
        return {"ok": True, "objects": []}

    # -- joint motion (not actually in mixins, but added for strategy tests) --
    def goto_xyzr_joint(self, x: float, y: float, z: float, r: float | None = None) -> None:
        self._call_log.append(f"goto_xyzr_joint({x},{y},{z},{r})")
        self.env.move(x, y, z, r)
