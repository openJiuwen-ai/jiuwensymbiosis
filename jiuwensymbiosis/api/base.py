# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Capability-mixin base for robot control APIs.

Every concrete robot API mixes a `BaseRobotApi` subclass with one or more
capability mixins from `jiuwensymbiosis.api.mixins`. The mixins declare what
the API offers; the env declares what hardware can do; the agent profile
intersects the two and only exposes tools both sides agree on.

Example:

    class RobotApi(MotionMixin, SuctionMixin, VisionMixin, BaseRobotApi):
        # Implementations of the abstract methods declared by each mixin.
        def goto_xyzr(self, x, y, z, r=None): ...
        def activate_suction(self): ...
"""

from __future__ import annotations

from typing import Any

from jiuwensymbiosis.env.base import BaseRobotEnv


class BaseRobotApi:
    """Holds a reference to the underlying env and exposes @robot_tool methods."""

    def __init__(self, env: BaseRobotEnv) -> None:
        """Store a reference to the underlying env."""
        self.env = env

    @property
    def capabilities(self) -> frozenset[str]:
        """Union of ``capability`` attrs declared across the MRO."""
        caps: set[str] = set()
        for cls in type(self).__mro__:
            cap = getattr(cls, "capability", None)
            if isinstance(cap, str):
                caps.add(cap)
            elif isinstance(cap, (set, frozenset, list, tuple)):
                caps.update(cap)
        return frozenset(caps)

    def describe(self) -> dict[str, Any]:
        """Short JSON-able summary; goes into the system prompt of the agent."""
        return {
            "name": getattr(self.env, "name", "robot"),
            "env_capabilities": sorted(self.env.capabilities),
            "api_capabilities": sorted(self.capabilities),
        }
