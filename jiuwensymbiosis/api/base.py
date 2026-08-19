# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Base class for robot control APIs.

A concrete robot API subclasses `BaseRobotApi` and binds each action it offers to
one entry of the shared vocabulary (`jiuwensymbiosis.api.actions`) with
`@implements`. The bindings declare what the API offers; the env declares what
hardware can do; the agent profile intersects the two and only exposes tools both
sides agree on.

Example:

    class RobotApi(BaseRobotApi):
        @implements(GOTO_XYZR)
        def goto_xyzr(self, x, y, z, r=None): ...     # body-specific geometry

        @implements(ACTIVATE_SUCTION)
        def activate_suction(self): return defaults.activate_suction(self)
"""

from __future__ import annotations

from typing import Any

from jiuwensymbiosis.api import defaults
from jiuwensymbiosis.api.actions import HOME, implements
from jiuwensymbiosis.api.memory import ExecutionMemory
from jiuwensymbiosis.env.base import BaseRobotEnv


class BaseRobotApi:
    """Holds a reference to the underlying env and exposes the body's actions."""

    def __init__(self, env: BaseRobotEnv) -> None:
        """Store a reference to the underlying env."""
        self.env = env
        # What running actions has established (locations sensed, self-state).
        # Maintained by the dispatch layer from each action's declared contract —
        # adapters must not hand-roll their own caches for this.
        self.memory = ExecutionMemory()
        # The last sensing, kept HERE rather than inside the Scene3D component because both
        # the component and the approach loops (motion/approach.py) read it, and a body's
        # own grasp/place steps consume it. One copy, one owner, no forwarding.
        self._last_detection: dict | None = None
        self._last_surface: dict | None = None

    # The Scene3D / Approach components a body HOLDS read and write this state, so it is a
    # cross-object contract: these accessors say so instead of exposing the underscore.
    @property
    def last_detection(self) -> dict | None:
        """The most recent object sensing, or None when nothing valid has been sensed."""
        return self._last_detection

    @last_detection.setter
    def last_detection(self, value: dict | None) -> None:
        self._last_detection = value

    @property
    def last_surface(self) -> dict | None:
        """The most recent support-surface sensing, or None."""
        return self._last_surface

    @last_surface.setter
    def last_surface(self, value: dict | None) -> None:
        self._last_surface = value

    # ``home`` lives here rather than on a capability mixin because returning to a
    # safe posture is not an optional capability — every body owes one, and
    # ``BaseRobotEnv.home()`` is part of the base hardware contract. A mobile body
    # with no Cartesian arm needs it just as much as a 6-DoF one.
    @implements(HOME)
    def home(self) -> None:
        """Return to the home pose (delegates to the Env verb)."""
        defaults.home(self)

    @property
    def capabilities(self) -> frozenset[str]:
        """What this api offers: the capabilities of every action it implements, plus
        any ``capability`` attr declared across the MRO.

        Deriving from the implemented actions means a body cannot implement an action
        and forget to advertise its capability — the gate would then silently drop the
        tool. The ``capability`` attrs still contribute, because a *marker* capability
        (``vision.camera`` on a body that only streams frames, ``planning.reachability``)
        has no action of its own to be inferred from.
        """
        caps: set[str] = set()
        for cls in type(self).__mro__:
            cap = getattr(cls, "capability", None)
            if isinstance(cap, str):
                caps.add(cap)
            elif isinstance(cap, (set, frozenset, list, tuple)):
                caps.update(cap)
            for attr_value in cls.__dict__.values():
                meta = getattr(attr_value, "__robot_tool__", None)
                if meta is not None and meta.capability:
                    caps.add(meta.capability)
        return frozenset(caps)

    def describe(self) -> dict[str, Any]:
        """Short JSON-able summary; goes into the system prompt of the agent."""
        return {
            "name": getattr(self.env, "name", "robot"),
            "env_capabilities": sorted(self.env.capabilities),
            "api_capabilities": sorted(self.capabilities),
        }
