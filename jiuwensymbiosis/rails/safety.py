# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""SafetyRail — software pre-flight checks before motion tools fire.

This rail does NOT replace hardware E-stop. It's a cheap second line of
defense against LLM hallucinations like "goto_xyzr(0, 0, -50)" when the
table is at z=0. The env-level driver should already enforce these limits;
we just reject earlier with a clearer message so the LLM can self-correct without raising.

Currently checks:
- Cartesian Z floor (default: env's ``z_min_safe`` if exposed, else None)
- Cartesian XY workspace bounds (if ``session.workspace_bounds`` is set)

Reject mechanism: forces a tool error result instead of executing the tool,
by raising via ``ctx.request_force_finish`` is not appropriate here (we
only want to skip the one tool, not the whole loop). Instead we raise a
``ValueError`` from ``before_tool_call``; openjiuwen turns this into a
tool-exception event the LLM sees and reasons about.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from jiuwensymbiosis.agent.abstractions import AgentRail

logger = logging.getLogger(__name__)


class SafetyRail(AgentRail):
    """Reject obviously unsafe motion-tool calls before the env sees them."""

    def __init__(
        self,
        session: Any,
        *,
        z_floor_mm: Optional[float] = None,
        xy_bounds_mm: Optional[tuple[float, float, float, float]] = None,  # (xmin, ymin, xmax, ymax)
        watch_tools: Optional[set[str]] = None,
    ) -> None:
        """Initialize the safety rail with optional bounds."""
        super().__init__()
        self.session = session
        self.z_floor = z_floor_mm
        self.xy_bounds = xy_bounds_mm
        # Default: only intercept cartesian moves. ``home`` is always safe.
        self.watch_tools = set(watch_tools) if watch_tools else {"goto_xyzr", "goto_pose"}

    async def before_tool_call(self, ctx: Any) -> None:
        """Reject unsafe motion tool calls before execution.

        When ``RobotControlTool`` is used, the tool name is ``"robot_control"``
        and the actual action / params live inside ``tool_args``.  We unwrap
        this so that motion actions dispatched through the single entry point
        are still safety-checked.
        """
        inputs = getattr(ctx, "inputs", None)
        tool_name = getattr(inputs, "tool_name", "") or ""
        args = getattr(inputs, "tool_args", None) or {}
        if not isinstance(args, dict):
            args = {}

        # Unwrap robot_control dispatch: tool_name="robot_control", action inside args
        if tool_name == "robot_control":
            action = args.get("action", "")
            params = args.get("params", {})
            if isinstance(params, dict) and action:
                tool_name = str(action)
                args = params

        if tool_name not in self.watch_tools:
            return

        z = args.get("z")
        z_floor = self._resolve_z_floor()
        if z is not None and z_floor is not None and float(z) < float(z_floor):
            raise ValueError(
                f"SafetyRail: refusing {tool_name}: z={z} below z_floor={z_floor}. "
                "Either raise z, or call home() first."
            )

        if self.xy_bounds is not None:
            xmin, ymin, xmax, ymax = self.xy_bounds
            x, y = args.get("x"), args.get("y")
            if x is not None and not xmin <= float(x) <= xmax:
                raise ValueError(f"SafetyRail: refusing {tool_name}: x={x} out of bounds [{xmin}, {xmax}].")
            if y is not None and not ymin <= float(y) <= ymax:
                raise ValueError(f"SafetyRail: refusing {tool_name}: y={y} out of bounds [{ymin}, {ymax}].")

    def _resolve_z_floor(self) -> Optional[float]:
        """Resolve the Z floor from the explicit value or the env."""
        if self.z_floor is not None:
            return self.z_floor
        env = getattr(self.session, "env", None)
        for attr in ("z_min_safe", "_z_min_safe"):
            try:
                val = getattr(env, attr, None)
                if callable(val):
                    val = val()
                if val is not None:
                    return float(val)
            except Exception:  # noqa: BLE001
                continue
        return None
