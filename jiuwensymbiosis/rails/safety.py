# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""SafetyRail — software pre-flight checks before motion tools fire.

This rail does NOT replace hardware E-stop. It's a cheap second line of
defense against LLM hallucinations like "goto_xyzr(0, 0, -50)" when the
table is at z=0. The env-level driver should already enforce these limits;
we just reject earlier with a clearer message so the LLM can self-correct without raising.

Currently checks:
- Cartesian Z floor: explicit ``z_floor_mm``, else the env's ``z_min_safe``.
- Cartesian XY bounds: explicit ``xy_bounds_mm``, else the env's
  ``workspace_bounds`` (unless ``enforce_xy_from_env=False``).

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
from jiuwensymbiosis.agent.trace import TraceEventSink

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
        enforce_xy_from_env: bool = True,
        trace_sink: Optional[TraceEventSink] = None,
    ) -> None:
        """Initialize the safety rail with optional bounds.

        When ``xy_bounds_mm`` is not given, XY bounds fall back to the env's
        ``workspace_bounds`` property unless ``enforce_xy_from_env`` is False.

        ``trace_sink`` (optional) receives a structured event when this rail
        rejects a call — injected by ``build_robot_agent`` when tracing is on.
        """
        super().__init__()
        self.session = session
        self.z_floor = z_floor_mm
        self.xy_bounds = xy_bounds_mm
        self.enforce_xy_from_env = enforce_xy_from_env
        # Default: only intercept cartesian moves. ``home`` is always safe.
        self.watch_tools = set(watch_tools) if watch_tools else {"goto_xyzr", "goto_pose"}
        self.trace_sink = trace_sink

    def _notify_reject(self, tool_name: str, reason: str) -> None:
        sink = self.trace_sink
        if sink is None:
            return
        try:
            sink.record_rail_event(
                rail_name="SafetyRail",
                kind="reject",
                detail={"tool_name": tool_name, "reason": reason},
                success=False,
            )
        except (AttributeError, TypeError, ValueError):
            # tracing must never break safety enforcement
            pass

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
            reason = f"z={z} below z_floor={z_floor}"
            self._notify_reject(tool_name, reason)
            raise ValueError(
                f"SafetyRail: refusing {tool_name}: {reason}. "
                "Either raise z, or call home() first."
            )

        xy_bounds = self._resolve_xy_bounds()
        if xy_bounds is not None:
            xmin, ymin, xmax, ymax = xy_bounds
            x, y = args.get("x"), args.get("y")
            if x is not None and not xmin <= float(x) <= xmax:
                reason = f"x={x} out of bounds [{xmin}, {xmax}]"
                self._notify_reject(tool_name, reason)
                raise ValueError(f"SafetyRail: refusing {tool_name}: {reason}.")
            if y is not None and not ymin <= float(y) <= ymax:
                reason = f"y={y} out of bounds [{ymin}, {ymax}]"
                self._notify_reject(tool_name, reason)
                raise ValueError(f"SafetyRail: refusing {tool_name}: {reason}.")

    def _resolve_z_floor(self) -> Optional[float]:
        """Z floor: explicit ``z_floor``, else the env's ``z_min_safe``, else None."""
        if self.z_floor is not None:
            return self.z_floor
        env = getattr(self.session, "env", None)
        val = getattr(env, "z_min_safe", None)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def _resolve_xy_bounds(self) -> Optional[tuple[float, float, float, float]]:
        """XY bounds: explicit ``xy_bounds``, else the env's ``workspace_bounds``."""
        if self.xy_bounds is not None:
            return self.xy_bounds
        if not self.enforce_xy_from_env:
            return None
        env = getattr(self.session, "env", None)
        bounds = getattr(env, "workspace_bounds", None)
        if bounds is None:
            return None
        try:
            xmin, ymin, xmax, ymax = bounds
            return (float(xmin), float(ymin), float(xmax), float(ymax))
        except (TypeError, ValueError):
            return None
