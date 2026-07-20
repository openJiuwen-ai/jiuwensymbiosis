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
- Joint soft limits on ``move_joint(q)``: explicit ``joint_limits``, else the
  env's ``joint_limits``. Each rejected branch (missing q / wrong type /
  wrong length / non-finite / out of range) gets its own message.

Reject mechanism: forces a tool error result instead of executing the tool,
by raising via ``ctx.request_force_finish`` is not appropriate here (we
only want to skip the one tool, not the whole loop). Instead we raise a
``ValueError`` from ``before_tool_call``; openjiuwen turns this into a
tool-exception event the LLM sees and reasons about.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, cast

from jiuwensymbiosis.agent.abstractions import AgentRail
from jiuwensymbiosis.agent.trace import TraceEventSink

logger = logging.getLogger(__name__)


def _coerce_tool_args(value: Any) -> dict[str, Any]:
    """Normalise ``tool_args`` to a dict, parsing a JSON string if needed.

    openjiuwen's ``ToolCall.arguments`` (and thus ``ctx.inputs.tool_args`` at
    ``before_tool_call`` time) is a **JSON string**, not a dict — the dict only
    appears inside the tool's ``invoke`` *after* rails run. Without this parse,
    a string ``tool_args`` would be dropped to ``{}`` and every motion-param
    check (z floor, XY bounds) would silently no-op. Returns ``{}`` for ``None``,
    non-dict, or unparseable input — never raises, so safety enforcement degrades
    to "no params → no rejection" (the prior behaviour) rather than crashing.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


class SafetyRail(AgentRail):
    """Reject obviously unsafe motion-tool calls before the env sees them."""

    def __init__(
        self,
        session: Any,
        *,
        z_floor_mm: float | None = None,
        xy_bounds_mm: tuple[float, float, float, float] | None = None,  # (xmin, ymin, xmax, ymax)
        joint_limits: dict[str, tuple[float, float]] | None = None,
        watch_tools: set[str] | None = None,
        enforce_xy_from_env: bool = True,
        trace_sink: TraceEventSink | None = None,
    ) -> None:
        """Initialize the safety rail with optional bounds.

        When ``xy_bounds_mm`` is not given, XY bounds fall back to the env's
        ``workspace_bounds`` property unless ``enforce_xy_from_env`` is False.
        When ``joint_limits`` is not given, it falls back to the env's
        ``joint_limits`` property.

        ``trace_sink`` (optional) receives a structured event when this rail
        rejects a call — injected by ``build_robot_agent`` when tracing is on.
        """
        super().__init__()
        self.session = session
        self.z_floor = z_floor_mm
        self.xy_bounds = xy_bounds_mm
        self.joint_limits = joint_limits
        self.enforce_xy_from_env = enforce_xy_from_env
        # Default: intercept cartesian + joint moves. ``home`` is always safe.
        self.watch_tools = set(watch_tools) if watch_tools else {"goto_xyzr", "goto_pose", "move_joint"}
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

        openjiuwen delivers ``tool_args`` as a **JSON string** (``ToolCall.arguments``
        is typed ``str``), not a dict — the dict only materialises inside the
        tool's ``invoke`` *after* rails run. So we parse the string here;
        unparseable / non-dict args fall back to ``{}`` (→ no motion params →
        no rejection, same as before, never a false positive).
        """
        inputs = getattr(ctx, "inputs", None)
        tool_name = getattr(inputs, "tool_name", "") or ""
        args = _coerce_tool_args(getattr(inputs, "tool_args", None))

        # Unwrap robot_control dispatch: tool_name="robot_control", action inside args
        if tool_name == "robot_control":
            action = args.get("action", "")
            params = args.get("params", {})
            if isinstance(params, dict) and action:
                tool_name = str(action)
                args = params

        if tool_name not in self.watch_tools:
            return

        if tool_name == "move_joint":
            self._check_joint_limits(tool_name, args)
            return

        # goto_pose ships x/y/z inside a nested ``pose`` object (one Cartesian
        # pose as a value object); goto_xyzr keeps them top-level. Flatten the
        # nested pose so the Z/XY checks below cover the SO-101 ``goto_pose``
        # signature. Missing fields fall through to None — same "no param → no
        # rejection" path as a flat call missing a coordinate.
        #
        # NOTE: this only unpacks SO-101's ``x/y/z`` field names. Piper's
        # ``goto_pose`` uses ``x_mm/y_mm/z_mm`` in the FLANGE frame, while the
        # rail's ``z_min_safe`` is the TIP-frame floor — different coordinate
        # systems. Unpacking ``z_mm`` against ``z_min_safe`` would let a
        # flange-Z below the tip floor pass the pre-check (false safety).
        # Piper's nested pose is therefore NOT unpacked here; its driver-level
        # ``check_flange_z`` remains the enforcement. A future change that
        # exposes ``flange_z_min_safe`` on PiperEnv could close that gap.
        pose_obj = args.get("pose") if tool_name == "goto_pose" else None
        if isinstance(pose_obj, dict):
            x = pose_obj.get("x")
            y = pose_obj.get("y")
            z = pose_obj.get("z")
        else:
            x, y, z = args.get("x"), args.get("y"), args.get("z")

        z_floor = self._resolve_z_floor()
        if z is not None and z_floor is not None and float(z) < float(z_floor):
            reason = f"z={z} below z_floor={z_floor}"
            self._notify_reject(tool_name, reason)
            raise ValueError(f"SafetyRail: refusing {tool_name}: {reason}. Either raise z, or call home() first.")

        xy_bounds = self._resolve_xy_bounds()
        if xy_bounds is not None:
            xmin, ymin, xmax, ymax = xy_bounds
            if x is not None and not xmin <= float(x) <= xmax:
                reason = f"x={x} out of bounds [{xmin}, {xmax}]"
                self._notify_reject(tool_name, reason)
                raise ValueError(f"SafetyRail: refusing {tool_name}: {reason}.")
            if y is not None and not ymin <= float(y) <= ymax:
                reason = f"y={y} out of bounds [{ymin}, {ymax}]"
                self._notify_reject(tool_name, reason)
                raise ValueError(f"SafetyRail: refusing {tool_name}: {reason}.")

    def _resolve_z_floor(self) -> float | None:
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

    def _resolve_xy_bounds(self) -> tuple[float, float, float, float] | None:
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

    def _resolve_joint_limits(self) -> dict[str, tuple[float, float]] | None:
        """Joint limits: explicit ``joint_limits``, else the env's, else None."""
        if self.joint_limits is not None:
            return self.joint_limits
        env = getattr(self.session, "env", None)
        limits = getattr(env, "joint_limits", None)
        if limits is None:
            return None
        return cast("dict[str, tuple[float, float]] | None", limits)

    def _check_joint_limits(self, tool_name: str, args: dict[str, Any]) -> None:
        """Validate a ``move_joint(q)`` call before it reaches the env.

        Distinct error messages per failure so the LLM doesn't misread "length
        mismatch" as "lower a joint angle".
        """
        q = args.get("q")
        if q is None:
            reason = "missing required joint vector q"
            self._notify_reject(tool_name, reason)
            raise ValueError(f"SafetyRail: refusing {tool_name}: {reason}.")
        if not isinstance(q, (list, tuple)):
            reason = f"q must be a list or tuple, got {type(q).__name__}"
            self._notify_reject(tool_name, reason)
            raise ValueError(f"SafetyRail: refusing {tool_name}: {reason}.")

        limits = self._resolve_joint_limits()
        names: list[str] = list(limits.keys()) if limits is not None else []
        if limits is not None and len(q) != len(names):
            reason = f"q has {len(q)} joints but limits has {len(names)}"
            self._notify_reject(tool_name, reason)
            raise ValueError(f"SafetyRail: refusing {tool_name}: {reason}.")

        for i, raw in enumerate(q):
            try:
                v = float(raw)
            except (TypeError, ValueError):
                reason = f"{self._joint_label(names, i)} is not a number: {raw!r}"
                self._notify_reject(tool_name, reason)
                raise ValueError(f"SafetyRail: refusing {tool_name}: {reason}.") from None
            if not math.isfinite(v):
                label = self._joint_label(names, i)
                reason = f"{label} is non-finite: {raw!r}"
                self._notify_reject(tool_name, reason)
                raise ValueError(f"SafetyRail: refusing {tool_name}: {reason}.")
            if limits is not None:
                lo, hi = limits[names[i]]
                if not float(lo) <= v <= float(hi):
                    label = self._joint_label(names, i)
                    reason = f"{label}={v} out of limits [{float(lo)}, {float(hi)}]"
                    self._notify_reject(tool_name, reason)
                    raise ValueError(f"SafetyRail: refusing {tool_name}: {reason}.")

    @staticmethod
    def _joint_label(names: list[str], i: int) -> str:
        """Best-effort joint name for an error message: ``J2`` if limits are
        configured with that key, else the positional ``q[1]``."""
        return names[i] if i < len(names) else f"q[{i}]"
