# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""SafetyRail — software pre-flight checks before motion tools fire.

This rail does NOT replace hardware E-stop. It's a cheap second line of
defense against LLM hallucinations like "goto_xyzr(0, 0, -50)" when the
table is at z=0. The env-level driver should already enforce these limits;
we just reject earlier with a clearer message so the LLM can self-correct without raising.

Which tools are watched is derived from the body's declared capabilities
(``_CAPABILITY_WATCH_TOOLS``), so a mobile manipulator gets its base / lifter /
waist commands checked without anyone re-listing tool names per adapter.

Currently checks:
- ``motion.cartesian`` — Z floor: explicit ``z_floor_mm``, else the env's
  ``z_min_safe``; XY bounds: explicit ``xy_bounds_mm``, else the env's
  ``workspace_bounds`` (unless ``enforce_xy_from_env=False``).
- ``motion.joint`` — joint soft limits on ``move_joint(q)``: explicit
  ``joint_limits``, else the env's. Each rejected branch (missing q / wrong
  type / wrong length / non-finite / out of range) gets its own message.
- ``motion.base`` — per-command translation / turn cap from the env's
  ``base_step_limits``, so a hallucinated "drive 50 m" never reaches the wheels.
- ``motion.lift`` — lifter joint range from the env's ``lift_limits``.
- ``motion.waist`` — per-command torso yaw cap from ``waist_step_limit_rad``.

Absent limits mean "no range check" (the type / finite checks still run) —
same contract as ``joint_limits``, so a body opts in by declaring an envelope.

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

# Capability → the tools whose policy that capability switches on. Adding a mobile
# body means declaring its capabilities, not editing a tool-name list here.
_CAPABILITY_WATCH_TOOLS: dict[str, frozenset[str]] = {
    # goto_flange_pose is a body-only entry (it speaks the FLANGE frame, so it never enters the
    # shared vocabulary) — but it drives the arm, so it is watched exactly like the others.
    "motion.cartesian": frozenset({"goto_xyzr", "goto_pose", "goto_flange_pose"}),
    "motion.joint": frozenset({"move_joint"}),
    "motion.base": frozenset({"navigate_relative", "rotate_base", "drive_arc"}),
    "motion.lift": frozenset({"set_lift_pose"}),
    "motion.waist": frozenset({"turn_waist"}),
}

# Always watched on top of the derived set. These three predate capability-derived
# watching and some envs under-declare the arm capability behind them; watching a
# tool a body does not have is free (it never fires), missing one it does have is a hole.
_ALWAYS_WATCHED: frozenset[str] = frozenset({"goto_xyzr", "goto_pose", "move_joint"})

_BASE_TOOLS: frozenset[str] = _CAPABILITY_WATCH_TOOLS["motion.base"]


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
        # Default: whatever the body's capabilities imply. ``home`` is always safe.
        self.watch_tools = set(watch_tools) if watch_tools else self._derive_watch_tools(session)
        self.trace_sink = trace_sink

    @staticmethod
    def _derive_watch_tools(session: Any) -> set[str]:
        """Tools to intercept, from the session's declared capabilities.

        Reads env ∪ api rather than the tool-gating intersection: a capability
        either side declares is hardware that can move, and over-watching costs
        nothing (the tool simply never fires).
        """
        caps: set[str] = set()
        for holder in (getattr(session, "env", None), getattr(session, "api", None)):
            declared = getattr(holder, "capabilities", None)
            if declared:
                caps |= set(declared)
        watched = set(_ALWAYS_WATCHED)
        for cap in caps:
            watched |= _CAPABILITY_WATCH_TOOLS.get(cap, frozenset())
        return watched

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

        self.validate_motion(tool_name, args)

    def validate_motion(self, tool_name: str, args: dict[str, Any]) -> None:
        """Synchronously apply the rail's motion policy without callback overhead.

        Real-time servo bindings call :meth:`validate_pose` (the flat-key fast
        path) on every target pose before dispatch. The normal tool path
        delegates here from ``before_tool_call``, so both paths share one
        Z/XY/joint policy implementation.
        """

        if tool_name not in self.watch_tools:
            return

        if tool_name == "move_joint":
            self._check_joint_limits(tool_name, args)
            return
        if tool_name in _BASE_TOOLS:
            self._check_base_step(tool_name, args)
            return
        if tool_name == "set_lift_pose":
            self._check_lift_pose(tool_name, args)
            return
        if tool_name == "turn_waist":
            self._check_waist_step(tool_name, args)
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
            x, y, z = pose_obj.get("x"), pose_obj.get("y"), pose_obj.get("z")
        else:
            x, y, z = args.get("x"), args.get("y"), args.get("z")

        self._apply_xyz_policy(x, y, z, tool_name=tool_name)

    def validate_pose(self, pose: Any, *, tool_name: str = "servo_to_pose") -> None:
        """Flat-key fast path for real-time servo ticks.

        ``ServoBinding.servo_to`` calls this on every target pose. Unlike
        :meth:`validate_motion`, it takes the pose dict directly (no ``{"pose": …}``
        wrapper, no tool-name dispatch) and skips the string-arg coercion the
        ``before_tool_call`` path needs — so a 30–200 Hz servo loop pays only
        the finite + Z-floor + XY-bounds check, with the correct ``tool_name``
        in any trace reject event (not a synthesized ``goto_pose``).
        """
        if isinstance(pose, dict):
            x, y, z = pose.get("x"), pose.get("y"), pose.get("z")
        else:
            x = getattr(pose, "x", None)
            y = getattr(pose, "y", None)
            z = getattr(pose, "z", None)
        self._apply_xyz_policy(x, y, z, tool_name=tool_name)

    def _apply_xyz_policy(self, x: Any, y: Any, z: Any, *, tool_name: str) -> None:
        """Shared Z-floor + XY-bounds + finite check for one Cartesian target.

        ``x``/``y``/``z`` are the raw coordinate values (numbers, numeric
        strings, or ``None`` for "absent"). ``None`` means "no coordinate
        supplied" → that axis is skipped (same "no param → no rejection"
        contract as the tool path). Non-numeric / non-finite values raise.
        """

        cx = self._coerce_number(tool_name, "x", x)
        cy = self._coerce_number(tool_name, "y", y)
        cz = self._coerce_number(tool_name, "z", z)

        z_floor = self._resolve_z_floor()
        if cz is not None and z_floor is not None and cz < float(z_floor):
            reason = f"z={z} below z_floor={z_floor}"
            self._notify_reject(tool_name, reason)
            raise ValueError(f"SafetyRail: refusing {tool_name}: {reason}. Either raise z, or call home() first.")

        xy_bounds = self._resolve_xy_bounds()
        if xy_bounds is not None:
            xmin, ymin, xmax, ymax = xy_bounds
            if cx is not None and not xmin <= cx <= xmax:
                reason = f"x={x} out of bounds [{xmin}, {xmax}]"
                self._notify_reject(tool_name, reason)
                raise ValueError(f"SafetyRail: refusing {tool_name}: {reason}.")
            if cy is not None and not ymin <= cy <= ymax:
                reason = f"y={y} out of bounds [{ymin}, {ymax}]"
                self._notify_reject(tool_name, reason)
                raise ValueError(f"SafetyRail: refusing {tool_name}: {reason}.")

    def _coerce_number(self, tool_name: str, name: str, raw: Any) -> float | None:
        """One motion argument as a finite float. ``None`` stays ``None`` ("absent → skip")."""
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            reason = f"{name} is not a number: {raw!r}"
            self._notify_reject(tool_name, reason)
            raise ValueError(f"SafetyRail: refusing {tool_name}: {reason}.") from None
        if not math.isfinite(value):
            reason = f"{name} is non-finite: {raw!r}"
            self._notify_reject(tool_name, reason)
            raise ValueError(f"SafetyRail: refusing {tool_name}: {reason}.")
        return value

    def _reject(self, tool_name: str, reason: str) -> ValueError:
        """Notify the trace sink and build the rejection to raise."""
        self._notify_reject(tool_name, reason)
        return ValueError(f"SafetyRail: refusing {tool_name}: {reason}.")

    def _check_base_step(self, tool_name: str, args: dict[str, Any]) -> None:
        """Cap one mobile-base command's translation and turn.

        Relative verbs have no absolute frame to bound, so the envelope is a
        per-command step cap: it stops a hallucinated "drive 50 m" without
        constraining where the base may legitimately end up.
        """
        dyaw = self._coerce_number(tool_name, "dyaw_rad", args.get("dyaw_rad"))
        if tool_name == "drive_arc":
            radius = self._coerce_number(tool_name, "radius_m", args.get("radius_m"))
            # Arc length is what the wheels actually travel; radius alone says nothing.
            distance = None if radius is None or dyaw is None else abs(radius * dyaw)
        elif tool_name == "rotate_base":
            distance = 0.0
        else:
            dx = self._coerce_number(tool_name, "dx_m", args.get("dx_m"))
            dy = self._coerce_number(tool_name, "dy_m", args.get("dy_m"))
            distance = None if dx is None and dy is None else math.hypot(dx or 0.0, dy or 0.0)

        limits = self._resolve_base_step_limits()
        if limits is None:
            return
        max_step, max_turn = limits
        if distance is not None and distance > max_step:
            raise self._reject(tool_name, f"base step {distance:.3f}m exceeds max {max_step}m")
        if dyaw is not None and abs(dyaw) > max_turn:
            raise self._reject(tool_name, f"base turn {abs(dyaw):.3f}rad exceeds max {max_turn}rad")

    def _check_lift_pose(self, tool_name: str, args: dict[str, Any]) -> None:
        """Validate an absolute lifter target, the ``move_joint`` analogue for the torso."""
        q_lifter = args.get("q_lifter")
        if q_lifter is None:
            raise self._reject(tool_name, "missing required lifter joint map q_lifter")
        if not isinstance(q_lifter, dict):
            raise self._reject(tool_name, f"q_lifter must be a mapping, got {type(q_lifter).__name__}")

        limits = self._resolve_lift_limits()
        for name, raw in q_lifter.items():
            value = self._coerce_number(tool_name, str(name), raw)
            if value is None or limits is None:
                continue
            bounds = limits.get(str(name))
            if bounds is None:
                raise self._reject(tool_name, f"unknown lifter joint {name!r} (known: {sorted(limits)})")
            lo, hi = float(bounds[0]), float(bounds[1])
            if not lo <= value <= hi:
                raise self._reject(tool_name, f"{name}={value} out of limits [{lo}, {hi}]")

    def _check_waist_step(self, tool_name: str, args: dict[str, Any]) -> None:
        """Cap one torso-yaw command; ``turn_waist`` is a delta, so the envelope is a step cap."""
        delta = self._coerce_number(tool_name, "delta_rad", args.get("delta_rad"))
        limit = self._resolve_waist_step_limit()
        if delta is not None and limit is not None and abs(delta) > limit:
            raise self._reject(tool_name, f"waist turn {abs(delta):.3f}rad exceeds max {limit}rad")

    def _env_limit(self, attr: str) -> Any:
        """Read one safety-envelope value off the env; missing / unreadable → None."""
        return getattr(getattr(self.session, "env", None), attr, None)

    def _resolve_base_step_limits(self) -> tuple[float, float] | None:
        """Base step cap from the env's ``base_step_limits``, else None."""
        limits = self._env_limit("base_step_limits")
        if limits is None:
            return None
        try:
            max_step, max_turn = limits
            return (float(max_step), float(max_turn))
        except (TypeError, ValueError):
            return None

    def _resolve_lift_limits(self) -> dict[str, tuple[float, float]] | None:
        """Lifter joint limits from the env's ``lift_limits``, else None."""
        limits = self._env_limit("lift_limits")
        if limits is None:
            return None
        return cast("dict[str, tuple[float, float]] | None", limits)

    def _resolve_waist_step_limit(self) -> float | None:
        """Waist step cap from the env's ``waist_step_limit_rad``, else None."""
        limit = self._env_limit("waist_step_limit_rad")
        if limit is None:
            return None
        try:
            return float(limit)
        except (TypeError, ValueError):
            return None

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
