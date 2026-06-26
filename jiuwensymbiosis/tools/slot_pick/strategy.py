# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Body-specific motion + grasp strategy for the slot-pick loop.

``run_slot_pick`` (``skill.py``) is body-agnostic: it decides *where* to move and
*when* to grasp/release, but delegates *how* to a ``SlotPickStrategy``. The
body-specific concerns (reach clamping, singularity retries, gripper vs suction)
are packed into one injected object the adapter supplies.

``GripperStrategy`` is the default 6-DoF + parallel-gripper implementation. Piper uses it
directly (its tilted-tool geometry already lives inside ``PiperApi.goto_xyzr``).
A future suction / SCARA body would supply its own strategy implementing the
same protocol (e.g. via-point motion + ``activate_suction``/wiggle/DI).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class SlotPickStrategy(Protocol):
    """The body-specific seam ``run_slot_pick`` calls into.

    Frames: every ``(x, y, z, r)`` is the agent-facing TIP-frame pose in mm/deg,
    exactly what ``api.goto_xyzr`` accepts.
    """

    def goto_transit(self, x: float, y: float, z: float, r: float) -> None:
        """Cross-workspace transit / reposition move (observe poses, chip-observe)."""

    def goto_process(self, x: float, y: float, z: float, r: float) -> None:
        """Best-effort PROCESS waypoint — must NOT raise; gets as close as the
        workspace allows (reach clamp / safe-z floor / boundary search).
        """

    def goto_critical(self, x: float, y: float, z: float, r: float) -> None:
        """CRITICAL point that must be reached EXACTLY (pick-final / place-final
        descents). Raises if genuinely unreachable.
        """

    def goto_exact(self, x: float, y: float, z: float, r: float) -> None:
        """Plain exact move with no clamping (contact self-centering nudges)."""

    def grasp(self) -> dict:
        """Acquire the object (gripper close / suction on). Returns an ``ok`` dict."""

    def release(self) -> dict:
        """Release the object (gripper open / suction off). Returns an ``ok`` dict."""

    def z_min_safe(self) -> float | None:
        """Lowest safe TIP-frame z the body advertises, or None if unknown."""


def _api_z_min_safe(api: Any) -> float | None:
    """Lowest safe TIP-frame z from ``api.env.z_min_safe``, or None if unset."""
    env = getattr(api, "env", None)
    z_min = getattr(env, "z_min_safe", None) if env is not None else None
    if z_min is None:
        return None
    try:
        out = float(z_min)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _clamp_radius(x: float, y: float, max_r: float) -> tuple[float, float, bool]:
    """Scale (x, y) inward along its base-frame azimuth so its radius does not
    exceed ``max_r`` (z / orientation unaffected). ``max_r <= 0`` disables the
    cap. Returns ``(x, y, clamped)``.
    """
    if max_r <= 0.0:
        return x, y, False
    r = math.hypot(x, y)
    if r <= max_r or r == 0.0:
        return x, y, False
    s = max_r / r
    return x * s, y * s, True


class GripperStrategy:
    """Default 6-DoF + parallel-gripper strategy.

    Args:
      api: the robot api (needs ``goto_xyzr`` and ``open_gripper``/``close_gripper``;
        optionally ``goto_xyzr_joint`` for joint-space arcs).
      max_reach_radius_mm: soft reach cap for PROCESS waypoints only (0 = off).
        When a process xy radius exceeds it, the point is scaled inward along its
        azimuth so the arm gets "as close as reachable along the direction"
        instead of tripping a reach-edge alarm. Critical points ignore this.
      safe_travel_z_min_mm: minimum TIP-frame z for PROCESS waypoints (0 = off).
        Only the EXEMPT vertical pick/place strokes dip below it.
      error_clearer: optional callable to clear latched controller errors
        (e.g. ``lambda: env.clear_errors()``).  When ``None``, error recovery
        is skipped and the strategy re-raises the original exception.
    """

    def __init__(
        self,
        api: Any,
        *,
        max_reach_radius_mm: float = 0.0,
        safe_travel_z_min_mm: float = 0.0,
        error_clearer: Callable[[], None] | None = None,
    ) -> None:
        """Store the api reference and motion guard constants."""
        self._api = api
        self._max_reach_radius_mm = float(max_reach_radius_mm)
        self._safe_travel_z_min_mm = float(safe_travel_z_min_mm)
        self._error_clearer = error_clearer

    # ----------------------------------------------------------------- grasp
    def grasp(self) -> dict:
        """Acquire the object by closing the parallel gripper."""
        return self._api.close_gripper()

    def release(self) -> dict:
        """Release the object by opening the parallel gripper."""
        return self._api.open_gripper()

    def z_min_safe(self) -> float | None:
        """Lowest safe TIP-frame z advertised by the body, or None."""
        return _api_z_min_safe(self._api)

    # ----------------------------------------------------------------- motion
    def goto_exact(self, x: float, y: float, z: float, r: float) -> None:
        """Plain exact move with no clamping (contact self-centering nudges)."""
        self._api.goto_xyzr(x=x, y=y, z=z, r=r)

    def goto_transit(self, x: float, y: float, z: float, r: float) -> None:
        """Prefer joint-space MovJ (PTP): it arcs around reach-edge / wrist-
        singularity regions where a straight Cartesian MovL alarms. Falls back to
        ``goto_xyzr`` when the api has no joint twin (e.g. MockArmEnv).
        """
        mover = getattr(self._api, "goto_xyzr_joint", None) or self._api.goto_xyzr
        mover(x=x, y=y, z=z, r=r)

    def goto_critical(self, x: float, y: float, z: float, r: float) -> None:
        """Prefer a straight Cartesian MovL. If the hover above was radius-clamped,
        the descent is a long diagonal line that can graze a singularity and alarm
        even though the END point is reachable; on that alarm clear and retry as
        joint-space MovJ. Re-raises if MovJ also fails (genuinely out of reach).
        """
        try:
            self._api.goto_xyzr(x=x, y=y, z=z, r=r)
            return
        except Exception as exc:  # noqa: BLE001 - straight-line singularity; endpoint may still be reachable
            self._clear_errors()
            mover = getattr(self._api, "goto_xyzr_joint", None)
            if mover is None:
                raise
            logger.warning(
                "[slot_pick] straight MovL to critical point (%.1f, %.1f, %.1f) "
                "tripped %s; retrying via joint-space MovJ.",
                x,
                y,
                z,
                exc,
            )
            mover(x=x, y=y, z=z, r=r)

    def goto_process(self, x: float, y: float, z: float, r: float) -> None:
        """Best-effort move to a PROCESS waypoint — never raises. Raises z to the
        safe floor, pre-clamps the xy radius, MovJ there; if the controller still
        alarms, clear and binary-search from the current pose toward the (clamped)
        target, parking at the farthest reachable point.
        """
        safe_z_min_mm = self._safe_travel_z_min_mm
        if safe_z_min_mm > 0.0:
            z = max(z, safe_z_min_mm)
        cx, cy, _ = _clamp_radius(x, y, self._max_reach_radius_mm)
        mover = getattr(self._api, "goto_xyzr_joint", None) or self._api.goto_xyzr
        try:
            mover(x=cx, y=cy, z=z, r=r)
            return
        except Exception as exc:  # noqa: BLE001 - best-effort: clamp toward reachable
            logger.warning(
                "[slot_pick] process waypoint (%.1f, %.1f, %.1f) unreachable "
                "(%s); clamping toward the reachable boundary.",
                cx,
                cy,
                z,
                exc,
            )
            self._clear_errors()
            try:
                cur = self._api.get_pose()
                ax = float(cur["x"])
                ay = float(cur["y"])
                az = float(cur.get("z", z))
            except Exception:  # noqa: BLE001 - no readback → give up quietly
                return
            # Binary-search t in [0, 1] for the farthest reachable point along the
            # segment current -> (clamped) target; end parked at that point.
            lo, hi = 0.0, 1.0
            best: tuple[float, float, float] | None = None
            for _ in range(6):
                t = (lo + hi) / 2.0
                tx = ax + t * (cx - ax)
                ty = ay + t * (cy - ay)
                tz = az + t * (z - az)
                try:
                    mover(x=tx, y=ty, z=tz, r=r)
                    best = (tx, ty, tz)
                    lo = t
                except Exception:  # noqa: BLE001
                    self._clear_errors()
                    hi = t
            if best is not None:
                try:
                    mover(x=best[0], y=best[1], z=best[2], r=r)
                except Exception:  # noqa: BLE001
                    self._clear_errors()

    # ---------------------------------------------------------------- helpers
    def _clear_errors(self) -> None:
        """Clear any latched error state via the injected callback, if available."""
        if self._error_clearer is not None:
            try:
                self._error_clearer()
            except Exception:  # noqa: BLE001
                pass
