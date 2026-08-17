# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Reusable action implementations, as plain functions.

Most actions have an implementation that works on any body meeting the Env
contract: ``goto_xyzr`` is ``env.move_to_flange(...)``, ``navigate_relative`` is
``env.navigate_relative(...)``. Those live here so a new adapter does not rewrite
them — but as **functions it calls**, not as a base class it inherits:

    class PiperApi(BaseRobotApi):
        @implements(MOVE_DIRECTION)
        def move_direction(self, direction: str, distance_mm: float) -> dict:
            return defaults.move_direction(self, direction, distance_mm)

Inheritance would bundle: taking ``goto_xyzr`` used to mean taking ``get_pose``,
``get_home_pose`` and ``move_direction`` too, and the MRO decided which of two
mixins won when both defined a name. A function has neither problem, and — the
point of the shape — **the adapter file lists every action the body offers**,
instead of that list being spread across a base-class tuple.

Each function takes the api object first (``api``), reads ``api.env`` and nothing
else about the body, and matches its ``ActionSpec`` signature exactly.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any


def pose_to_dict(pose: Any) -> dict:
    """Best-effort vendor-pose object → dict.

    Tolerates both the SCARA convention (``.r``) and the 6-DoF convention
    (``.rx/.ry/.rz``); only the fields the pose actually exposes are emitted.
    """
    out: dict[str, float] = {}
    for key in ("x", "y", "z", "rx", "ry", "rz", "r"):
        val = getattr(pose, key, None)
        if val is not None:
            out[key] = float(val)
    return out


# Base-frame unit offsets per direction word (arm faces +x; +y is left; +z up).
_DIRECTION_OFFSETS: dict[str, tuple[float, float, float]] = {
    "forward": (1.0, 0.0, 0.0),
    "back": (-1.0, 0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
    "up": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
}


def _check_translation_bounds(env: Any, x: float, y: float, z: float) -> None:
    """Reject a target below ``env.z_min_safe`` or outside ``env.workspace_bounds``."""
    if not all(math.isfinite(value) for value in (x, y, z)):
        raise ValueError(f"move blocked: target coordinates must be finite, got x={x}, y={y}, z={z}")
    z_floor = getattr(env, "z_min_safe", None)
    if z_floor is not None and z < float(z_floor):
        raise ValueError(f"move blocked: z={z:.1f} below z_min_safe={float(z_floor):.1f}")
    bounds = getattr(env, "workspace_bounds", None)
    if bounds is not None:
        xmin, ymin, xmax, ymax = bounds
        if not xmin <= x <= xmax:
            raise ValueError(f"move blocked: x={x:.1f} out of [{xmin}, {xmax}]")
        if not ymin <= y <= ymax:
            raise ValueError(f"move blocked: y={y:.1f} out of [{ymin}, {ymax}]")


# =============================================================================
# Posture / Cartesian motion
# =============================================================================
def home(api: Any) -> None:
    """Return to the home pose (delegates to the Env verb)."""
    api.env.home()


def get_pose(api: Any) -> dict:
    """Current pose. Reports the flange pose — assumes tip == flange, so a body with
    a tool offset overrides this rather than calling it."""
    return pose_to_dict(api.env.get_flange_pose())


def get_home_pose(api: Any) -> dict:
    """Home pose constants, read from the env."""
    return pose_to_dict(api.env.home_pose)


def goto_xyzr(api: Any, x: float, y: float, z: float, r: float | None = None) -> None:
    """Top-down move to an absolute Cartesian pose (tip == flange, rx=180, ry=0).

    A body with a tool offset or a tilted tool writes its own instead of calling this.
    """
    if r is None:
        cur = api.env.get_flange_pose()
        r = getattr(cur, "rz", getattr(cur, "r", 0.0))
    api.env.move_to_flange(SimpleNamespace(x=float(x), y=float(y), z=float(z), rx=180.0, ry=0.0, rz=float(r)))


def move_direction(api: Any, direction: str, distance_mm: float) -> dict:
    """Relative Cartesian nudge with a self-contained bounds check.

    ``move_direction`` is NOT in SafetyRail's watch set (the rail only sees
    ``goto_xyzr`` at the tool layer), so it validates the target against
    ``env.z_min_safe`` / ``env.workspace_bounds`` here and raises ``ValueError`` on a
    violation — the agent sees that and self-corrects. A relative translation is
    frame-agnostic, so a tool offset does not matter.
    """
    key = (direction or "").strip().lower()
    if key not in _DIRECTION_OFFSETS:
        raise ValueError(f"unknown direction {direction!r}; expected one of {sorted(_DIRECTION_OFFSETS)}")
    dist = float(distance_mm)
    if not math.isfinite(dist) or dist <= 0:
        raise ValueError("distance_mm must be finite and positive; the direction controls the sign")
    ux, uy, uz = _DIRECTION_OFFSETS[key]
    cur = api.env.get_flange_pose()
    tx, ty, tz = cur.x + ux * dist, cur.y + uy * dist, cur.z + uz * dist
    _check_translation_bounds(api.env, tx, ty, tz)
    target = SimpleNamespace(
        x=tx,
        y=ty,
        z=tz,
        rx=getattr(cur, "rx", 180.0),
        ry=getattr(cur, "ry", 0.0),
        rz=getattr(cur, "rz", getattr(cur, "r", 0.0)),
    )
    api.env.move_to_flange(target)
    return {"ok": True, "direction": key, "distance_mm": dist, "pose": {"x": tx, "y": ty, "z": tz}}


# =============================================================================
# Joint motion
# =============================================================================
def move_joint(api: Any, q: list[float]) -> None:
    """Move to a joint configuration (delegates to the Env verb)."""
    api.env.move_joint(q)


# =============================================================================
# End effector
# =============================================================================
def activate_suction(api: Any) -> dict:
    """Turn suction ON (delegates to the Env end-effector verb)."""
    api.env.set_end_effector(True)
    return {"ok": True, "state": "on"}


def deactivate_suction(api: Any) -> dict:
    """Turn suction OFF (delegates to the Env end-effector verb)."""
    api.env.set_end_effector(False)
    return {"ok": True, "state": "off"}


def open_gripper(api: Any, width_mm: float = 80.0) -> dict:
    """Open the gripper. ``width_mm`` is accepted for contract parity and ignored —
    a body with real width control writes its own."""
    del width_mm
    api.env.set_end_effector(False)
    return {"ok": True, "state": "open"}


def close_gripper(api: Any, force_n: float | None = None) -> dict:
    """Close the gripper. ``force_n`` is accepted for contract parity and ignored —
    a body with real force control writes its own."""
    del force_n
    api.env.set_end_effector(True)
    return {"ok": True, "state": "closed"}


# =============================================================================
# Camera
# =============================================================================
def get_image(api: Any) -> Any:
    """Latest RGB frame, or None when the body has no camera (delegates to the env)."""
    return api.env.grab_rgb()


# =============================================================================
# Mobile base / torso
# =============================================================================
def navigate_relative(api: Any, dx_m: float, dy_m: float = 0.0, dyaw_rad: float = 0.0) -> dict:
    """Relative base move (delegates to the Env verb)."""
    return api.env.navigate_relative(float(dx_m), float(dy_m), float(dyaw_rad))


def rotate_base(api: Any, dyaw_rad: float) -> dict:
    """Turn in place — the same Env verb with the translation pinned to zero, which is
    exactly the guarantee ``rotate_base`` makes over ``navigate_relative``."""
    return api.env.navigate_relative(0.0, 0.0, float(dyaw_rad))


def drive_arc(api: Any, radius_m: float, dyaw_rad: float) -> dict:
    """One constant-curvature arc (delegates to the Env verb)."""
    return api.env.navigate_arc(float(radius_m), float(dyaw_rad))


def set_lift_pose(api: Any, q_lifter: dict) -> dict:
    """Set lifter joints (delegates to the Env verb)."""
    return api.env.set_lifter(q_lifter)


def turn_waist(api: Any, delta_rad: float) -> dict:
    """Rotate waist (delegates to the Env verb)."""
    return api.env.turn_waist(float(delta_rad))
