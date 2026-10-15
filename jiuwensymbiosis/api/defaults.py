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

Inheritance would bundle: taking ``goto_xyzr`` would mean taking ``get_pose``,
``get_home_pose`` and ``move_direction`` too, and the MRO decided which of two
mixins won when both defined a name. A function has neither problem, and — the
point of the shape — **the adapter file lists every action the body offers**,
instead of that list being spread across a base-class tuple.

Each function takes the api object first (``api``) and reads ``api.env``.
Adapter-specific geometry can be supplied explicitly as an implementation-only
callback; it does not change the action's public signature.
"""

from __future__ import annotations

import math
from collections.abc import Callable
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
    a tool offset overrides this rather than calling it.
    """
    return pose_to_dict(api.env.get_flange_pose())


def get_home_pose(api: Any) -> dict:
    """Home pose constants, read from the env."""
    return pose_to_dict(api.env.home_pose)


def goto_xyzr(
    api: Any,
    x: float,
    y: float,
    z: float,
    r: float | None = None,
    *,
    orientation_policy: str = "top_down",
) -> None:
    """Move to an absolute Cartesian pose (tip == flange), tilt chosen by ``orientation_policy``.

    ``top_down`` commands rx=180 / ry=0; ``preserve`` keeps the tilt the arm is already in, so
    the move is a pure translation. ``top_down`` is the default here only because it is what
    this function has always done — the contract does not promise it, and a body states its own
    default by what it passes.

    A body with a tool offset or a tilted tool writes its own instead of calling this: the
    flange conversion then depends on the resolved tilt, which this tip==flange form has not got.
    """
    # Read the live pose only when something actually needs it — an explicit r with top_down
    # needs no hardware round-trip, which is the common path.
    cur = api.env.get_flange_pose() if (r is None or orientation_policy == "preserve") else None
    if r is None:
        r = getattr(cur, "rz", getattr(cur, "r", 0.0))
    if orientation_policy == "top_down":
        rx, ry = 180.0, 0.0
    elif orientation_policy == "preserve":
        rx = float(getattr(cur, "rx", 180.0))
        ry = float(getattr(cur, "ry", 0.0))
    else:
        raise ValueError(
            f"goto_xyzr: orientation_policy must be 'top_down' or 'preserve', got "
            f"{orientation_policy!r}. A calibrated 'grasp' tilt is a per-body policy."
        )
    api.env.move_to_flange(SimpleNamespace(x=float(x), y=float(y), z=float(z), rx=rx, ry=ry, rz=float(r)))


def move_direction(
    api: Any,
    direction: str,
    distance_mm: float,
    *,
    flange_to_tip: Callable[[Any], dict[str, float]] | None = None,
) -> dict:
    """Relative Cartesian nudge with a self-contained bounds check.

    ``move_direction`` is NOT in SafetyRail's watch set (the rail only sees
    ``goto_xyzr`` at the tool layer), so it validates the target against
    ``env.z_min_safe`` / ``env.workspace_bounds`` here and raises ``ValueError`` on a
    violation — the agent sees that and self-corrects. Translation preserves
    the full flange orientation. A body with a tool offset supplies
    ``flange_to_tip`` so absolute bounds and returned XYZ use TIP coordinates.
    The target and its TIP are derived from a single live flange sample.
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
    target = SimpleNamespace(
        x=tx,
        y=ty,
        z=tz,
        rx=getattr(cur, "rx", 180.0),
        ry=getattr(cur, "ry", 0.0),
        rz=getattr(cur, "rz", getattr(cur, "r", 0.0)),
    )
    reported = flange_to_tip(target) if flange_to_tip is not None else pose_to_dict(target)
    tip_xyz = {axis: reported[axis] for axis in ("x", "y", "z")}
    _check_translation_bounds(api.env, **tip_xyz)
    api.env.move_to_flange(target)
    return {"ok": True, "direction": key, "distance_mm": dist, "pose": tip_xyz}


# =============================================================================
# Joint motion
# =============================================================================
def move_joint(api: Any, targets: dict[str, float]) -> Any:
    """Move the named joints to absolute positions (delegates to the Env verb)."""
    return api.env.move_joint(targets)


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
    a body with real width control writes its own.
    """
    del width_mm
    api.env.set_end_effector(False)
    return {"ok": True, "state": "open"}


def close_gripper(api: Any, force_n: float | None = None) -> dict:
    """Close the gripper. ``force_n`` is accepted for contract parity and ignored —
    a body with real force control writes its own.
    """
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
    exactly the guarantee ``rotate_base`` makes over ``navigate_relative``.
    """
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


# =============================================================================
# Approach — motion.goal / vision.search
# =============================================================================
# One hop, like every other action here: the implementation and its algorithm both live in
# ``motion/approach.py``. They used to reach it through an ``Approach`` component the adapter
# held, which made these three the only actions in the vocabulary whose implementation path
# differed from the other twenty-seven.
def search_target(api: Any, object_name: str = "box", reference: str | None = None, relation: str = "on") -> dict:
    """Look through every camera at the current heading; report the first bearing found."""
    from jiuwensymbiosis.motion import approach

    return approach.search_target(api, object_name, reference, relation)


def approach_for_grasp(api: Any, object_name: str = "box", reference: str | None = None, relation: str = "on") -> dict:
    """Find the target, face it, drive square to its face at the work distance."""
    from jiuwensymbiosis.motion import approach

    return approach.approach_target_for_grasp(api, object_name, reference, relation)


def approach_for_place(
    api: Any, object_name: str = "table", reference: str | None = None, relation: str = "on"
) -> dict:
    """Find the surface, face it, drive to its near edge at placing distance."""
    from jiuwensymbiosis.motion import approach

    return approach.approach_target_for_place(api, object_name, reference, relation)


# =============================================================================
# 3-D scene sensing — vision.detection
# =============================================================================
# Same one hop as everything else: the pipeline and its orchestration both live in
# ``perception/scene3d.py``. These three used to reach it through a ``Scene3D`` component
# the adapter held.
def locate_for_grasp(api: Any, object_name: str = "box", reference: str | None = None, relation: str = "on") -> dict:
    """Measure a target and return what PICKING IT UP needs, in the base frame."""
    from jiuwensymbiosis.perception import scene3d

    return scene3d.locate_for_grasp(api, object_name, reference, relation)


def locate_for_place(api: Any, object_name: str = "table", reference: str | None = None, relation: str = "on") -> dict:
    """Measure a support surface and return what PUTTING SOMETHING ON IT needs."""
    from jiuwensymbiosis.perception import scene3d

    return scene3d.locate_for_place(api, object_name, reference, relation)


def analyze_scene(api: Any, object_name: str = "box") -> dict:
    """Every instance of ``object_name`` in the current frame, with base-frame geometry."""
    from jiuwensymbiosis.perception import scene3d

    return scene3d.analyze_scene(api, object_name)
