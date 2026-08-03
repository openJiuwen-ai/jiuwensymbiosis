# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""FK/IK backend seam + pure motion planning for locally-solved arms.

An arm whose SDK takes only joint targets needs its Cartesian layer built above
the SDK. This module owns the reusable, hardware-free half of that layer:

  * ``FkIkBackend`` — the injection seam for a kinematics library (a URDF solver
    such as ikpy / pinocchio / a vendor kinematics class). The framework never
    imports one; the adapter wraps its own behind this Protocol.
  * ``MotionLimits`` + ``plan_*`` — pure functions that turn a target into a
    validated list of joint waypoints, **rejecting** an unreachable / out-of-
    limit / discontinuous target before any command is sent. No I/O, no clock,
    no hardware — unit-testable with a fake backend.

SE(3) convention: 4x4 homogeneous matrices with translation in **metres**
(see ``_common.geometry``). Joint values are in the backend's native unit
(degrees for most; the adapter keeps it consistent).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from jiuwensymbiosis.adapters._common.geometry import orientation_error_deg, position_error_mm
from jiuwensymbiosis.utils.geometry import make_transform


@runtime_checkable
class FkIkBackend(Protocol):
    """Forward / inverse kinematics over the arm joints (no gripper).

    Wraps a URDF solver. ``forward_kinematics`` maps a joint vector to a 4x4
    SE(3) pose (metres) of the control frame; ``inverse_kinematics`` maps a
    desired SE(3) pose to a joint vector, seeded by ``current`` so successive
    waypoints stay on the same branch. ``orientation_weight`` lets an
    underactuated arm trade orientation for reachable position (soft posture).
    """

    def forward_kinematics(self, q: list[float]) -> np.ndarray:
        """Joint vector → 4x4 SE(3) control-frame pose (translation in metres)."""

    def inverse_kinematics(self, current: list[float], target: np.ndarray, *, orientation_weight: float) -> list[float]:
        """Desired 4x4 SE(3) pose → joint vector, seeded by ``current``."""


@dataclass(frozen=True, slots=True)
class MotionLimits:
    """Planning bounds and tolerances, ordered to match the arm joints.

    Roles are deliberately separate:
      * ``max_joint_step_deg`` — granularity of joint-space interpolation.
      * ``max_cartesian_step_mm`` — translation granularity of a Cartesian path.
      * ``max_ik_jump_deg`` — per-waypoint joint-jump ceiling that flags an IK
        branch flip / near-singularity (looser than ``max_joint_step_deg``,
        which curvature alone can exceed on a coarse Cartesian step).
    """

    joint_names: tuple[str, ...]
    joint_limits: tuple[tuple[float, float] | None, ...]
    max_joint_step_deg: float = 2.0
    max_cartesian_step_mm: float = 5.0
    max_ik_jump_deg: float = 30.0
    ik_position_tolerance_mm: float = 3.0
    ik_orientation_tolerance_deg: float | None = None
    ik_orientation_weight: float = 0.01

    def __post_init__(self) -> None:
        if len(self.joint_limits) != len(self.joint_names):
            raise ValueError(
                f"joint_limits has {len(self.joint_limits)} entries but there are {len(self.joint_names)} joints"
            )


def _check_length(q: list[float], limits: MotionLimits) -> None:
    if len(q) != len(limits.joint_names):
        raise ValueError(f"expected {len(limits.joint_names)} joints, got {len(q)}")


def _check_finite(q: list[float], limits: MotionLimits) -> None:
    for name, value in zip(limits.joint_names, q, strict=True):
        if not math.isfinite(value):
            raise ValueError(f"non-finite target for joint '{name}': {value}")


def _check_limits(q: list[float], limits: MotionLimits) -> None:
    for name, value, lim in zip(limits.joint_names, q, limits.joint_limits, strict=True):
        if lim is None:
            continue
        low, high = lim
        if not low <= value <= high:
            raise ValueError(f"joint '{name}'={value:.3f} out of soft limit [{low}, {high}]")


def interpolate_joint_waypoints(current: list[float], target: list[float], max_step_deg: float) -> list[list[float]]:
    """Linear joint interpolation from ``current`` to ``target``.

    Returns waypoints excluding the start and including the target, each no more
    than ``max_step_deg`` from the previous on any joint.
    """
    if len(current) != len(target):
        raise ValueError(f"current/target length mismatch: {len(current)} vs {len(target)}")
    cur = [float(v) for v in current]
    tgt = [float(v) for v in target]
    max_delta = max((abs(t - c) for c, t in zip(cur, tgt, strict=True)), default=0.0)
    steps = max(1, math.ceil(max_delta / max_step_deg)) if max_step_deg > 0 else 1
    return [[c + (k / steps) * (t - c) for c, t in zip(cur, tgt, strict=True)] for k in range(1, steps + 1)]


def plan_joint_move(current: list[float], target: list[float], limits: MotionLimits) -> list[list[float]]:
    """Validate ``target`` and return interpolated joint waypoints, or raise.

    Rejects a non-finite / out-of-limit / wrong-length target before any motion.
    """
    _check_length(target, limits)
    _check_finite(target, limits)
    _check_limits(target, limits)
    waypoints = interpolate_joint_waypoints(current, target, limits.max_joint_step_deg)
    for wp in waypoints:
        _check_finite(wp, limits)
        _check_limits(wp, limits)
    return waypoints


def interpolate_se3(start: np.ndarray, end: np.ndarray, num_steps: int) -> list[np.ndarray]:
    """SE(3) path from ``start`` to ``end``: linear translation + slerp rotation.

    Returns ``num_steps`` matrices excluding the start and including the end.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    t0, t1 = start[:3, 3], end[:3, 3]
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([start[:3, :3], end[:3, :3]])))
    out: list[np.ndarray] = []
    for k in range(1, num_steps + 1):
        alpha = k / num_steps
        rot = slerp([alpha])[0].as_matrix()
        trans = t0 + alpha * (t1 - t0)
        out.append(make_transform(np.asarray(rot, dtype=np.float64), trans))
    return out


def interp_se3_at(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    """One SE(3) waypoint at fraction ``alpha`` in [0, 1]: linear trans + slerp rot."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([start[:3, :3], end[:3, :3]])))
    rot = slerp([alpha])[0].as_matrix()
    trans = start[:3, 3] + alpha * (end[:3, 3] - start[:3, 3])
    return make_transform(np.asarray(rot, dtype=np.float64), trans)


class CartesianServoError(ValueError):
    """Typed fail-closed rejection from the real-time Cartesian servo step.

    ``code`` is a stable machine-readable reason the real-time runner can
    propagate (e.g. ``cartesian_bounds_rejected``, ``cartesian_progress_reversed``,
    ``joint_velocity_limited_no_progress``); ``detail`` is the human message.
    """

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


@dataclass(frozen=True, slots=True)
class Waypoint:
    """One planned joint waypoint plus the residuals of its IK solution."""

    q: list[float]
    position_error_mm: float
    orientation_error_deg: float


def solve_ik_checked(
    backend: FkIkBackend,
    prev_q: list[float],
    target: np.ndarray,
    limits: MotionLimits,
    *,
    orientation_weight: float | None = None,
) -> Waypoint:
    """One IK solve toward ``target`` (4x4 SE(3), metres), seeded/bounded by ``prev_q``.

    Rejects — before any command — a solution that is non-finite, out of joint
    limits, jumps more than ``max_ik_jump_deg`` from ``prev_q`` (IK branch flip /
    near singularity), exceeds the position tolerance (unreachable), or, when
    ``ik_orientation_tolerance_deg`` is set, exceeds the orientation tolerance.
    ``orientation_weight`` overrides ``limits.ik_orientation_weight`` for this solve
    (the servo path tries a position-only weight first on an underactuated arm).
    Shared by ``plan_cartesian_move`` (per waypoint) and the servo path (one step).
    """
    weight = limits.ik_orientation_weight if orientation_weight is None else orientation_weight
    q = [float(v) for v in backend.inverse_kinematics(prev_q, target, orientation_weight=weight)]
    _check_length(q, limits)
    _check_finite(q, limits)
    _check_limits(q, limits)
    jump = max((abs(a - b) for a, b in zip(q, prev_q, strict=True)), default=0.0)
    if jump > limits.max_ik_jump_deg:
        raise ValueError(
            f"IK branch jump {jump:.1f}deg > max_ik_jump_deg={limits.max_ik_jump_deg} "
            "(near singularity or unreachable orientation); target rejected"
        )
    achieved = backend.forward_kinematics(q)
    pos_err = position_error_mm(achieved, target)
    ori_err = orientation_error_deg(achieved, target)
    if pos_err > limits.ik_position_tolerance_mm:
        raise ValueError(
            f"IK position residual {pos_err:.2f}mm > tolerance "
            f"{limits.ik_position_tolerance_mm}mm; target unreachable, rejected"
        )
    if limits.ik_orientation_tolerance_deg is not None and ori_err > limits.ik_orientation_tolerance_deg:
        raise ValueError(
            f"IK orientation residual {ori_err:.1f}deg > tolerance "
            f"{limits.ik_orientation_tolerance_deg}deg; target rejected"
        )
    return Waypoint(q=q, position_error_mm=pos_err, orientation_error_deg=ori_err)


def plan_cartesian_move(
    backend: FkIkBackend, current_q: list[float], target: Any, limits: MotionLimits
) -> list[Waypoint]:
    """Plan a Cartesian move as validated joint waypoints, or raise.

    ``target`` is a 4x4 SE(3) matrix (metres). Interpolates an SE(3) path, solves
    IK for each waypoint (seeded by the previous solution), and rejects — before
    any command — a target that is non-finite, out of joint limits, exceeds the
    position tolerance (unreachable), flips IK branch (``max_ik_jump_deg``), or,
    when ``ik_orientation_tolerance_deg`` is set, exceeds the orientation
    tolerance. Orientation residual is always recorded for the soft-posture case.
    """
    current_q = [float(v) for v in current_q]
    _check_length(current_q, limits)
    target = np.asarray(target, dtype=np.float64)
    if target.shape != (4, 4):
        raise ValueError(f"target must be a 4x4 SE(3) matrix, got shape {target.shape}")

    start = backend.forward_kinematics(current_q)
    dist_mm = position_error_mm(start, target)
    num_steps = max(1, math.ceil(dist_mm / limits.max_cartesian_step_mm)) if limits.max_cartesian_step_mm > 0 else 1

    plan: list[Waypoint] = []
    prev = current_q
    for matrix in interpolate_se3(start, target, num_steps):
        waypoint = solve_ik_checked(backend, prev, matrix, limits)
        plan.append(waypoint)
        prev = waypoint.q
    return plan
