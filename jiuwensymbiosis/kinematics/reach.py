# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Framework-generic URDF reachability for planning — body-agnostic.

Given a single URDF arm ``Chain``, judge whether its end effector can reach a base-frame point, and
estimate a coarse reachable envelope. Reused by ``Reachability`` so any body that exposes a URDF
gets a planning-time reach prior; bodies without a URDF never call this. Uses the numpy 5-DoF DLS
(``ik.ik_solve_5dof``) so it works without pinocchio — a coarse workspace judge, not a grasp solve.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from jiuwensymbiosis.kinematics.ik import ik_solve_5dof
from jiuwensymbiosis.kinematics.urdf_chain import Chain


def reachable_point(
    chain: Chain,
    target_xyz_mm: Any,
    q_current: dict[str, float] | None = None,
    *,
    q_fixed: dict[str, float] | None = None,
    pos_tol_m: float = 0.03,
    tool_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> bool:
    """Can this arm chain's end effector reach the base-frame point ``target_xyz_mm`` (mm)?

    Position-focused, loose orientation: the tool axis is only asked to point roughly from the target
    back toward the robot (the typical approach direction), so this is a coarse "is the point in the
    arm's workspace" judge for planning — NOT a grasp-pose solve. Returns the IK ``converged`` flag.
    ``q_current`` supplies the current joint angles (arm joints seed the solve, the rest are held).
    """
    q_current = {str(k): float(v) for k, v in (q_current or {}).items()}
    arm_joints = chain.movable_names()
    q_init = {j: q_current.get(j, 0.0) for j in arm_joints}
    q_fixed = q_current if q_fixed is None else {str(k): float(v) for k, v in q_fixed.items()}
    tx, ty, tz = (float(target_xyz_mm[0]) / 1000.0, float(target_xyz_mm[1]) / 1000.0,
                  float(target_xyz_mm[2]) / 1000.0)
    rng = math.hypot(tx, ty) or 1.0
    target_normal = np.array([-tx / rng, -ty / rng, 0.0])  # horizontal, pointing target → robot
    res = ik_solve_5dof(chain, q_fixed, arm_joints, np.array([tx, ty, tz]), target_normal,
                        tool_normal_local=tool_axis_local, q_init=q_init, pos_tol_m=pos_tol_m)
    return bool(res.converged)


def reach_envelope(
    chain: Chain,
    q_current: dict[str, float] | None = None,
    *,
    q_fixed: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Coarse reachable-workspace envelope of one arm chain at the current pose: probe a small grid of
    base-frame points with ``reachable_point`` and report the reachable forward/lateral/height extent
    (metres). A rough prior for planning when no target is detected — not an exact workspace.
    """
    fwd = (0.35, 0.6, 0.85)
    lat = (-0.35, 0.0, 0.35)
    hgt = (0.55, 0.95)
    reached = []
    for f in fwd:
        for la in lat:
            for h in hgt:
                if reachable_point(chain, (f * 1000.0, la * 1000.0, h * 1000.0),
                                   q_current, q_fixed=q_fixed):
                    reached.append((f, la, h))
    if not reached:
        return {"reachable": False}
    fs = [p[0] for p in reached]
    ls = [p[1] for p in reached]
    hs = [p[2] for p in reached]
    return {"reachable": True,
            "forward_m": [min(fs), max(fs)],
            "lateral_m": [min(ls), max(ls)],
            "height_m": [min(hs), max(hs)]}
