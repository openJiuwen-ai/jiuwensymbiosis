# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""5-DoF 阻尼最小二乘 (DLS) 数值逆运动学。

任务 = 位置(3) + 掌法向(3, rank-2)。雅可比用有限差分。冗余(7 关节)下用零空间项
把关节推向限位中点。所有长度单位为米。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jiuwensymbiosis.kinematics.fk import fk_chain
from jiuwensymbiosis.kinematics.urdf_chain import Chain


@dataclass
class IKResult:
    q: dict[str, float]
    converged: bool
    pos_err_m: float
    normal_err: float
    iters: int


def tool_normal_base(chain: Chain, q: dict[str, float], tool_normal_local) -> np.ndarray:
    """Tool palm-normal expressed in the base frame for joint config ``q``."""
    t = fk_chain(chain, q)
    n = t[:3, :3] @ np.asarray(tool_normal_local, dtype=np.float64)
    return n / max(np.linalg.norm(n), 1e-12)


def _task(chain: Chain, q: dict[str, float], tool_normal_local) -> np.ndarray:
    """Forward task map g(q) = [position(3); palm_normal(3)] in base frame."""
    t = fk_chain(chain, q)
    p = t[:3, 3]
    n = t[:3, :3] @ np.asarray(tool_normal_local, dtype=np.float64)
    n = n / max(np.linalg.norm(n), 1e-12)
    return np.concatenate([p, n])


def ik_solve_5dof(
    chain: Chain,
    q_fixed: dict[str, float],
    arm_joints: list[str],
    target_pos_m: np.ndarray,
    target_normal: np.ndarray,
    *,
    tool_normal_local=(0.0, 0.0, 1.0),
    q_init: dict[str, float] | None = None,
    max_iters: int = 200,
    damping: float = 0.05,
    pos_tol_m: float = 0.005,
    normal_tol: float = 0.02,
    nullspace_gain: float = 0.0,
    step_clip: float = 0.2,
) -> IKResult:
    """Solve 7-DoF arm for a 5-DoF (position + palm-normal) task via DLS."""
    limits = chain.limits()
    target_pos = np.asarray(target_pos_m, dtype=np.float64)
    target_n = np.asarray(target_normal, dtype=np.float64)
    target_n = target_n / max(np.linalg.norm(target_n), 1e-12)

    q = dict(q_fixed)
    for name in arm_joints:
        if q_init and name in q_init:
            q[name] = float(q_init[name])
        else:
            lo, hi = limits.get(name, (-1.0, 1.0))
            q[name] = 0.5 * (lo + hi)

    q_mid = {n: 0.5 * (limits[n][0] + limits[n][1]) for n in arm_joints}
    q_rng = {n: max(limits[n][1] - limits[n][0], 1e-6) for n in arm_joints}
    eps = 1e-6
    pos_err = normal_err = float("inf")
    it = 0
    for it in range(1, max_iters + 1):
        g = _task(chain, q, tool_normal_local)
        p, n = g[:3], g[3:]
        e = np.concatenate([target_pos - p, target_n - n])  # (6,)
        pos_err = float(np.linalg.norm(target_pos - p))
        normal_err = float(np.linalg.norm(target_n - n))
        if pos_err < pos_tol_m and normal_err < normal_tol:
            break

        # finite-difference Jacobian (6 x 7) over arm joints
        jac = np.zeros((6, len(arm_joints)))
        for c, name in enumerate(arm_joints):
            q[name] += eps
            jac[:, c] = (_task(chain, q, tool_normal_local) - g) / eps
            q[name] -= eps

        # damped least squares: dq = Jt (J Jt + l^2 I)^-1 e
        jjt = jac @ jac.T + (damping ** 2) * np.eye(6)
        dq = jac.T @ np.linalg.solve(jjt, e)

        if nullspace_gain > 0.0:
            j_pinv = jac.T @ np.linalg.solve(jjt, np.eye(6))
            null = np.eye(len(arm_joints)) - j_pinv @ jac
            grad = np.array([-(q[name] - q_mid[name]) / (q_rng[name] ** 2) for name in arm_joints])
            dq = dq + null @ (nullspace_gain * grad)

        dq = np.clip(dq, -step_clip, step_clip)
        for c, name in enumerate(arm_joints):
            lo, hi = limits[name]
            q[name] = float(np.clip(q[name] + dq[c], lo, hi))

    converged = pos_err < pos_tol_m and normal_err < normal_tol
    return IKResult(q={n: q[n] for n in arm_joints}, converged=converged,
                    pos_err_m=pos_err, normal_err=normal_err, iters=it)


def ik_solve_pose(
    chain: Chain,
    q_fixed: dict[str, float],
    arm_joints: list[str],
    target_pos_m: np.ndarray,
    approach_target: np.ndarray,
    paddle_target: np.ndarray,
    *,
    tool_approach_local=(0.0, 0.0, 1.0),
    tool_paddle_local=(1.0, 0.0, 0.0),
    tcp_offset_local=(0.0, 0.0, 0.0),
    q_init: dict[str, float] | None = None,
    max_iters: int = 1500,
    damping: float = 0.06,
    pos_tol_m: float = 0.012,
    orient_tol: float = 0.06,
    step_clip: float = 0.12,
) -> IKResult:
    """Full-orientation DLS IK: place a tool TCP and align TWO tool axes.

    Unlike :func:`ik_solve_5dof` (one axis), this constrains the tool's
    ``tool_approach_local`` axis to ``approach_target`` AND its
    ``tool_paddle_local`` axis to ``paddle_target`` (base directions), fully
    fixing orientation. The position target is the TCP — a point offset from the
    chain's leaf by ``tcp_offset_local`` (tool frame, metres) — so a paddle whose
    contact face is ahead of the wrist lands on target, not the wrist.

    Task error (9-vector) = [TCP pos err; approach-axis err; paddle-axis err],
    solved with a damped least-squares step and a finite-difference Jacobian.
    ``normal_err`` in the result holds the combined orientation error norm.
    """
    limits = chain.limits()
    target_pos = np.asarray(target_pos_m, dtype=np.float64)
    a_tgt = np.asarray(approach_target, dtype=np.float64)
    a_tgt = a_tgt / max(np.linalg.norm(a_tgt), 1e-12)
    p_tgt = np.asarray(paddle_target, dtype=np.float64)
    p_tgt = p_tgt / max(np.linalg.norm(p_tgt), 1e-12)
    a_loc = np.asarray(tool_approach_local, dtype=np.float64)
    p_loc = np.asarray(tool_paddle_local, dtype=np.float64)
    v_tcp = np.asarray(tcp_offset_local, dtype=np.float64)

    q = dict(q_fixed)
    for name in arm_joints:
        if q_init and name in q_init:
            q[name] = float(q_init[name])
        else:
            lo, hi = limits.get(name, (-1.0, 1.0))
            q[name] = 0.5 * (lo + hi)

    def _err(qq: dict[str, float]) -> np.ndarray:
        t = fk_chain(chain, qq)
        rot = t[:3, :3]
        tcp = t[:3, 3] + rot @ v_tcp
        return np.concatenate([
            target_pos - tcp,
            a_tgt - rot @ a_loc,
            p_tgt - rot @ p_loc,
        ])

    eps = 1e-6
    pos_err = orient_err = float("inf")
    best_total = float("inf")
    stall = 0
    it = 0
    for it in range(1, max_iters + 1):
        e = _err(q)
        pos_err = float(np.linalg.norm(e[:3]))
        orient_err = float(np.linalg.norm(e[3:]))
        if pos_err < pos_tol_m and orient_err < orient_tol:
            break
        # Early-stop: if the total error stops improving, the target is out of
        # reach for this config — bail instead of grinding all max_iters. Keeps a
        # full-grid lifter search (many hopeless poses) fast.
        total = pos_err + orient_err
        if total < best_total - 1e-5:
            best_total, stall = total, 0
        else:
            stall += 1
            if stall >= 60:
                break
        jac = np.zeros((9, len(arm_joints)))
        for c, name in enumerate(arm_joints):
            q[name] += eps
            jac[:, c] = (_err(q) - e) / eps
            q[name] -= eps
        jjt = jac @ jac.T + (damping ** 2) * np.eye(9)
        # jac is d(error)/dq (error = target - forward_map), so the DLS step that
        # reduces the error is NEGATIVE jacᵀ(jjt)⁻¹ e.
        dq = -jac.T @ np.linalg.solve(jjt, e)
        dq = np.clip(dq, -step_clip, step_clip)
        for c, name in enumerate(arm_joints):
            lo, hi = limits[name]
            q[name] = float(np.clip(q[name] + dq[c], lo, hi))

    converged = pos_err < pos_tol_m and orient_err < orient_tol
    return IKResult(q={n: q[n] for n in arm_joints}, converged=converged,
                    pos_err_m=pos_err, normal_err=orient_err, iters=it)
