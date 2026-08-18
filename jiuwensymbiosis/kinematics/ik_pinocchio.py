# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""pinocchio-backed arm IK: analytic-Jacobian Gauss-Newton + random restarts.

Replaces the custom DLS solver's single-seed + finite-difference weakness (which
manufactured false "unreachable"). FK is identical to ``fk_chain`` (same URDF).
"""

from __future__ import annotations

import numpy as np

from jiuwensymbiosis.kinematics.ik import IKResult

try:  # pinocchio is optional; solve_arm_ik falls back to the legacy DLS without it
    import pinocchio as pin
    _PIN_OK = True
except ImportError:  # pragma: no cover
    pin = None
    _PIN_OK = False

_MODELS: dict[str, tuple] = {}


def pin_available() -> bool:
    """True when pinocchio imported successfully."""
    return _PIN_OK


def _model(urdf_path: str):
    m = _MODELS.get(urdf_path)
    if m is None:
        model = pin.buildModelFromUrdf(urdf_path)
        _MODELS[urdf_path] = m = (model, model.createData())
    return m


def warm(urdf_path: str) -> bool:
    """Pre-build and cache the pinocchio model so the first IK solve doesn't pay
    ``buildModelFromUrdf``. No-op returning False when pinocchio isn't installed.
    """
    if not pin_available():
        return False
    _model(urdf_path)
    return True


def _orthobasis(x, y) -> np.ndarray:
    """Right-handed basis with column 0 = normalized x, column 2 ⟂ (x,y) plane."""
    x = np.asarray(x, dtype=float)
    x = x / np.linalg.norm(x)
    z = np.cross(x, np.asarray(y, dtype=float))
    z = z / np.linalg.norm(z)
    return np.column_stack([x, np.cross(z, x), z])


def _target_rotation(approach_target, paddle_target, tool_approach_local, tool_paddle_local) -> np.ndarray:
    """Rotation mapping the (paddle, approach) tool axes onto the (paddle, approach) base targets."""
    u = _orthobasis(tool_paddle_local, tool_approach_local)
    v = _orthobasis(paddle_target, approach_target)
    return v @ u.T


def solve_pose_ik_pin(
    urdf_path: str,
    arm_joints: list[str],
    leaf_link: str,
    limits: dict[str, tuple[float, float]],
    target_pos_m,
    *,
    approach_target,
    paddle_target,
    tool_approach_local,
    tool_paddle_local,
    tcp_offset_local,
    q_fixed: dict[str, float],
    q_init: dict[str, float] | None = None,
    pos_tol_m: float = 0.012,
    rot_tol: float = 0.06,
    max_iters: int = 200,
    n_restarts: int = 4,
    damping: float = 1e-6,
    step_clip: float = 0.4,
    seed: int = 0,
    check_collision: bool = False,
    package_dir: str | None = None,
) -> IKResult:
    """Full-pose arm IK for the paddle TCP via pinocchio (Gauss-Newton + random restarts).

    Optimizes only ``arm_joints`` with the rest of the body at ``q_fixed``. The
    target orientation is built from the two axis targets; the leaf-frame target
    is the TCP target minus the (rotated) ``tcp_offset_local``. Returns the converged
    solution closest to the warm start (collision-free when ``check_collision``), else
    the best attempt.
    """
    model, data = _model(urdf_path)
    fid = model.getFrameId(leaf_link)
    r_target = _target_rotation(approach_target, paddle_target, tool_approach_local, tool_paddle_local)
    leaf_pos = np.asarray(target_pos_m, dtype=float) - r_target @ np.asarray(tcp_offset_local, dtype=float)
    target_se3 = pin.SE3(r_target, leaf_pos)

    idx_q = {j: model.joints[model.getJointId(j)].idx_q for j in arm_joints}
    idx_v = [model.joints[model.getJointId(j)].idx_v for j in arm_joints]
    lo = np.array([limits[j][0] for j in arm_joints])
    hi = np.array([limits[j][1] for j in arm_joints])
    rng = np.random.default_rng(seed)

    def _base_q() -> np.ndarray:
        q = pin.neutral(model)
        for name, val in q_fixed.items():
            if model.existJointName(name):
                q[model.joints[model.getJointId(name)].idx_q] = float(val)
        return q

    q_warm = None
    if q_init:
        q_warm = np.array([float(q_init.get(j, 0.5 * (limits[j][0] + limits[j][1]))) for j in arm_joints])
    nominal = q_warm if q_warm is not None else 0.5 * (lo + hi)

    from jiuwensymbiosis.kinematics import self_collision as _sc

    gate = check_collision and _sc.available(urdf_path, package_dir)

    cands: list[tuple[float, IKResult]] = []   # (dist_to_nominal, result) for collision-free converged
    best: IKResult | None = None               # closest-error fallback (incl. converged-but-colliding)
    for restart in range(n_restarts + 1):
        q = _base_q()
        arm = q_warm.copy() if (restart == 0 and q_warm is not None) else lo + rng.random(len(arm_joints)) * (hi - lo)
        for k, j in enumerate(arm_joints):
            q[idx_q[j]] = arm[k]

        pos_err = rot_err = float("inf")
        it = 0
        for it in range(1, max_iters + 1):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacement(model, data, fid)
            i_md = data.oMf[fid].actInv(target_se3)
            err = pin.log6(i_md).vector
            pos_err = float(np.linalg.norm(err[:3]))
            rot_err = float(np.linalg.norm(err[3:]))
            if pos_err < pos_tol_m and rot_err < rot_tol:
                break
            jac = pin.computeFrameJacobian(model, data, q, fid, pin.ReferenceFrame.LOCAL)
            jac = -pin.Jlog6(i_md.inverse()) @ jac
            j_arm = jac[:, idx_v]
            step = -j_arm.T @ np.linalg.solve(j_arm @ j_arm.T + damping * np.eye(6), err)
            step = np.clip(step, -step_clip, step_clip)
            for k, j in enumerate(arm_joints):
                q[idx_q[j]] = float(np.clip(q[idx_q[j]] + step[k], lo[k], hi[k]))

        converged = pos_err < pos_tol_m and rot_err < rot_tol
        cand = IKResult(q={j: float(q[idx_q[j]]) for j in arm_joints},
                        converged=converged, pos_err_m=pos_err, normal_err=rot_err, iters=it)
        if converged and gate and _sc.in_self_collision(urdf_path, package_dir, q.copy()):
            converged = False   # self-colliding: unusable as a solution (kept only as best-effort)
            cand.converged = False   # keep the returned result honest — never report a colliding pose as converged
        if converged:
            arm_vec = np.array([q[idx_q[j]] for j in arm_joints])
            dist = float(np.linalg.norm(arm_vec - nominal))
            if dist <= 1e-6:
                return cand   # warm-start (nominal) itself, collision-free: optimal, early exit
            cands.append((dist, cand))
        elif best is None or (pos_err + rot_err) < (best.pos_err_m + best.normal_err):
            best = cand

    if cands:
        cands.sort(key=lambda c: c[0])
        return cands[0][1]   # closest-to-nominal, collision-free (posture fix)
    # nothing usable: return best-effort; if a collision gate rejected everything, it is converged=False
    return best if best is not None else IKResult(
        q={j: float(nominal[k]) for k, j in enumerate(arm_joints)},
        converged=False, pos_err_m=float("inf"), normal_err=float("inf"), iters=0)
