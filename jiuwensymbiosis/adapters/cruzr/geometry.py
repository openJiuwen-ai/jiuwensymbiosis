# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cruzr body geometry — the physical-machine facts of this robot.

Two irreducibly body-specific pieces (no generic abstraction exists for a sample
of one — see the porting notes): the dual-arm **paddle grasp planner** (paddle TCP
offset + tool axes + side-clamp targeting, per-arm IK) and the 3-joint **lifter**
level-torso manifold reach search. Both call the *generic* solvers in
``jiuwensymbiosis.kinematics`` — only the paddle/lifter geometry lives here.

Merged from the former ``grasp_planner.py`` + ``lifter.py`` so the adapter keeps
only the skeleton files (config/env/session/lowlevel/api/geometry/_calibration).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from jiuwensymbiosis.kinematics.fk import fk_chain
from jiuwensymbiosis.kinematics.ik import IKResult, ik_solve_pose
from jiuwensymbiosis.kinematics.urdf_chain import Chain
from jiuwensymbiosis.perception.object_geometry import ObjectGeometry3D

logger = logging.getLogger(__name__)




@dataclass
class ArmTarget:
    arm: str
    pos_m: tuple[float, float, float]          # paddle TCP contact target (base, m)
    approach: tuple[float, float, float]       # base dir the tool approach axis aligns to
    paddle: tuple[float, float, float]         # base dir the tool paddle-face axis aligns to
    tcp_offset_local: tuple[float, float, float]  # paddle offset from leaf in tool frame (m)


@dataclass
class GraspPlan:
    ok: bool
    reason: str
    lifter_center_z_mm: float
    approach: dict[str, ArmTarget]   # above the box sides (raise clear of the table)
    descend: dict[str, ArmTarget]    # straight down to clamp height, still outside faces
    clamp: dict[str, ArmTarget]      # inward onto the side faces
    ik: dict[str, IKResult]          # clamp IK


ARM_JOINTS = {
    "left": ["L_shoulder_pitch_joint", "L_shoulder_roll_joint", "L_shoulder_yaw_joint",
             "L_elbow_roll_joint", "L_elbow_yaw_joint", "L_wrist_pitch_joint", "L_wrist_roll_joint"],
    "right": ["R_shoulder_pitch_joint", "R_shoulder_roll_joint", "R_shoulder_yaw_joint",
              "R_elbow_roll_joint", "R_elbow_yaw_joint", "R_wrist_pitch_joint", "R_wrist_roll_joint"],
}

# Hardware-fixed paddle geometry (validated live). Both hands constrain tool-x
# (paddle-face normal) to +Y because the hands are mirrored: physical paddle
# normal = +tool-x (right) / -tool-x (left), so the same +Y target presses both
# inward. The TCP offset sign mirrors per arm.
APPROACH_FORWARD = (1.0, 0.0, 0.0)        # end-plane (tool-z) points forward +X
PADDLE_INWARD = (0.0, 1.0, 0.0)           # paddle face (tool-x) -> +Y for both arms
TOOL_APPROACH_LOCAL = (0.0, 0.0, 1.0)     # tool-z is the end-plane normal
TOOL_PADDLE_LOCAL = (1.0, 0.0, 0.0)       # tool-x is the paddle-face normal
PADDLE_TCP_OFFSET_M = 0.09
_TCP_SIGN = {"left": -1.0, "right": 1.0}  # paddle 9 cm along -tool-x (L) / +tool-x (R)


def ready_targets(
    *, x_m: float = 0.38, y_m: float = 0.26, z_m: float = 0.80,
    tcp_offset_m: float = PADDLE_TCP_OFFSET_M,
) -> dict[str, ArmTarget]:
    """Box-independent 'ready' (pre-grasp embrace) paddle-TCP targets.

    Both paddles in front of the chest, vertical and facing INWARD (same
    forward+inward orientation as the clamp) but spread WIDE (large spacing) and
    held high — clear of a table in front. Used as a transit waypoint so the
    home<->grasp motion never sweeps the arms up into the table from below.
    """
    def _tgt(arm: str, y: float) -> ArmTarget:
        off = (_TCP_SIGN[arm] * tcp_offset_m, 0.0, 0.0)
        return ArmTarget(arm, (x_m, y, z_m), APPROACH_FORWARD, PADDLE_INWARD, off)

    return {"left": _tgt("left", y_m), "right": _tgt("right", -y_m)}


def _depth_mid_mm(box: ObjectGeometry3D) -> float:
    """Side-face depth middle; falls back to front_x if no back face was seen."""
    if box.back_x_mm > box.front_x_mm:
        return (box.front_x_mm + box.back_x_mm) / 2.0
    return box.front_x_mm


def plan_clamp_targets(
    box: ObjectGeometry3D,
    *,
    inset_mm: float = 8.0,
    pre_clear_mm: float = 60.0,
    approach_dz_mm: float = 100.0,
    tcp_offset_m: float = PADDLE_TCP_OFFSET_M,
) -> tuple[dict[str, ArmTarget], dict[str, ArmTarget], dict[str, ArmTarget]]:
    """Top-down paddle-TCP waypoints (meters), no IK: approach, descend, clamp.

    ``approach`` is ``approach_dz_mm`` ABOVE the box at the side+clearance — the
    arms raise here first (clear of the table). ``descend`` is straight down to
    clamp height, still outside the faces. ``clamp`` then moves inward. The
    approach-from-above avoids sweeping the hand up into the table from below.
    """
    _, cy, _ = box.center_mm
    half = box.width_mm / 2.0
    # Clamp at the vertical MIDDLE of the box's z-extent, NOT center_mm[2] (= median of the masked
    # cloud). A downward eye-in-hand camera sees the dense TOP FACE plus a sparser front face, so the
    # median z is pulled UP toward the top → the paddles grip near the top face. The extent midpoint
    # (top_z - height/2 = (z_hi+z_lo)/2) is density-independent and sits at the true middle. Place
    # stays consistent: it re-plans from the same box then surface-shifts the box BOTTOM onto the
    # table, so the grip height it assumes matches this one (the shift cancels the absolute z).
    z_mid_mm = box.top_z_mm - box.height_mm / 2.0
    x_m, z_m = _depth_mid_mm(box) / 1000.0, z_mid_mm / 1000.0
    z_above = (z_mid_mm + approach_dz_mm) / 1000.0

    def _tgt(arm: str, y_mm: float, z: float) -> ArmTarget:
        off = (_TCP_SIGN[arm] * tcp_offset_m, 0.0, 0.0)
        return ArmTarget(arm, (x_m, y_mm / 1000.0, z), APPROACH_FORWARD, PADDLE_INWARD, off)

    wide = {"left": cy + half + pre_clear_mm, "right": cy - half - pre_clear_mm}
    inner = {"left": cy + half - inset_mm, "right": cy - half + inset_mm}
    approach = {a: _tgt(a, wide[a], z_above) for a in ("left", "right")}
    descend = {a: _tgt(a, wide[a], z_m) for a in ("left", "right")}
    clamp = {a: _tgt(a, inner[a], z_m) for a in ("left", "right")}
    return approach, descend, clamp


def solve_arm_ik(
    chain: Chain, q_fixed: dict, arm: str, tgt: ArmTarget, *,
    pos_tol_m: float = 0.016, q_init: dict | None = None, max_iters: int = 1500,
    n_restarts: int = 4, check_collision: bool = False, package_dir: str | None = None,
) -> IKResult:
    """Solve full-orientation IK for one arm's paddle target.

    Dispatches to the pinocchio solver (analytic Jacobian + random restarts) when
    pinocchio is importable and the chain carries its URDF path/leaf; otherwise
    falls back to the retained legacy DLS. ``q_init`` warm-starts either solver.
    """
    from jiuwensymbiosis.kinematics import ik_pinocchio as _pik

    init = q_init if q_init is not None else {j: q_fixed.get(j, 0.0) for j in ARM_JOINTS[arm]}
    if _pik.pin_available() and getattr(chain, "urdf_path", "") and getattr(chain, "leaf_link", ""):
        try:
            return _pik.solve_pose_ik_pin(
                chain.urdf_path, ARM_JOINTS[arm], chain.leaf_link, chain.limits(),
                tgt.pos_m, tgt.approach, tgt.paddle,
                tool_approach_local=TOOL_APPROACH_LOCAL, tool_paddle_local=TOOL_PADDLE_LOCAL,
                tcp_offset_local=tgt.tcp_offset_local, q_fixed=q_fixed, q_init=init,
                pos_tol_m=pos_tol_m, max_iters=min(max_iters, 200), n_restarts=n_restarts,
                check_collision=check_collision, package_dir=package_dir,
            )
        except Exception as exc:  # noqa: BLE001  # never let IK-backend errors abort a grasp; fall back
            logger.warning("[cruzr] pinocchio IK failed (%s); falling back to legacy DLS", exc)

    return ik_solve_pose(
        chain, q_fixed, ARM_JOINTS[arm], tgt.pos_m, tgt.approach, tgt.paddle,
        tool_approach_local=TOOL_APPROACH_LOCAL, tool_paddle_local=TOOL_PADDLE_LOCAL,
        tcp_offset_local=tgt.tcp_offset_local, q_init=init, pos_tol_m=pos_tol_m, max_iters=max_iters,
    )


def solve_grasp(
    box: ObjectGeometry3D,
    left_chain: Chain,
    right_chain: Chain,
    q_fixed: dict[str, float],
    *,
    inset_mm: float = 8.0,
    pre_clear_mm: float = 60.0,
    tcp_offset_m: float = PADDLE_TCP_OFFSET_M,
    pos_tol_m: float = 0.016,
    ik_max_iters: int = 1500,
    check_collision: bool = False,
    package_dir: str | None = None,
) -> GraspPlan:
    """Plan paddle targets then solve full-orientation IK for both clamp poses.

    ``ik_max_iters`` can be lowered (e.g. ~300) for a fast reach-feasibility pass
    such as the adaptive-lifter search, which evaluates many body poses.
    """
    if not box.ok:
        return GraspPlan(False, f"box:{box.reason}", 0.0, {}, {}, {}, {})
    approach, descend, clamp = plan_clamp_targets(
        box, inset_mm=inset_mm, pre_clear_mm=pre_clear_mm, tcp_offset_m=tcp_offset_m)
    chains = {"left": left_chain, "right": right_chain}
    ik = {arm: solve_arm_ik(chains[arm], q_fixed, arm, clamp[arm], pos_tol_m=pos_tol_m,
                            max_iters=ik_max_iters, check_collision=check_collision, package_dir=package_dir)
          for arm in ("left", "right")}
    ok = all(r.converged for r in ik.values())
    return GraspPlan(ok, "" if ok else "ik_no_converge", box.center_mm[2], approach, descend, clamp, ik)


# Comfortable fraction of an arm's max straight-line reach to aim the clamp
# target at (0 = at the shoulder, 1 = fully outstretched). ~0.7 keeps the elbow
# bent and away from both the inner singularity and the reach limit.
_COMFORT_FRAC = 0.7
_REACH_LIMIT_FRAC = 0.99  # beyond this fraction of max reach: treat as unreachable

LIFTER_JOINTS = ["lifter_pitch_1_joint", "lifter_pitch_2_joint", "lifter_pitch_3_joint"]
_P1, _P2, _P3 = LIFTER_JOINTS
LIFTER_LIMITS = {
    _P1: (-1.5708, 1.5708),
    _P2: (-2.6179939, 2.6179939),
    _P3: (-1.5708, 1.5708),
}


def level_config(p1: float, p3: float) -> dict[str, float] | None:
    """Level-torso lifter config ``{p1, -(p1+p3), p3}``; None if any joint is out of limit."""
    vals = {_P1: float(p1), _P2: float(-(p1 + p3)), _P3: float(p3)}
    for j, (lo, hi) in LIFTER_LIMITS.items():
        if not (lo <= vals[j] <= hi):
            return None
    return vals


@dataclass
class LifterPlan:
    found: bool
    q_lifter: dict[str, float] | None
    score: float
    improves: bool          # True => the lifter should move (current pose can't grasp)
    reason: str


def _reach_score(box, left_chain, right_chain, q_fixed, ik_max_iters) -> float | None:
    """Reach margin for a body pose: -max(arm pos_err) if both arms grasp, else None."""
    plan = solve_grasp(box, left_chain, right_chain, q_fixed, ik_max_iters=ik_max_iters)
    if not plan.ok:
        return None
    return -max(plan.ik["left"].pos_err_m, plan.ik["right"].pos_err_m)


def _arm_base_z(chain: Chain, lifter_cfg: dict[str, float], waist_yaw: float) -> float:
    """z (m) of the arm leaf with arm joints at 0. On the level manifold the torso
    keeps its orientation, so this point translates exactly with the shoulder
    mount — its change across lifter configs equals how far a rigidly held arm's
    hand moves vertically when the body leans."""
    return float(fk_chain(chain, {**lifter_cfg, "waist_yaw_joint": waist_yaw})[2, 3])


class _SubChain:
    """A view of the leading joints of a Chain (for partial FK to the shoulder)."""
    __slots__ = ("joints",)

    def __init__(self, joints):
        self.joints = joints


def _shoulder_pos(chain: Chain, arm: str, lifter_cfg: dict[str, float], waist_yaw: float) -> np.ndarray:
    """Base-frame position (m) of the arm root (first arm joint) for a lifter cfg."""
    arm_set = set(ARM_JOINTS[arm])
    js = chain.joints
    idx = next((i for i, j in enumerate(js) if j.name in arm_set), len(js))
    t = fk_chain(_SubChain(js[:idx]), {**lifter_cfg, "waist_yaw_joint": waist_yaw})
    return t[:3, 3]


def _arm_reach_max(chain: Chain, arm: str) -> float:
    """Max straight-line reach (m) from the arm root: sum of arm link offsets."""
    arm_set = set(ARM_JOINTS[arm])
    started, total = False, 0.0
    for j in chain.joints:
        if j.name in arm_set:
            started = True
        if started:
            total += float(np.linalg.norm(np.asarray(j.xyz, dtype=float)))
    return total


def lower_torso_lifter(
    chain: Chain,
    arm: str,
    lifter_now: dict[str, float],
    waist_yaw: float,
    dz_m: float,
    *,
    iters: int = 5,
) -> dict[str, float] | None:
    """Level-manifold lifter config that lowers the torso ~``dz_m`` vertically.

    Solves (Newton on the FK shoulder x/z) for a new (p1, p3) on the level
    manifold whose shoulder is ``dz_m`` lower in z with ~unchanged x — i.e. the
    upper body descends straight down. On the level manifold the torso keeps its
    orientation, so a rigidly held arm's paddle drops by the same vector. Returns
    ``None`` if no nearby valid config achieves it (joint limits / singular).
    """
    p1 = float(lifter_now.get(_P1, 0.0))
    p3 = float(lifter_now.get(_P3, 0.0))
    base = level_config(p1, p3)
    if base is None:
        return None
    s0 = _shoulder_pos(chain, arm, base, waist_yaw)
    tx, tz = float(s0[0]), float(s0[2]) - dz_m
    h = 1e-4
    for _ in range(iters):
        lc = level_config(p1, p3)
        if lc is None:
            return None
        s = _shoulder_pos(chain, arm, lc, waist_yaw)
        ex, ez = float(s[0]) - tx, float(s[2]) - tz
        if abs(ex) < 1e-4 and abs(ez) < 1e-4:
            break
        lp1, lp3 = level_config(p1 + h, p3), level_config(p1, p3 + h)
        if lp1 is None or lp3 is None:
            return None
        s1 = _shoulder_pos(chain, arm, lp1, waist_yaw)
        s3 = _shoulder_pos(chain, arm, lp3, waist_yaw)
        jac = np.array([[(s1[0] - s[0]) / h, (s3[0] - s[0]) / h],
                        [(s1[2] - s[2]) / h, (s3[2] - s[2]) / h]])
        try:
            d = np.linalg.solve(jac, -np.array([ex, ez]))
        except np.linalg.LinAlgError:
            return None
        p1 += float(d[0])
        p3 += float(d[1])
    return level_config(p1, p3)


def _geo_reach_score(chains: dict, clamp: dict, lc: dict[str, float], waist_yaw: float) -> float | None:
    """Cheap geometric reachability proxy for a lifter cfg (no IK).

    For each arm: distance from the shoulder (at this lifter cfg) to the clamp
    target, as a fraction of the arm's max reach. Returns ``None`` if either arm
    is beyond reach; otherwise ``-max_arm penalty`` where penalty rewards a
    comfortable mid-reach fraction (so higher score = better-centered)."""
    worst = 0.0
    for a in ("left", "right"):
        sh = _shoulder_pos(chains[a], a, lc, waist_yaw)
        tgt = np.asarray(clamp[a].pos_m, dtype=float)
        d = float(np.linalg.norm(tgt - sh))
        r_max = _arm_reach_max(chains[a], a)
        if r_max <= 0.0 or d > _REACH_LIMIT_FRAC * r_max:
            return None
        worst = max(worst, (d / r_max - _COMFORT_FRAC) ** 2)
    return -worst


def search_lifter_for_box(
    box: ObjectGeometry3D,
    left_chain: Chain,
    right_chain: Chain,
    current_lifter: dict[str, float],
    waist_yaw: float = 0.0,
    *,
    p1_range: tuple[float, float] = (-0.6, 1.2),
    p3_range: tuple[float, float] = (-0.6, 1.2),
    step: float = 0.3,
    ik_max_iters: int = 500,
    table_clearance_m: float = 0.05,
    scan_ik_iters: int = 80,
) -> LifterPlan:
    """Find the lifter config (level manifold) that best reaches ``box``.

    Fast strategy — geometric PRUNE, then reduced-iteration IK ranking:

    1. If the CURRENT lifter already grasps, keep it (one IK, ``improves=False``).
    2. Otherwise, for each (p1, p3) on the level manifold: a cheap *geometric*
       check prunes cells whose clamp targets lie beyond the arm's reach (sound —
       it never drops a graspable cell), and the table-floor guard prunes cells
       that would drag the held ready arms below the table. Surviving cells are
       ranked by a REDUCED-iteration dual-arm clamp IK margin. Geometry alone
       cannot predict the paddle-orientation IK (the binding constraint), so it
       is used only to prune — never to rank. The reduced iteration budget is
       faithful because the IK position error is stable well below the full
       budget; the precise solve happens later in ``dual_arm_grasp``.

    Table-floor guard: after the body leans, the arms are still held at the ready
    (pre-grasp) joints solved at the upright torso, so the hands translate down
    with the shoulders. Any config that would drag those held hands below
    ``table_z + table_clearance_m`` (the box sits on the table) is rejected.

    Returns ``found=False`` with reason ``"unreachable_any_lifter"`` if no config
    grasps, or ``"no_safe_lifter"`` if the only reachable configs were rejected
    by the table-floor guard.
    """
    if not box.ok:
        return LifterPlan(False, None, float("-inf"), False, f"box:{box.reason}")

    chains = {"left": left_chain, "right": right_chain}

    # 1. current pose already grasps -> keep it, do not move the lifter. (No lean
    #    happens, so the table-floor guard below is unnecessary for this case.)
    cur = {j: float(current_lifter.get(j, 0.0)) for j in LIFTER_JOINTS}
    cur_score = _reach_score(box, left_chain, right_chain, {**cur, "waist_yaw_joint": waist_yaw}, ik_max_iters)
    if cur_score is not None:
        return LifterPlan(True, cur, cur_score, False, "")

    # Table-floor references (only needed once we must search for a new pose).
    table_z = (box.top_z_mm - box.height_mm) / 1000.0
    floor_z = table_z + table_clearance_m
    ready_z = ready_targets()["left"].pos_m[2]
    ref_cur = {a: _arm_base_z(chains[a], cur, waist_yaw) for a in ("left", "right")}

    def _floor_ok(lc: dict[str, float]) -> bool:
        """True iff both held-ready hands stay above the table after leaning to lc."""
        for a in ("left", "right"):
            held_z = ready_z + (_arm_base_z(chains[a], lc, waist_yaw) - ref_cur[a])
            if held_z < floor_z:
                return False
        return True

    # Clamp targets are box-fixed (lifter-invariant); used by the geometric prune.
    clamp = plan_clamp_targets(box)[2]

    def grid(rng: tuple[float, float]) -> list[float]:
        lo, hi = rng
        n = int(round((hi - lo) / step))
        return [lo + i * step for i in range(n + 1)]

    # 2. geometric prune (sound) + floor prune, then rank survivors by a
    #    reduced-iteration IK reach margin.
    best: dict | None = None
    best_score = float("-inf")
    any_geo_unsafe = False
    for p1 in grid(p1_range):
        for p3 in grid(p3_range):
            lc = level_config(p1, p3)
            if lc is None:
                continue
            if _geo_reach_score(chains, clamp, lc, waist_yaw) is None:
                continue  # clamp targets beyond arm reach: cannot grasp from here
            if not _floor_ok(lc):
                any_geo_unsafe = True
                continue  # would drag the held ready arms below the table
            rs = _reach_score(box, left_chain, right_chain, {**lc, "waist_yaw_joint": waist_yaw}, scan_ik_iters)
            if rs is not None and rs > best_score:
                best_score, best = rs, lc

    if best is None:
        reason = "no_safe_lifter" if any_geo_unsafe else "unreachable_any_lifter"
        return LifterPlan(False, None, float("-inf"), False, reason)
    return LifterPlan(True, best, best_score, True, "")


def search_lifter_for_place(
    clamp: dict,
    left_chain: Chain,
    right_chain: Chain,
    current_lifter: dict[str, float],
    waist_yaw: float = 0.0,
    *,
    max_lean_rad: float = 0.8,
    step: float = 0.15,
    scan_ik_iters: int = 80,
) -> LifterPlan:
    """Find the SMALLEST lifter adjustment (level manifold) from which BOTH arms reach
    the given PLACE clamp targets — the arms do the placing, the lifter only nudges.

    The mirror of :func:`search_lifter_for_box`, with three differences that matter for
    placing a HELD box (rather than reaching an empty box on a table):

    1. The clamp targets are passed in directly — for place they come from live FK of the
       held box + the sensed surface, not from box geometry.
    2. There is NO ready-arm table-floor guard (the arms HOLD the box and must descend to
       the surface, so that guard would reject exactly the low configs place needs).
    3. It prefers the MINIMAL lifter change (least ``|Δp1|+|Δp3|`` from the current lifter)
       among reachable configs — NOT the best reach margin — and never leans more than
       ``max_lean_rad`` off each pitch. A big forward lean drops the torso onto the table;
       the arms, not the lifter, should position the box. If even the largest allowed nudge
       cannot reach, returns ``found=False`` / ``"place_unreachable_any_lifter"`` so the
       caller drives closer instead of leaning into the table.
    """
    chains = {"left": left_chain, "right": right_chain}

    def _reach(lc: dict[str, float]) -> float | None:
        """-max(arm pos_err) if both arms reach the place targets from ``lc``, else None."""
        qf = {**lc, "waist_yaw_joint": waist_yaw}
        worst = 0.0
        for a in ("left", "right"):
            r = solve_arm_ik(chains[a], qf, a, clamp[a], max_iters=scan_ik_iters)
            if not r.converged:
                return None
            worst = max(worst, r.pos_err_m)
        return -worst

    # 1. current pose already reaches -> keep it, do not move the lifter at all.
    cur = {j: float(current_lifter.get(j, 0.0)) for j in LIFTER_JOINTS}
    cur_score = _reach(cur)
    if cur_score is not None:
        return LifterPlan(True, cur, cur_score, False, "")

    # 2. small symmetric grid within +-max_lean_rad of each pitch (includes 0); geometric
    #    prune, then keep the reachable config with the SMALLEST lift change from current.
    k = max(1, int(max_lean_rad / step))
    cells = [i * step for i in range(-k, k + 1)]
    cur_p1, cur_p3 = cur[_P1], cur[_P3]
    best: dict | None = None
    best_lean = float("inf")
    best_score = float("-inf")
    for p1 in cells:
        for p3 in cells:
            lean = abs(p1 - cur_p1) + abs(p3 - cur_p3)
            if lean >= best_lean:
                continue  # cannot beat the smallest-lean reachable config found so far
            lc = level_config(p1, p3)
            if lc is None or _geo_reach_score(chains, clamp, lc, waist_yaw) is None:
                continue  # invalid / place targets beyond arm reach from here
            rs = _reach(lc)
            if rs is not None:
                best, best_lean, best_score = lc, lean, rs

    if best is None:
        return LifterPlan(False, None, float("-inf"), False, "place_unreachable_any_lifter")
    return LifterPlan(True, best, best_score, True, "")
