# coding: utf-8
"""pinocchio IK: FK agrees with fk_chain, and GN+restarts recovers reachable poses."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
from jiuwensymbiosis.adapters.cruzr.geometry import (
    ARM_JOINTS, TOOL_APPROACH_LOCAL, TOOL_PADDLE_LOCAL, APPROACH_FORWARD, PADDLE_INWARD,
)
from jiuwensymbiosis.kinematics import ik_pinocchio as pik
from jiuwensymbiosis.kinematics.fk import fk_chain
from jiuwensymbiosis.kinematics.urdf_chain import parse_chain

_FIXED = {"lifter_pitch_1_joint": 0.0, "lifter_pitch_2_joint": 0.0,
          "lifter_pitch_3_joint": 0.0, "waist_yaw_joint": 0.0}


def _cfg_or_skip():
    cfg = CruzrConfig()
    if not Path(cfg.urdf_path).exists():
        pytest.skip("urdf not present")
    return cfg


def test_recovers_reachable_poses_via_restarts():
    """FK a random arm config to its FULL leaf pose (guaranteed reachable), then confirm
    the solver recovers it — random restarts rescue the seeds the warm-start misses."""
    cfg = _cfg_or_skip()
    chain = parse_chain(cfg.urdf_path, "base_link", cfg.left_arm_leaf)
    limits = chain.limits()
    arm = ARM_JOINTS["left"]
    lo = np.array([limits[j][0] for j in arm]); hi = np.array([limits[j][1] for j in arm])
    rng = np.random.default_rng(7)
    solved = 0
    for t in range(6):
        qa = lo + rng.random(len(arm)) * (hi - lo)
        qd = {**_FIXED, **{j: float(qa[k]) for k, j in enumerate(arm)}}
        tf = fk_chain(chain, qd)
        rot, tcp = tf[:3, :3], tf[:3, 3]
        # derive the axis targets from the actual FK rotation so the target pose is the
        # exact (reachable) FK pose, not a fixed orientation at that position
        approach = rot @ np.asarray(TOOL_APPROACH_LOCAL, dtype=float)
        paddle = rot @ np.asarray(TOOL_PADDLE_LOCAL, dtype=float)
        res = pik.solve_pose_ik_pin(
            cfg.urdf_path, arm, cfg.left_arm_leaf, limits,
            tcp, approach_target=approach, paddle_target=paddle,
            tool_approach_local=TOOL_APPROACH_LOCAL, tool_paddle_local=TOOL_PADDLE_LOCAL,
            tcp_offset_local=(0.0, 0.0, 0.0), q_fixed=_FIXED, q_init=None, seed=t,
        )
        solved += res.converged
    assert solved >= 5   # random restarts recover the vast majority of reachable poses


def test_returns_ikresult_contract():
    cfg = _cfg_or_skip()
    chain = parse_chain(cfg.urdf_path, "base_link", cfg.left_arm_leaf)
    res = pik.solve_pose_ik_pin(
        cfg.urdf_path, ARM_JOINTS["left"], cfg.left_arm_leaf, chain.limits(),
        (0.40, 0.20, 0.65), approach_target=APPROACH_FORWARD, paddle_target=PADDLE_INWARD,
        tool_approach_local=TOOL_APPROACH_LOCAL, tool_paddle_local=TOOL_PADDLE_LOCAL,
        tcp_offset_local=(-0.09, 0.0, 0.0), q_fixed=_FIXED, q_init=None, seed=0,
    )
    assert set(res.q) == set(ARM_JOINTS["left"])
    assert isinstance(res.converged, bool)
    assert res.pos_err_m >= 0.0 and res.iters >= 1


def test_prefers_solution_closest_to_warm_start():
    """Two IK solutions reach the same wrist point; the one nearer the warm start is returned."""
    cfg = _cfg_or_skip()
    chain = parse_chain(cfg.urdf_path, "base_link", cfg.left_arm_leaf)
    limits = chain.limits()
    arm = ARM_JOINTS["left"]
    # a natural target from a mild config; warm-start AT that config -> solver should return ~it
    qd = {**_FIXED, **{j: 0.2 for j in arm}}
    tf = fk_chain(chain, qd)
    approach = tf[:3, :3] @ np.asarray(TOOL_APPROACH_LOCAL, float)
    paddle = tf[:3, :3] @ np.asarray(TOOL_PADDLE_LOCAL, float)
    res = pik.solve_pose_ik_pin(
        cfg.urdf_path, arm, cfg.left_arm_leaf, limits, tf[:3, 3],
        approach_target=approach, paddle_target=paddle,
        tool_approach_local=TOOL_APPROACH_LOCAL, tool_paddle_local=TOOL_PADDLE_LOCAL,
        tcp_offset_local=(0.0, 0.0, 0.0), q_fixed=_FIXED,
        q_init={j: 0.2 for j in arm}, seed=0)
    assert res.converged
    assert max(abs(res.q[j] - 0.2) for j in arm) < 0.2   # stayed near the warm start, not a far config


def test_check_collision_rejects_self_colliding(monkeypatch):
    """With check_collision, converged-but-self-colliding solutions are not returned."""
    from jiuwensymbiosis.kinematics import self_collision as sc
    cfg = _cfg_or_skip()
    chain = parse_chain(cfg.urdf_path, "base_link", cfg.left_arm_leaf)
    monkeypatch.setattr(sc, "available", lambda *a, **k: True)
    monkeypatch.setattr(sc, "in_self_collision", lambda *a, **k: True)   # everything "collides"
    res = pik.solve_pose_ik_pin(
        cfg.urdf_path, ARM_JOINTS["left"], cfg.left_arm_leaf, chain.limits(),
        (0.40, 0.20, 0.65), approach_target=APPROACH_FORWARD, paddle_target=PADDLE_INWARD,
        tool_approach_local=TOOL_APPROACH_LOCAL, tool_paddle_local=TOOL_PADDLE_LOCAL,
        tcp_offset_local=(-0.09, 0.0, 0.0), q_fixed=_FIXED, q_init=None,
        check_collision=True, seed=0)
    assert res.converged is False   # no collision-free solution -> report unreachable/unsafe
