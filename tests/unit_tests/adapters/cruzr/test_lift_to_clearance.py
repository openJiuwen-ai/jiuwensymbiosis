# coding: utf-8
"""lift_to_clearance: stand up (lifter -> 0) then raise the box to the PRESET absolute
height transit_lift_z_m (x/y + gap preserved, no squeeze); unreachable -> no motion."""

from types import SimpleNamespace

import pytest

import jiuwensymbiosis.adapters.cruzr.geometry as gp
from jiuwensymbiosis.adapters.cruzr.api import CruzrApi
from jiuwensymbiosis.adapters.cruzr.geometry import (
    ARM_JOINTS, TOOL_APPROACH_LOCAL, TOOL_PADDLE_LOCAL,
)
from jiuwensymbiosis.kinematics.ik import IKResult
from jiuwensymbiosis.adapters.cruzr.geometry import LIFTER_JOINTS

_ARMS = [j for a in ("left", "right") for j in ARM_JOINTS[a]]
_DET = {"center_mm": [350.0, 0.0, 700.0], "width_mm": 270.0, "height_mm": 200.0,
        "front_x_mm": 290.0, "back_x_mm": 410.0, "top_z_mm": 800.0, "n_points": 5000}


class _FakeChain:
    def limits(self):
        return {j: (-3.14, 3.14) for j in _ARMS}


class _LL:
    def __init__(self, lifter=None):
        self.moves = []
        self._lifter = lifter or {j: 0.0 for j in LIFTER_JOINTS}

    def get_joint_positions(self):
        q = {j: 0.0 for j in _ARMS}
        q.update({j: float(self._lifter.get(j, 0.0)) for j in LIFTER_JOINTS})
        q["waist_yaw_joint"] = 0.0
        return q

    def move_joints_blocking(self, targets, **kw):
        self.moves.append(dict(targets))
        return {"ok": True}


class _Env:
    def __init__(self, lifter=None):
        self.low_level = _LL(lifter)
        self.cfg = SimpleNamespace(
            transit_lift_z_m=0.95, urdf_path="/nonexistent.urdf",
            left_arm_leaf="L_sixforce_link", right_arm_leaf="R_sixforce_link",
            urdf_package_dir="/home/riemann/Robot/Cruzr_ws/cruzr_s2_description")


def _api(monkeypatch, *, converged=True, lifter=None):
    def _fake_parse(urdf, base, leaf):
        c = _FakeChain()
        c.leaf = leaf
        return c

    def _fake_fk(chain, q):
        import numpy as np
        # measured paddle-flange base pose per arm (identity rotation for a simple check)
        tf = np.eye(4)
        tf[:3, 3] = (0.30, 0.13, 0.70) if "L_" in chain.leaf else (0.30, -0.13, 0.70)
        return tf

    monkeypatch.setattr("jiuwensymbiosis.kinematics.urdf_chain.parse_chain", _fake_parse)
    monkeypatch.setattr("jiuwensymbiosis.kinematics.fk.fk_chain", _fake_fk)
    captured = []

    def _fake_ik(chain, q_fixed, arm, tgt, **k):
        captured.append((arm, tgt))
        return IKResult(q={j: 0.1 for j in ARM_JOINTS[arm]}, converged=converged,
                        pos_err_m=0.001 if converged else 0.5, normal_err=0.001, iters=3)

    monkeypatch.setattr(gp, "solve_arm_ik", _fake_ik)
    env = _Env(lifter)
    api = CruzrApi(env)
    api._last_grasped_box = dict(_DET)
    return api, env, captured


def test_lift_raises_to_preset_absolute_height(monkeypatch):
    # The lift raises each paddle to the PRESET absolute z (transit_lift_z_m), keeping
    # the FK x/y and orientation -> gap preserved, NOT a relative +dz and NOT re-derived
    # from box geometry.
    api, env, captured = _api(monkeypatch, converged=True)
    out = api.lift_to_clearance()
    assert out["ok"] is True and out["target_z_m"] == pytest.approx(0.95)
    by_arm = {arm: tgt for arm, tgt in captured}
    # left flange (0.30, 0.13, ...) + tcp_offset (-0.09, 0, 0) -> tcp x/y (0.21, 0.13); z -> preset 0.95
    assert by_arm["left"].pos_m == pytest.approx((0.21, 0.13, 0.95))
    # right flange (0.30, -0.13, ...) + tcp_offset (+0.09, 0, 0) -> tcp x/y (0.39, -0.13); z -> preset 0.95
    assert by_arm["right"].pos_m == pytest.approx((0.39, -0.13, 0.95))
    # both paddles commanded to the SAME absolute z, and x/y preserved -> gap unchanged
    assert by_arm["left"].pos_m[1] - by_arm["right"].pos_m[1] == pytest.approx(0.26)
    # orientation is carried from the live pose (identity R here -> the tool-local axes),
    # so a box pitched by the grasp / turned by turn_waist is raised without being twisted.
    for a in ("left", "right"):
        assert by_arm[a].approach == pytest.approx(TOOL_APPROACH_LOCAL)
        assert by_arm[a].paddle == pytest.approx(TOOL_PADDLE_LOCAL)
    # already upright (lifter all 0) -> no stand-up; only the raise was commanded.
    assert out["stood_up"] is False
    assert len(env.low_level.moves) == 1


def test_lift_merges_standup_and_raise_when_leaned(monkeypatch):
    # Torso leaned at grasp: the lifter -> 0 and the arm raise are commanded as ONE
    # coordinated move (arms to the raised IK pose + lifter to 0 together).
    leaned = {"lifter_pitch_1_joint": 0.5, "lifter_pitch_2_joint": -0.5, "lifter_pitch_3_joint": 0.0}
    api, env, _ = _api(monkeypatch, converged=True, lifter=leaned)
    out = api.lift_to_clearance()
    assert out["ok"] is True and out["stood_up"] is True
    assert out["lifter_from"]["lifter_pitch_1_joint"] == pytest.approx(0.5)
    assert len(env.low_level.moves) == 1          # single coordinated motion
    cmd = env.low_level.moves[0]
    for j in LIFTER_JOINTS:                        # lifter driven to 0 in the SAME move
        assert cmd[j] == pytest.approx(0.0)
    for a in ("left", "right"):                    # arms to the raised IK pose (0.1) together
        for j in ARM_JOINTS[a]:
            assert cmd[j] == pytest.approx(0.1)


def test_lift_skips_standup_within_tol(monkeypatch):
    # Lifter within upright_tol_rad of 0 -> treated as upright, no stand-up move (raise only).
    near0 = {j: 0.03 for j in LIFTER_JOINTS}
    api, env, _ = _api(monkeypatch, converged=True, lifter=near0)
    out = api.lift_to_clearance()               # default upright_tol_rad=0.05
    assert out["ok"] is True and out["stood_up"] is False
    assert len(env.low_level.moves) == 1


def test_lift_unreachable_makes_no_move(monkeypatch):
    api, env, _ = _api(monkeypatch, converged=False)
    out = api.lift_to_clearance()
    assert out["ok"] is False and out["reason"] == "lift_unreachable"
    assert env.low_level.moves == []


def test_lift_no_box(monkeypatch):
    monkeypatch.setattr("jiuwensymbiosis.kinematics.urdf_chain.parse_chain", lambda *a, **k: _FakeChain())
    env = _Env()
    api = CruzrApi(env)
    assert api.lift_to_clearance()["reason"] == "no_box_to_lift"
