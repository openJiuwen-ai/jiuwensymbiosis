# coding: utf-8
"""dual_arm_place: land the box at a VALIDATED xy point ON the sensed table (Y centred, X
just inside the near edge; bail if it can't fit), lean the lifter to reach it, then
lower/release. Legacy surface=None path keeps the live-FK carried xy."""

from types import SimpleNamespace

import pytest

import jiuwensymbiosis.adapters.cruzr.geometry as gp
import jiuwensymbiosis.adapters.cruzr.geometry as lifter_mod
from jiuwensymbiosis.adapters.cruzr.api import CruzrApi
from jiuwensymbiosis.adapters.cruzr.geometry import ARM_JOINTS
from jiuwensymbiosis.kinematics.ik import IKResult
from jiuwensymbiosis.adapters.cruzr.geometry import LIFTER_JOINTS, LifterPlan

_ARMS = [j for a in ("left", "right") for j in ARM_JOINTS[a]]
# Grasp-time geometry (what place would WRONGLY use if it read stale geometry):
# center x=350, y=0; box depth = back-front = 120 mm, half-width = 135 mm.
_DET = {"center_mm": [350.0, 0.0, 700.0], "width_mm": 270.0, "height_mm": 200.0,
        "front_x_mm": 290.0, "back_x_mm": 410.0, "top_z_mm": 800.0, "n_points": 5000}
# A sensed table footprint: x in [800,1200] (depth 400), side centre y=30, width 600, top z 500.
_SURFACE = {"ok": True, "surface_z_mm": 500.0, "center_mm": [1000.0, 30.0, 500.0],
            "front_x_mm": 800.0, "back_x_mm": 1200.0, "width_mm": 600.0, "n_points": 8000}


class _FakeChain:
    def limits(self):
        return {j: (-3.14, 3.14) for j in _ARMS}


class _LL:
    def __init__(self):
        self.moves = []
        self.streams = []
        self._lifter = {j: 0.0 for j in LIFTER_JOINTS}   # tracks the lean so FK reflects it post-move

    def get_joint_positions(self):
        q = {j: 0.0 for j in _ARMS}
        q.update(self._lifter)
        q["waist_yaw_joint"] = 0.0
        return q

    def move_joints_blocking(self, targets, **kw):
        self.moves.append(dict(targets))
        for j in LIFTER_JOINTS:                           # a lifter lean updates the tracked pose
            if j in targets:
                self._lifter[j] = float(targets[j])
        return {"ok": True}

    def stream_joint_trajectory(self, waypoints, **kw):
        # the gap-locked lower is streamed as a Cartesian path; record the final knot
        # (== the clamp target) so tests still see that a lower happened.
        self.streams.append([dict(w) for w in waypoints])
        self.moves.append(dict(waypoints[-1]))
        return {"ok": True}


class _Env:
    def __init__(self):
        self.low_level = _LL()
        self.cfg = SimpleNamespace(
            urdf_path="/nonexistent.urdf", left_arm_leaf="L_sixforce_link",
            right_arm_leaf="R_sixforce_link", place_edge_margin_mm=20.0,
            place_max_lift_lean_rad=0.35,
            urdf_package_dir="/home/riemann/Robot/Cruzr_ws/cruzr_s2_description")


def _api(monkeypatch, *, improves=False, q_lifter=None):
    def _fake_parse(urdf, base, leaf):
        c = _FakeChain()
        c.leaf = leaf
        return c

    def _fake_fk(chain, q):
        import numpy as np
        # Box carried at a "stood-up" location: left flange (0.30, 0.15, 0.90),
        # right (0.30, -0.05, 0.90). With tcp_offset (∓0.09, 0, 0) and identity R:
        # tcp_L=(0.21,0.15,0.90), tcp_R=(0.39,-0.05,0.90) -> centre x=0.30, y=0.05.
        # A lifter lean DROPS the flange z (0.2 m/rad of lifter_pitch_1); at lean 0 the
        # pose is unchanged, so non-lean tests see the original 0.90.
        z = 0.90 - 0.2 * float(q.get("lifter_pitch_1_joint", 0.0))
        tf = np.eye(4)
        tf[:3, 3] = (0.30, 0.15, z) if "L_" in chain.leaf else (0.30, -0.05, z)
        return tf

    captured = []

    def _fake_ik(chain, q_fixed, arm, tgt, **k):
        captured.append((arm, tgt))
        return IKResult(q={j: 0.1 for j in ARM_JOINTS[arm]}, converged=True,
                        pos_err_m=0.001, normal_err=0.001, iters=3)

    def _fake_search(clamp, lc, rc, current_lifter, waist_yaw, **k):
        return LifterPlan(found=True, q_lifter=(q_lifter or dict(current_lifter)),
                          score=-0.01, improves=improves, reason="")

    # dual_arm_place imports parse_chain LOCALLY from urdf_chain, so patch the source module.
    monkeypatch.setattr("jiuwensymbiosis.kinematics.urdf_chain.parse_chain", _fake_parse)
    monkeypatch.setattr("jiuwensymbiosis.kinematics.fk.fk_chain", _fake_fk)
    monkeypatch.setattr(gp, "solve_arm_ik", _fake_ik)
    monkeypatch.setattr(lifter_mod, "search_lifter_for_place", _fake_search)
    monkeypatch.setattr(lifter_mod, "lower_torso_lifter", lambda *a, **k: None)  # skip rim descend
    env = _Env()
    api = CruzrApi(env)
    api._last_grasped_box = dict(_DET)
    return api, env, captured


def test_place_legacy_uses_live_fk_carried_xy_when_no_surface(monkeypatch):
    api, env, captured = _api(monkeypatch)
    out = api.dual_arm_place()                                   # surface=None -> legacy z-only
    assert out["ok"] is True
    # legacy path: landing == carried, taken from FK (300, 50 mm), NOT grasp geometry (350, 0).
    assert out["carried_mm"] == pytest.approx([300.0, 50.0])
    assert out["landing_mm"] == pytest.approx([300.0, 50.0])
    # first two solve_arm_ik calls are the "lower" to the left then right clamp targets.
    (arm0, clamp_left), (arm1, clamp_right) = captured[0], captured[1]
    assert (arm0, arm1) == ("left", "right")
    # Two-hand spacing PRESERVED from the live FK grip (tcp_L y=0.15, tcp_R y=-0.05 → gap 0.20 m), NOT
    # re-derived from box.width-2·inset. The box CENTRE lands at the carried xy (0.30, 0.05 m); no
    # surface given -> grasp centre height 0.70 m.
    assert clamp_left.pos_m[1] - clamp_right.pos_m[1] == pytest.approx(0.20)         # gap unchanged
    assert 0.5 * (clamp_left.pos_m[0] + clamp_right.pos_m[0]) == pytest.approx(0.30)
    assert 0.5 * (clamp_left.pos_m[1] + clamp_right.pos_m[1]) == pytest.approx(0.05)
    assert clamp_left.pos_m[2] == pytest.approx(0.70) and clamp_right.pos_m[2] == pytest.approx(0.70)


def test_place_squeeze_tightens_gap_symmetrically(monkeypatch):
    # A held box slips during the lower unless the paddles PRESS inward (grasp overshoots the
    # faces; place must too). place_squeeze_mm pulls each paddle place_squeeze_mm/2 toward the box
    # centre PAST the preserved gap: the gap shrinks by exactly that much, centre + x unchanged.
    api, env, captured = _api(monkeypatch)
    env.cfg.place_squeeze_mm = 20.0
    out = api.dual_arm_place()
    assert out["ok"] is True
    (arm0, clamp_left), (arm1, clamp_right) = captured[0], captured[1]
    assert (arm0, arm1) == ("left", "right")
    assert clamp_left.pos_m[1] - clamp_right.pos_m[1] == pytest.approx(0.20 - 0.020)   # gap tightened 20 mm
    assert 0.5 * (clamp_left.pos_m[1] + clamp_right.pos_m[1]) == pytest.approx(0.05)    # centre preserved
    assert 0.5 * (clamp_left.pos_m[0] + clamp_right.pos_m[0]) == pytest.approx(0.30)    # x preserved


def test_place_no_squeeze_when_zero_preserves_gap(monkeypatch):
    # place_squeeze_mm=0 is the old behaviour: the held gap is preserved bit-for-bit (no press).
    api, env, captured = _api(monkeypatch)
    env.cfg.place_squeeze_mm = 0.0
    out = api.dual_arm_place()
    assert out["ok"] is True
    (_, clamp_left), (_, clamp_right) = captured[0], captured[1]
    assert clamp_left.pos_m[1] - clamp_right.pos_m[1] == pytest.approx(0.20)            # gap unchanged


def test_place_lands_box_on_table_centre_near_edge(monkeypatch):
    api, env, captured = _api(monkeypatch)
    out = api.dual_arm_place(surface=dict(_SURFACE))
    assert out["ok"] is True
    # landing XY comes from the TABLE footprint, NOT the carried XY (300, 50):
    # x = near_x(800) + box_depth(120)/2 + margin(20) = 880 mm; y = table cy = 30 mm.
    assert out["landing_mm"] == pytest.approx([880.0, 30.0])
    assert out["carried_mm"] == pytest.approx([300.0, 50.0])
    (_, clamp_left), (_, clamp_right) = captured[0], captured[1]
    # box CENTRE lands at the table landing (880, 30 mm), near edge inside; the two-hand spacing is
    # PRESERVED (0.20 m from the live FK grip), NOT box-derived; z lowered so the box bottom sits on 500.
    assert 0.5 * (clamp_left.pos_m[0] + clamp_right.pos_m[0]) == pytest.approx(0.880)
    assert 0.5 * (clamp_left.pos_m[1] + clamp_right.pos_m[1]) == pytest.approx(0.030)
    assert clamp_left.pos_m[1] - clamp_right.pos_m[1] == pytest.approx(0.20)         # gap unchanged
    assert clamp_left.pos_m[2] == pytest.approx(0.60)           # box bottom on surface 500


def test_dual_arm_place_uses_cached_surface_when_no_arg(monkeypatch):
    # locate_for_place() caches _last_surface; dual_arm_place() with no surface reuses it,
    # so an LLM calls locate_for_place() then dual_arm_place() without echoing the dict back.
    api, env, captured = _api(monkeypatch)
    api._last_surface = dict(_SURFACE)                          # as if locate_for_place() ran
    out = api.dual_arm_place()                                        # no surface arg
    assert out["ok"] is True
    assert out["landing_mm"] == pytest.approx([880.0, 30.0])    # landed ON the sensed table


def test_dual_arm_place_wider_than_table_bails(monkeypatch):
    api, env, captured = _api(monkeypatch)
    narrow = dict(_SURFACE, width_mm=200.0)                     # half=100 < box_half(135)+margin(20)
    out = api.dual_arm_place(surface=narrow)
    assert out == {"ok": False, "reason": "box_wider_than_table"}
    assert env.low_level.moves == [] and captured == []        # no motion


def test_dual_arm_place_deeper_than_table_bails(monkeypatch):
    api, env, captured = _api(monkeypatch)
    shallow = dict(_SURFACE, front_x_mm=800.0, back_x_mm=900.0)  # depth 100 < box_depth(120)+2*margin(40)
    out = api.dual_arm_place(surface=shallow)
    assert out == {"ok": False, "reason": "box_deeper_than_table"}
    assert env.low_level.moves == [] and captured == []


def test_surface_z_lands_box_bottom_on_surface(monkeypatch):
    api, env, captured = _api(monkeypatch)
    out = api.dual_arm_place(surface_z_mm=500.0)
    assert out["ok"] is True and out["surface_z_mm"] == 500.0
    # _apply_surface_z shifts by (surface_z - (top_z - height))/1000 = (500-600)/1000 = -0.10
    _, clamp_left = captured[0]
    assert clamp_left.pos_m[2] == pytest.approx(0.60)


def test_place_lower_streams_gap_locked_cartesian_waypoints(monkeypatch):
    # The lower must not be a single joint-space ramp (whose paddle gap only holds at the two
    # endpoints and bows OPEN mid-descent → the box slips). It interpolates each paddle TCP
    # linearly in CARTESIAN from the live held pose to the clamp target and STREAMS the knots,
    # so the commanded gap only ever TIGHTENS toward the squeeze (gap(f) = gap0 - f·squeeze)
    # and NEVER widens past the held gap.
    api, env, captured = _api(monkeypatch)
    env.cfg.place_squeeze_mm = 20.0
    env.cfg.place_lower_waypoints = 4
    out = api.dual_arm_place(surface_z_mm=500.0)                     # deterministic landing z
    assert out["ok"] is True
    assert env.low_level.streams, "lower should be streamed, not a single move"

    # captured[0:2] are the FINAL clamp IK targets (left, right); captured[2:8] are the 3
    # interior knots (f = 1/4, 2/4, 3/4), each left then right.
    (_, final_l), (_, final_r) = captured[0], captured[1]
    fl, fr = final_l.pos_m, final_r.pos_m
    held_l, held_r = (0.21, 0.15, 0.90), (0.39, -0.05, 0.90)          # live FK TCPs (see _fake_fk)
    held_gap = held_l[1] - held_r[1]                                  # 0.20 m
    for k, f in ((1, 0.25), (2, 0.50), (3, 0.75)):
        (_, knot_l), (_, knot_r) = captured[2 * k], captured[2 * k + 1]
        for j in range(3):                                            # Cartesian-linear per axis
            assert knot_l.pos_m[j] == pytest.approx(held_l[j] + f * (fl[j] - held_l[j]))
            assert knot_r.pos_m[j] == pytest.approx(held_r[j] + f * (fr[j] - held_r[j]))
        gap = knot_l.pos_m[1] - knot_r.pos_m[1]
        assert gap == pytest.approx(held_gap - f * 0.020)            # tightens by the squeeze
        assert gap <= held_gap + 1e-9                                # NEVER widens mid-descent


def test_place_lower_streams_from_post_lean_tcp_not_carried(monkeypatch):
    # REGRESSION (box hit the chest): the streamed lower must interpolate from the box's ACTUAL pose
    # AFTER the lifter lean, not the pre-lean carried held_tcp. Using the stale pre-lean TCP made the
    # first segment swing the box UP toward the carry height — into the chest. Here the lean drops the
    # flange z by 0.10 (fake fk: z = 0.90 - 0.2·lifter_pitch_1), so the interp START must be the LEANED
    # z (0.80), NOT the pre-lean 0.90.
    lean = {"lifter_pitch_1_joint": 0.5, "lifter_pitch_2_joint": -0.5, "lifter_pitch_3_joint": 0.0}
    api, env, captured = _api(monkeypatch, improves=True, q_lifter=lean)
    api._last_grasped_box = dict(_DET)
    env.cfg.place_squeeze_mm = 0.0
    env.cfg.place_lower_waypoints = 4
    out = api.dual_arm_place(surface_z_mm=500.0)              # clamp z = (700 mid) shifted to 0.60
    assert out["ok"] is True and env.low_level.streams
    (_, final_l) = captured[0]                               # first IK call = final clamp target (left)
    clamp_z = final_l.pos_m[2]
    assert clamp_z == pytest.approx(0.60)
    (_, knot_l) = captured[2]                                # first interior knot (f = 1/4), left
    # interp start is the POST-lean 0.80, not the carried 0.90 → z = 0.80 + 0.25·(0.60 - 0.80) = 0.75
    assert knot_l.pos_m[2] == pytest.approx(0.80 + 0.25 * (clamp_z - 0.80))
    assert knot_l.pos_m[2] != pytest.approx(0.90 + 0.25 * (clamp_z - 0.90))   # NOT the pre-lean start


def test_place_single_waypoint_falls_back_to_one_move(monkeypatch):
    # place_lower_waypoints=1 keeps the legacy single joint-space move (no streaming).
    api, env, captured = _api(monkeypatch)
    env.cfg.place_lower_waypoints = 1
    out = api.dual_arm_place(surface_z_mm=500.0)
    assert out["ok"] is True
    assert env.low_level.streams == []                               # no stream, single move used


def test_place_leans_lifter_before_lowering(monkeypatch):
    lean = {"lifter_pitch_1_joint": 0.5, "lifter_pitch_2_joint": -0.5, "lifter_pitch_3_joint": 0.0}
    api, env, _ = _api(monkeypatch, improves=True, q_lifter=lean)
    out = api.dual_arm_place(surface_z_mm=500.0)
    assert out["ok"] is True and out["leaned"] is True and out["lifter"] == lean
    # the FIRST commanded move is the lifter lean (holds arms + drives the lifter),
    # issued BEFORE the arm lower/release/raise moves.
    first = env.low_level.moves[0]
    for j in LIFTER_JOINTS:
        assert j in first
    assert first["lifter_pitch_1_joint"] == pytest.approx(0.5)
    assert len(env.low_level.moves) >= 2                    # lean + at least one arm move


def test_place_leans_first_then_places_no_lean_when_reachable(monkeypatch):
    # improves=False -> no lean move; the first move is an arm move (no lifter keys).
    api, env, _ = _api(monkeypatch, improves=False)
    out = api.dual_arm_place()
    assert out["ok"] is True and out["leaned"] is False
    first = env.low_level.moves[0]
    assert not any(j in first for j in LIFTER_JOINTS)       # arm-only move
