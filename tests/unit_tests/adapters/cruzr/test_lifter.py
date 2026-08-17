# coding: utf-8
import pytest

import jiuwensymbiosis.adapters.cruzr.geometry as lifter_mod
from jiuwensymbiosis.adapters.cruzr.geometry import GraspPlan
from jiuwensymbiosis.kinematics.ik import IKResult
from jiuwensymbiosis.adapters.cruzr.geometry import (
    LIFTER_JOINTS,
    level_config,
    search_lifter_for_box,
    search_lifter_for_place,
)
from jiuwensymbiosis.perception.object_geometry import ObjectGeometry3D

P1, P2, P3 = LIFTER_JOINTS


# ---- level_config: pure math (no URDF, fast) -------------------------------

_URDF = "/home/riemann/Robot/Cruzr_ws/cruzr_s2_description/cruzr_s2_description/urdf/cruzr_s2_v1/cruzr_s2_v1.urdf"


def test_lower_torso_lifter_descends_vertically():
    """lower_torso_lifter returns a level config that drops the shoulder ~dz in z
    with ~unchanged x (vertical descent), staying on the level manifold."""
    from pathlib import Path
    if not Path(_URDF).exists():
        pytest.skip("urdf not present")
    from jiuwensymbiosis.kinematics.urdf_chain import parse_chain
    from jiuwensymbiosis.adapters.cruzr.geometry import lower_torso_lifter, _shoulder_pos

    L = parse_chain(_URDF, "base_link", "L_sixforce_link")
    # Start from a realistic grasp-adapted (leaned) pose — the case dual_arm_place hits.
    # (From exactly upright, vertical descent is degenerate: z is stationary w.r.t.
    # pitch, so pitch-only cannot drop straight down and the helper returns None.)
    now = level_config(0.0, 0.9)
    new = lower_torso_lifter(L, "left", now, 0.0, 0.02)
    assert new is not None
    s0 = _shoulder_pos(L, "left", now, 0.0)
    s1 = _shoulder_pos(L, "left", new, 0.0)
    assert s1[2] == pytest.approx(s0[2] - 0.02, abs=2e-3)        # dropped ~2cm in z
    assert s1[0] == pytest.approx(s0[0], abs=2e-3)               # ~vertical (x held)
    assert new[P1] + new[P2] + new[P3] == pytest.approx(0.0, abs=1e-6)  # level manifold


def test_level_config_enforces_level_torso():
    c = level_config(0.9, 0.6)
    assert c is not None
    assert c[P1] == pytest.approx(0.9)
    assert c[P3] == pytest.approx(0.6)
    assert c[P2] == pytest.approx(-(0.9 + 0.6))  # torso level: p2 = -(p1+p3)


def test_level_config_rejects_out_of_limits():
    assert level_config(1.5, 1.5) is None   # p2 = -3.0 exceeds +-2.618
    assert level_config(2.0, 0.0) is None   # p1 beyond its own limit


# ---- search logic ----------------------------------------------------------
# The fast search PRUNES with the geometric proxy (_geo_reach_score: None ==
# beyond reach) and the table-floor FK (_arm_base_z), then RANKS survivors by a
# reduced-iteration dual-arm IK reach margin (solve_grasp, via _reach_score).
# Unit tests stub those three so they stay fast and deterministic with dummy
# chains. _box() is a real ObjectGeometry3D (plan_clamp_targets / ready_targets
# need only the box, not the chains). Geometry is used ONLY to prune — the
# winner is chosen by the IK reach margin (perr).

def _box():
    return ObjectGeometry3D(ok=True, reason="", center_mm=(350.0, 0.0, 700.0),
                          width_mm=270.0, height_mm=200.0, front_x_mm=290.0,
                          top_z_mm=800.0, n_points=5000, back_x_mm=410.0)


def _plan(ok, perr=0.01):
    ik = {a: IKResult(q={}, converged=ok, pos_err_m=perr, normal_err=0.0, iters=1)
          for a in ("left", "right")}
    return GraspPlan(ok, "" if ok else "ik_no_converge", 0.0, {}, {}, {}, ik)


def _is_current(q_fixed):
    return abs(q_fixed[P1]) < 1e-9 and abs(q_fixed[P3]) < 1e-9


def test_search_keeps_current_when_it_already_grasps(monkeypatch):
    # current config grasps -> do not move (improves=False), keep current lifter.
    # (Resolved by the precise IK at step 1, before any search.)
    monkeypatch.setattr(lifter_mod, "solve_grasp", lambda *a, **k: _plan(True, 0.01))
    current = {P1: 0.0, P2: 0.0, P3: 0.0}
    plan = search_lifter_for_box(_box(), object(), object(), current, waist_yaw=0.0)
    assert plan.found and plan.improves is False
    assert plan.q_lifter == current


def test_search_unreachable_when_nothing_grasps(monkeypatch):
    # current fails, geometry prunes everything -> unreachable.
    monkeypatch.setattr(lifter_mod, "solve_grasp", lambda *a, **k: _plan(False))
    monkeypatch.setattr(lifter_mod, "_geo_reach_score", lambda *a, **k: None)
    monkeypatch.setattr(lifter_mod, "_arm_base_z", lambda *a, **k: 0.9)
    current = {P1: 0.0, P2: 0.0, P3: 0.0}
    plan = search_lifter_for_box(_box(), object(), object(), current, step=0.4)
    assert plan.found is False
    assert plan.reason == "unreachable_any_lifter"
    assert plan.q_lifter is None


def test_search_ranks_by_ik_margin_not_geometry(monkeypatch):
    # Geometry does NOT rank: every cell is geo-feasible and table-safe, so the
    # winner is the one with the best (smallest) IK position error.
    target_p1, target_p3 = 0.6, 0.0
    monkeypatch.setattr(lifter_mod, "_geo_reach_score", lambda *a, **k: 0.0)  # none pruned
    monkeypatch.setattr(lifter_mod, "_arm_base_z", lambda *a, **k: 0.9)       # all safe

    def fake_solve(box, lc, rc, q_fixed, **k):
        if _is_current(q_fixed):
            return _plan(False)
        hit = abs(q_fixed[P1] - target_p1) < 1e-6 and abs(q_fixed[P3] - target_p3) < 1e-6
        return _plan(True, perr=0.002 if hit else 0.05)  # best margin at target

    monkeypatch.setattr(lifter_mod, "solve_grasp", fake_solve)
    current = {P1: 0.0, P2: 0.0, P3: 0.0}
    plan = search_lifter_for_box(_box(), object(), object(), current, step=0.2)
    assert plan.found and plan.improves
    assert plan.q_lifter[P1] == pytest.approx(target_p1)
    assert plan.q_lifter[P3] == pytest.approx(target_p3)
    assert plan.q_lifter[P2] == pytest.approx(-(target_p1 + target_p3))  # level


def test_search_geometric_prune_skips_out_of_reach_cells(monkeypatch):
    # Geometry prunes cells with p1 < 0 (out of reach); the only IK-reachable
    # cell sits among the survivors and is selected.
    monkeypatch.setattr(lifter_mod, "_geo_reach_score",
                        lambda chains, clamp, lc, w: None if lc[P1] < -1e-9 else 0.0)
    monkeypatch.setattr(lifter_mod, "_arm_base_z", lambda *a, **k: 0.9)

    def fake_solve(box, lc, rc, q_fixed, **k):
        if _is_current(q_fixed):
            return _plan(False)
        if abs(q_fixed[P1] - 0.6) < 1e-6 and abs(q_fixed[P3]) < 1e-6:
            return _plan(True, 0.002)
        return _plan(False)

    monkeypatch.setattr(lifter_mod, "solve_grasp", fake_solve)
    current = {P1: 0.0, P2: 0.0, P3: 0.0}
    plan = search_lifter_for_box(_box(), object(), object(), current, step=0.3)
    assert plan.found and plan.improves
    assert plan.q_lifter[P1] == pytest.approx(0.6)
    assert plan.q_lifter[P3] == pytest.approx(0.0)


# ---- table-floor guard: reject configs that drag held-ready arms below table --

def test_search_rejects_low_config_that_would_hit_table(monkeypatch):
    """A low/leaned config with the BEST IK reach must be rejected when it would
    drag the held ready arms below the table; the search picks a safe one."""
    # box: top_z 800, height 200 -> table_z 0.60, floor 0.65 (clearance 0.05); ready_z 0.80.
    # _arm_base_z = 1.0 - 0.5*p1  =>  held_z = 0.80 - 0.5*p1  => safe iff p1 <= 0.30.
    monkeypatch.setattr(lifter_mod, "_geo_reach_score", lambda *a, **k: 0.0)  # none pruned
    monkeypatch.setattr(lifter_mod, "_arm_base_z",
                        lambda chain, lc, w: 1.0 - 0.5 * lc[P1])  # held below floor for p1>0.3

    def fake_solve(box, lc, rc, q_fixed, **k):
        if _is_current(q_fixed):
            return _plan(False)
        # higher p1 = better (smaller) IK error -> low/unsafe cells look best
        return _plan(True, perr=max(0.001, 0.05 - 0.03 * q_fixed[P1]))

    monkeypatch.setattr(lifter_mod, "solve_grasp", fake_solve)
    current = {P1: 0.0, P2: 0.0, P3: 0.0}
    plan = search_lifter_for_box(_box(), object(), object(), current, step=0.3)
    assert plan.found and plan.improves
    assert plan.q_lifter[P1] == pytest.approx(0.3)  # best reach among table-SAFE cells


def test_search_no_safe_lifter_when_only_reachable_config_is_too_low(monkeypatch):
    """If every geometrically reachable config is rejected by the table-floor
    guard, the search reports no_safe_lifter (distinct from unreachable)."""
    monkeypatch.setattr(lifter_mod, "_geo_reach_score",
                        lambda chains, clamp, lc, w: 0.0 if lc[P1] > 0.3 + 1e-9 else None)
    monkeypatch.setattr(lifter_mod, "_arm_base_z",
                        lambda chain, lc, w: 1.0 - 0.5 * lc[P1])  # p1>0.3 -> below floor
    monkeypatch.setattr(lifter_mod, "solve_grasp", lambda *a, **k: _plan(False))
    current = {P1: 0.0, P2: 0.0, P3: 0.0}
    plan = search_lifter_for_box(_box(), object(), object(), current, step=0.3)
    assert plan.found is False
    assert plan.reason == "no_safe_lifter"
    assert plan.q_lifter is None


# ---- search_lifter_for_place: SMALLEST lean that reaches, capped, NO floor guard --
# Mirrors the grasp search but takes clamp targets directly and picks the reachable cell
# with the MINIMAL lift change (least |Δp1|+|Δp3|), never leaning more than max_lean_rad
# (the arms position the box; the lifter only nudges). Tests stub lifter_mod.solve_arm_ik
# (called per arm as solve_arm_ik(chain, q_fixed, arm, tgt)) and lifter_mod._geo_reach_score.

_CLAMP = {"left": object(), "right": object()}


def _ik(converged, perr=0.01):
    return IKResult(q={}, converged=converged, pos_err_m=perr, normal_err=0.0, iters=1)


def test_place_search_keeps_current_when_reachable(monkeypatch):
    # current lifter already reaches the place targets -> improves=False, keep it.
    monkeypatch.setattr(lifter_mod, "solve_arm_ik", lambda *a, **k: _ik(True, 0.01))
    current = {P1: 0.0, P2: 0.0, P3: 0.0}
    plan = search_lifter_for_place(_CLAMP, object(), object(), current, waist_yaw=0.0)
    assert plan.found and plan.improves is False
    assert plan.q_lifter == current


def test_place_search_picks_minimal_lean(monkeypatch):
    # current fails; among reachable cells the SMALLEST lean wins (arms do the rest),
    # NOT the best IK margin. Here every cell with p1 >= 0.15 reaches -> the least of
    # those is (p1=0.15, p3=0).
    monkeypatch.setattr(lifter_mod, "_geo_reach_score", lambda *a, **k: 0.0)  # none pruned

    def fake_ik(chain, q_fixed, arm, tgt, **k):
        reach = q_fixed[P1] >= 0.15 - 1e-9
        return _ik(reach, 0.01 if reach else 0.5)

    monkeypatch.setattr(lifter_mod, "solve_arm_ik", fake_ik)
    current = {P1: 0.0, P2: 0.0, P3: 0.0}
    plan = search_lifter_for_place(_CLAMP, object(), object(), current)  # default cap 0.8, step 0.15
    assert plan.found and plan.improves
    assert plan.q_lifter[P1] == pytest.approx(0.15)               # smallest reaching lean
    assert plan.q_lifter[P3] == pytest.approx(0.0)
    assert plan.q_lifter[P2] == pytest.approx(-0.15)             # level manifold


def test_place_search_respects_max_lean_cap(monkeypatch):
    # Only a lean beyond the cap would reach -> unreachable (never leans into the table).
    monkeypatch.setattr(lifter_mod, "_geo_reach_score", lambda *a, **k: 0.0)

    def fake_ik(chain, q_fixed, arm, tgt, **k):
        reach = q_fixed[P1] >= 0.6 - 1e-9                         # beyond the 0.35 cap
        return _ik(reach, 0.01 if reach else 0.5)

    monkeypatch.setattr(lifter_mod, "solve_arm_ik", fake_ik)
    current = {P1: 0.0, P2: 0.0, P3: 0.0}
    plan = search_lifter_for_place(_CLAMP, object(), object(), current, max_lean_rad=0.35, step=0.15)
    assert plan.found is False and plan.reason == "place_unreachable_any_lifter"


def test_place_search_unreachable_when_nothing_reaches(monkeypatch):
    # current fails and geometry prunes everything -> place_unreachable_any_lifter.
    monkeypatch.setattr(lifter_mod, "solve_arm_ik", lambda *a, **k: _ik(False, 0.5))
    monkeypatch.setattr(lifter_mod, "_geo_reach_score", lambda *a, **k: None)
    current = {P1: 0.0, P2: 0.0, P3: 0.0}
    plan = search_lifter_for_place(_CLAMP, object(), object(), current, step=0.4)
    assert plan.found is False
    assert plan.reason == "place_unreachable_any_lifter"
    assert plan.q_lifter is None


def test_place_search_no_floor_guard_accepts_forward_lean(monkeypatch):
    """Unlike the grasp search, place has NO ready-arm table-floor guard: a forward lean
    the arms need to reach a low table is accepted purely on reachability (no _arm_base_z
    patch needed — place never calls it). Only the deepest allowed lean reaches here."""
    monkeypatch.setattr(lifter_mod, "_geo_reach_score", lambda *a, **k: 0.0)  # none pruned

    def fake_ik(chain, q_fixed, arm, tgt, **k):
        reach = q_fixed[P1] >= 0.30 - 1e-9                        # only the deepest in-cap lean reaches
        return _ik(reach, 0.01 if reach else 0.5)

    monkeypatch.setattr(lifter_mod, "solve_arm_ik", fake_ik)
    current = {P1: 0.0, P2: 0.0, P3: 0.0}
    plan = search_lifter_for_place(_CLAMP, object(), object(), current)  # default cap 0.8, step 0.15
    assert plan.found and plan.improves
    assert plan.q_lifter[P1] == pytest.approx(0.30)               # forward lean accepted (no floor guard)
    assert plan.q_lifter[P3] == pytest.approx(0.0)                # minimal p3
