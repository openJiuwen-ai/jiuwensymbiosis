# coding: utf-8
import numpy as np

import jiuwensymbiosis.adapters.cruzr.geometry as lifter_mod
from jiuwensymbiosis.adapters.cruzr.api import CruzrApi
from jiuwensymbiosis.adapters.cruzr.geometry import (
    ArmTarget, GraspPlan, ARM_JOINTS,
)
from jiuwensymbiosis.kinematics.ik import IKResult
from jiuwensymbiosis.adapters.cruzr.geometry import LIFTER_JOINTS, LifterPlan


def _no_move_lifter(box, lc, rc, current_lifter, waist_yaw, **k):
    """Stub search: current pose already grasps -> don't move the lifter."""
    return LifterPlan(found=True, q_lifter=dict(current_lifter), score=-0.01,
                      improves=False, reason="")


def _no_move_place(clamp, lc, rc, current_lifter, waist_yaw, **k):
    """Stub place search: current pose already reaches -> don't lean the lifter."""
    return LifterPlan(found=True, q_lifter=dict(current_lifter), score=-0.01,
                      improves=False, reason="")


class _LL:
    def __init__(self): self.moves = []
    def grab_frames(self, camera="waist"):
        h, w = 360, 640
        depth = np.zeros((h, w), dtype=np.float32); depth[150:210, 290:350] = 0.5
        K = np.array([[345.0, 0, 320.0], [0, 345.0, 180.0], [0, 0, 1]])
        return np.zeros((h, w, 3), np.uint8), depth, K, np.eye(4)
    def get_joint_positions(self):
        return {"lifter_pitch_1_joint": 0.0, "lifter_pitch_2_joint": 0.0,
                "lifter_pitch_3_joint": 0.0, "waist_yaw_joint": 0.0}
    def move_joints_blocking(self, targets, **kw):
        self.moves.append(dict(targets))
        if not hasattr(self, "move_ramps"): self.move_ramps = []
        self.move_ramps.append(kw.get("ramp_duration_s"))
        return {"ok": True, "targets": targets}
    def stream_joint_trajectory(self, waypoints, **kw):
        self.moves.append(dict(waypoints[-1]))   # gap-locked lower: record final knot
        return {"ok": True, "waypoints": len(waypoints)}
    def set_lifter(self, q_lifter):
        if not hasattr(self, "lifter_calls"): self.lifter_calls = []
        self.lifter_calls.append(dict(q_lifter)); return {"ok": True}
    def read_hand_ft(self, arm):
        return {"ok": True, "fmag": 0.0}  # no contact
    def turn_waist_blocking(self, target_rad, *, hold, waist_joint="waist_yaw_joint", step_rad=None):
        if not hasattr(self, "turn_calls"): self.turn_calls = []
        self.turn_calls.append({"target": target_rad, "hold": dict(hold)})
        return {"ok": True, "readback": {waist_joint: target_rad}}


class _Env:
    def __init__(self): self.low_level = _LL()
    class cfg:
        default_arm = "left"
        urdf_path = "/home/riemann/Robot/Cruzr_ws/cruzr_s2_description/cruzr_s2_description/urdf/cruzr_s2_v1/cruzr_s2_v1.urdf"
        left_arm_leaf = "L_sixforce_link"
        right_arm_leaf = "R_sixforce_link"
        palm_normal_local = (0.0, 0.0, 1.0)
        contact_force_threshold_n = 5.0
        grasp_inset_mm = 3.0
        place_edge_margin_mm = 20.0
        place_max_lift_lean_rad = 0.35
        waist_yaw_joint = "waist_yaw_joint"
        urdf_package_dir = "/home/riemann/Robot/Cruzr_ws/cruzr_s2_description"
        arm_transit_ramp_duration_s = 0.6   # non-contact grasp moves (ready/descend) use this fast ramp
        arm_contact_ramp_duration_s = 0.7   # box-contact moves (clamp/place-lower) use this dedicated ramp


# A valid cached locate_for_grasp result. dual_arm_grasp() no longer detects — it
# consumes the most recent detection (or an explicit box arg), so the tests seed
# this on the api the way a prior locate_for_grasp call would.
_DET = {
    "ok": True, "reason": "", "object": "box",
    "center_mm": [350.0, 0.0, 700.0], "width_mm": 270.0, "height_mm": 200.0,
    "front_x_mm": 290.0, "back_x_mm": 410.0, "top_z_mm": 800.0, "n_points": 5000,
}


def _api(monkeypatch):
    env = _Env()
    api = CruzrApi(env)
    mask = np.zeros((360, 640), dtype=bool); mask[150:210, 290:350] = True
    monkeypatch.setattr(api, "_ensure_detector", lambda: None)
    api._seg_fn = lambda rgb, text_prompt="box": [
        {"score": 0.9, "label": "box", "box": [290, 150, 350, 210], "mask": mask}]
    api._last_detection = dict(_DET)  # as if locate_for_grasp ran first
    return api, env


def _make_converged_ik(arm):
    """Return a converged IKResult with zeros for all joints of the given arm."""
    return IKResult(
        q={j: 0.0 for j in ARM_JOINTS[arm]},
        converged=True,
        pos_err_m=0.001,
        normal_err=0.001,
        iters=5,
    )


def _at(arm, y, z=0.6):
    return ArmTarget(arm, (0.40, y, z), (1, 0, 0), (0, 1, 0), (-0.09 if arm == "left" else 0.09, 0, 0))


def _fake_solve_grasp(*args, **kwargs):
    """Mock solve_grasp that always returns a converged GraspPlan."""
    approach = {"left": _at("left", 0.20, 0.7), "right": _at("right", -0.20, 0.7)}
    descend = {"left": _at("left", 0.20), "right": _at("right", -0.20)}
    clamp = {"left": _at("left", 0.135), "right": _at("right", -0.135)}
    ik = {"left": _make_converged_ik("left"), "right": _make_converged_ik("right")}
    return GraspPlan(ok=True, reason="", lifter_center_z_mm=600.0,
                     approach=approach, descend=descend, clamp=clamp, ik=ik)


def _fake_solve_arm_ik(chain, q_fixed, arm, tgt, **kwargs):
    """Mock solve_arm_ik (pre-grasp) that always returns a converged IKResult."""
    return _make_converged_ik(arm)


def _fake_solve_arm_ik_nonzero(chain, q_fixed, arm, tgt, **kwargs):
    """Converged IK with NON-zero joints, so a place move is observably not homed-to-zero."""
    return IKResult(
        q={j: 0.1 for j in ARM_JOINTS[arm]},
        converged=True, pos_err_m=0.001, normal_err=0.001, iters=5,
    )


def test_dual_arm_grasp_no_joint_state(monkeypatch):
    """Guard: if get_joint_positions() returns {}, dual_arm_grasp must return no_joint_state."""
    from pathlib import Path
    import pytest
    if not Path(_Env.cfg.urdf_path).exists():
        pytest.skip("urdf not present")

    api, env = _api(monkeypatch)
    # Override get_joint_positions to return empty dict (subprocess backend behaviour)
    env.low_level.get_joint_positions = lambda: {}

    out = api.dual_arm_grasp()

    assert out["ok"] is False
    assert out["reason"] == "no_joint_state"


def test_dual_arm_grasp_aborts_lift_without_contact(monkeypatch):
    from pathlib import Path
    import pytest
    if not Path(_Env.cfg.urdf_path).exists():
        pytest.skip("urdf not present")

    # Monkeypatch solve_grasp and solve_arm_ik so IK always converges,
    # allowing the test to reach the FT-verify / no-contact abort path
    # WITH the plan.ok safety gate intact.
    import jiuwensymbiosis.adapters.cruzr.geometry as _gp_mod
    monkeypatch.setattr(_gp_mod, "solve_grasp", _fake_solve_grasp)
    monkeypatch.setattr(_gp_mod, "solve_arm_ik", _fake_solve_arm_ik)
    monkeypatch.setattr(lifter_mod, "search_lifter_for_box", _no_move_lifter)

    api, env = _api(monkeypatch)
    out = api.dual_arm_grasp()

    # FT returns fmag=0.0 (no contact) => must abort before lift
    # reason=="no_contact" proves abort-before-lift held even with converged plan
    assert out["ok"] is False
    assert out["reason"] == "no_contact"


def test_grasp_transit_fast_clamp_uses_contact_ramp(monkeypatch):
    """ready + descend (non-contact positioning) run on the fast arm_transit ramp; the clamp runs on
    the dedicated arm_contact ramp — a separate knob so the squeeze speed can be tuned against
    deform / contact-force accuracy without touching the transit or global ramps."""
    from pathlib import Path
    import pytest
    if not Path(_Env.cfg.urdf_path).exists():
        pytest.skip("urdf not present")

    import jiuwensymbiosis.adapters.cruzr.geometry as _gp_mod
    monkeypatch.setattr(_gp_mod, "solve_grasp", _fake_solve_grasp)
    monkeypatch.setattr(_gp_mod, "solve_arm_ik", _fake_solve_arm_ik)
    monkeypatch.setattr(lifter_mod, "search_lifter_for_box", _no_move_lifter)

    api, env = _api(monkeypatch)
    api.dual_arm_grasp()   # runs ready, descend, clamp (then aborts at the no-contact FT read)

    ramps = env.low_level.move_ramps
    assert len(ramps) == 3                 # ready, descend, clamp — three arm moves, no lifter lean
    assert ramps[0] == 0.6                  # ready   → fast transit ramp
    assert ramps[1] == 0.6                  # descend → fast transit ramp
    assert ramps[2] == 0.7                  # clamp   → dedicated contact ramp (not transit, not global)


def test_home_arms_moves_both_to_zero(monkeypatch):
    """home_arms (clear path) issues ONE combined command with all 14 arm joints at 0."""
    api, env = _api(monkeypatch)
    out = api.home_arms()
    assert out["ok"]
    assert env.low_level.moves, "expected a move command"
    last = env.low_level.moves[-1]
    assert len(last) == 14  # 7 joints per arm, both arms in one command
    assert all(v == 0.0 for v in last.values())


def test_home_arms_via_clearance_pose_when_configured(monkeypatch):
    """With cfg.home_clearance_arm_q set, home_arms abducts the arms OUT to the sides FIRST,
    then descends to 0 (two moves) — never a direct sweep across the torso."""
    import pytest
    api, env = _api(monkeypatch)
    monkeypatch.setattr(env.cfg, "home_clearance_arm_q",
                        {"L_shoulder_roll_joint": -0.7, "R_shoulder_roll_joint": -0.7}, raising=False)
    monkeypatch.setattr(api, "_arm_home_path_collides", lambda *a, **k: False)  # every leg clear
    out = api.home_arms()
    assert out["ok"]
    assert len(env.low_level.moves) == 2                        # clearance waypoint, then 0
    clearance, final = env.low_level.moves[0], env.low_level.moves[1]
    assert clearance["L_shoulder_roll_joint"] == pytest.approx(-0.7)   # arms abducted out
    assert clearance["R_shoulder_roll_joint"] == pytest.approx(-0.7)
    assert clearance["L_shoulder_pitch_joint"] == pytest.approx(0.0)   # other joints at 0
    assert len(final) == 14 and all(v == 0.0 for v in final.values())  # then fully home


def test_home_arms_refuses_when_no_safe_path(monkeypatch):
    """home_arms is self-collision-checked (not just home_safely): with no collision-free
    path it refuses rather than sweeping both arms straight through the torso."""
    from pathlib import Path
    import pytest
    if not Path(_Env.cfg.urdf_path).exists():
        pytest.skip("urdf not present")
    api, env = _api(monkeypatch)
    env.low_level.get_joint_positions = lambda: {
        "lifter_pitch_1_joint": 0.0, "lifter_pitch_2_joint": 0.0,
        "lifter_pitch_3_joint": 0.0, "waist_yaw_joint": 0.0,
        "L_shoulder_pitch_joint": 0.5,
    }
    monkeypatch.setattr(api, "_arm_home_path_collides", lambda *a, **k: True)  # every path collides
    out = api.home_arms()
    assert out == {"ok": False, "reason": "home_path_self_collision"}
    assert env.low_level.moves == []  # never swept the arms


def test_dual_arm_grasp_passes_grasp_inset_to_solve_grasp(monkeypatch):
    """dual_arm_grasp drives the clamp depth from cfg.grasp_inset_mm (gentler grip, less crush)."""
    from pathlib import Path
    import pytest
    if not Path(_Env.cfg.urdf_path).exists():
        pytest.skip("urdf not present")
    import jiuwensymbiosis.adapters.cruzr.geometry as _gp_mod
    captured = {}

    def _capture(box, left, right, q_fixed, **kwargs):
        captured["inset_mm"] = kwargs.get("inset_mm")
        return _fake_solve_grasp()

    monkeypatch.setattr(_gp_mod, "solve_grasp", _capture)
    monkeypatch.setattr(_gp_mod, "solve_arm_ik", _fake_solve_arm_ik)
    monkeypatch.setattr(lifter_mod, "search_lifter_for_box", _no_move_lifter)
    api, env = _api(monkeypatch)
    env.low_level.read_hand_ft = lambda arm: {"ok": True, "fmag": 10.0}  # contact
    api.dual_arm_grasp()
    assert captured["inset_mm"] == env.cfg.grasp_inset_mm == 3.0


def test_dual_arm_grasp_clamps_and_verifies_contact_no_lift(monkeypatch):
    """On FT contact grasp clamps to the plan pose and returns the FT reading.
    It does NOT lift — the pick-up/clearance lift is delegated to lift_to_clearance."""
    from pathlib import Path
    import pytest
    if not Path(_Env.cfg.urdf_path).exists():
        pytest.skip("urdf not present")
    import jiuwensymbiosis.adapters.cruzr.geometry as _gp_mod
    monkeypatch.setattr(_gp_mod, "solve_grasp", _fake_solve_grasp)
    monkeypatch.setattr(_gp_mod, "solve_arm_ik", _fake_solve_arm_ik)
    monkeypatch.setattr(lifter_mod, "search_lifter_for_box", _no_move_lifter)

    api, env = _api(monkeypatch)
    env.low_level.read_hand_ft = lambda arm: {"ok": True, "fmag": 10.0}  # contact
    out = api.dual_arm_grasp()

    assert out["ok"] is True
    assert all(out["ft"][a]["fmag"] >= _Env.cfg.contact_force_threshold_n for a in ("left", "right"))
    # last move is the clamp pose (plan.ik, zeros here), NOT a shoulder-pitch lift
    last = env.low_level.moves[-1]
    assert last["L_shoulder_pitch_joint"] == 0.0
    assert last["R_shoulder_pitch_joint"] == 0.0


def test_dual_arm_grasp_unreachable_any_lifter(monkeypatch):
    from pathlib import Path
    import pytest
    if not Path(_Env.cfg.urdf_path).exists():
        pytest.skip("urdf not present")

    def _unreachable(box, lc, rc, current_lifter, waist_yaw, **k):
        return LifterPlan(found=False, q_lifter=None, score=float("-inf"),
                          improves=False, reason="unreachable_any_lifter")

    monkeypatch.setattr(lifter_mod, "search_lifter_for_box", _unreachable)
    api, env = _api(monkeypatch)
    out = api.dual_arm_grasp()
    assert out["ok"] is False
    assert out["reason"] == "unreachable_any_lifter"
    assert not getattr(env.low_level, "lifter_calls", [])  # lifter NOT moved


def test_dual_arm_grasp_adjusts_lifter_when_it_helps(monkeypatch):
    from pathlib import Path
    import pytest
    if not Path(_Env.cfg.urdf_path).exists():
        pytest.skip("urdf not present")

    target = {LIFTER_JOINTS[0]: -0.3, LIFTER_JOINTS[1]: -0.3, LIFTER_JOINTS[2]: 0.6}

    def _improves(box, lc, rc, current_lifter, waist_yaw, **k):
        return LifterPlan(found=True, q_lifter=dict(target), score=-0.005,
                          improves=True, reason="")

    import jiuwensymbiosis.adapters.cruzr.geometry as _gp_mod
    monkeypatch.setattr(_gp_mod, "solve_grasp", _fake_solve_grasp)
    monkeypatch.setattr(_gp_mod, "solve_arm_ik", _fake_solve_arm_ik)
    monkeypatch.setattr(lifter_mod, "search_lifter_for_box", _improves)

    api, env = _api(monkeypatch)
    env.low_level.read_hand_ft = lambda arm: {"ok": True, "fmag": 10.0}  # contact
    out = api.dual_arm_grasp()
    assert out["ok"] is True
    assert getattr(env.low_level, "lifter_calls", []) == [target]  # lifter moved to best


def test_home_safely_skips_when_already_home(monkeypatch):
    """Already upright (lifter 0) with arms at 0 -> do nothing (no weird raise/lower)."""
    api, env = _api(monkeypatch)  # _LL reports lifter 0; arm joints default to 0
    out = api._home_safely()
    assert out == {"ok": True, "skipped": "already_home"}
    assert env.low_level.moves == []                       # no arm motion
    assert not getattr(env.low_level, "lifter_calls", [])  # lifter not touched


def test_home_safely_only_homes_arms_when_upright_but_arms_out(monkeypatch):
    """Body upright but an arm is off-zero -> just home the arms; no ready-raise,
    no lifter command (there is no leaned body to straighten)."""
    api, env = _api(monkeypatch)
    env.low_level.get_joint_positions = lambda: {
        "lifter_pitch_1_joint": 0.0, "lifter_pitch_2_joint": 0.0,
        "lifter_pitch_3_joint": 0.0, "waist_yaw_joint": 0.0,
        "L_shoulder_pitch_joint": 0.5,  # an arm is not home
    }
    out = api._home_safely()
    assert out == {"ok": True}
    assert len(env.low_level.moves) == 1                   # exactly one move: home_arms
    assert all(v == 0.0 for v in env.low_level.moves[-1].values())
    assert not getattr(env.low_level, "lifter_calls", [])  # lifter NOT straightened


def test_home_safely_full_sequence_when_leaned(monkeypatch):
    """Body leaned forward (lifter != 0) -> straighten the lifter then home."""
    from pathlib import Path
    import pytest
    if not Path(_Env.cfg.urdf_path).exists():
        pytest.skip("urdf not present")
    api, env = _api(monkeypatch)
    env.low_level.get_joint_positions = lambda: {
        "lifter_pitch_1_joint": 0.5, "lifter_pitch_2_joint": -0.5,
        "lifter_pitch_3_joint": 0.0, "waist_yaw_joint": 0.0,
    }
    out = api._home_safely()
    assert out == {"ok": True}
    assert getattr(env.low_level, "lifter_calls", [])      # lifter straightened
    assert all(v == 0.0 for v in env.low_level.lifter_calls[-1].values())
    assert env.low_level.moves and all(v == 0.0 for v in env.low_level.moves[-1].values())  # ended homed


def test_dual_arm_place_does_not_return_to_zero(monkeypatch):
    """dual_arm_place places + raises clear but delegates the return-to-zero to
    home_safely: it must NOT straighten the lifter or home the arms."""
    from pathlib import Path
    import pytest
    if not Path(_Env.cfg.urdf_path).exists():
        pytest.skip("urdf not present")
    # Force the place lower/open/raise IK to converge to a NON-zero pose so the
    # move is deterministically observable (dual_arm_place no longer ends on a
    # box-independent ready pose, so we can't rely on real IK converging here).
    import jiuwensymbiosis.adapters.cruzr.geometry as _gp_mod
    monkeypatch.setattr(_gp_mod, "solve_arm_ik", _fake_solve_arm_ik_nonzero)
    # place now searches a place lifter (its own solve_arm_ik bound in lifter_mod);
    # stub it to "already reaches" so no lean move runs and the test stays hardware-free.
    monkeypatch.setattr(lifter_mod, "search_lifter_for_place", _no_move_place)
    api, env = _api(monkeypatch)               # fixture seeds a cached grasp box
    api._last_grasped_box = dict(_DET)
    out = api.dual_arm_place()
    assert out["ok"] is True                   # return now carries lean/surface telemetry
    # dual_arm_place may lower the torso (a NON-zero set_lifter descent) but must NOT
    # straighten the lifter to zero or home the arms — that is home_safely's job.
    lifter_calls = getattr(env.low_level, "lifter_calls", [])
    assert not any(all(v == 0.0 for v in c.values()) for c in lifter_calls)  # no return-to-zero
    assert env.low_level.moves                               # it did move (place/raise)
    assert not all(v == 0.0 for v in env.low_level.moves[-1].values())  # arms NOT homed to 0


def test_dual_arm_grasp_caches_box_for_place(monkeypatch):
    """On a successful grasp, the detected box is cached on the api so dual_arm_place()
    can later be called WITHOUT the LLM echoing the geometry back."""
    from pathlib import Path
    import pytest
    if not Path(_Env.cfg.urdf_path).exists():
        pytest.skip("urdf not present")
    import jiuwensymbiosis.adapters.cruzr.geometry as _gp_mod
    monkeypatch.setattr(_gp_mod, "solve_grasp", _fake_solve_grasp)
    monkeypatch.setattr(_gp_mod, "solve_arm_ik", _fake_solve_arm_ik)
    monkeypatch.setattr(lifter_mod, "search_lifter_for_box", _no_move_lifter)

    api, env = _api(monkeypatch)
    assert api._last_grasped_box is None
    env.low_level.read_hand_ft = lambda arm: {"ok": True, "fmag": 10.0}  # contact
    out = api.dual_arm_grasp()

    assert out["ok"] is True
    assert api._last_grasped_box is out["box"]      # cached the same geometry
    assert "center_mm" in api._last_grasped_box


def test_dual_arm_place_without_box_and_no_cache_errors(monkeypatch):
    """dual_arm_place() with neither an explicit box nor a cached grasp returns a
    clean error BEFORE any motion (no urdf / hardware needed)."""
    api, env = _api(monkeypatch)
    assert api._last_grasped_box is None
    out = api.dual_arm_place()                            # no arg, nothing grasped yet
    assert out == {"ok": False, "reason": "no_box_to_place"}
    out2 = api.dual_arm_place(target="")                     # empty string from a flailing LLM
    assert out2 == {"ok": False, "reason": "no_box_to_place"}
    assert env.low_level.moves == []                 # never moved the arms


def test_home_safely_neutralizes_rotated_waist(monkeypatch):
    """Upright + arms home but waist rotated -> turn waist to 0, then home arms."""
    api, env = _api(monkeypatch)
    env.low_level.get_joint_positions = lambda: {
        "lifter_pitch_1_joint": 0.0, "lifter_pitch_2_joint": 0.0,
        "lifter_pitch_3_joint": 0.0, "waist_yaw_joint": 0.5,   # waist off-zero
    }
    out = api._home_safely()
    assert out == {"ok": True}
    turn_calls = getattr(env.low_level, "turn_calls", [])
    assert turn_calls and turn_calls[-1]["target"] == 0.0       # waist neutralized
    assert env.low_level.moves and all(v == 0.0 for v in env.low_level.moves[-1].values())  # arms homed


def test_home_safely_skip_requires_waist_home(monkeypatch):
    """Already home INCLUDING waist -> skip, no waist turn."""
    api, env = _api(monkeypatch)   # fake reports lifter/arms/waist all 0
    out = api._home_safely()
    assert out == {"ok": True, "skipped": "already_home"}
    assert not getattr(env.low_level, "turn_calls", [])         # no waist turn


def test_dual_arm_grasp_final_solve_checks_collision(monkeypatch):
    """dual_arm_grasp's final solve_grasp runs with self-collision checking on."""
    from pathlib import Path
    import pytest
    if not Path(_Env.cfg.urdf_path).exists():
        pytest.skip("urdf not present")
    import jiuwensymbiosis.adapters.cruzr.geometry as _gp_mod
    seen = {}

    def _capture(box, left, right, q_fixed, **kwargs):
        seen["check_collision"] = kwargs.get("check_collision")
        seen["package_dir"] = kwargs.get("package_dir")
        return _fake_solve_grasp()

    monkeypatch.setattr(_gp_mod, "solve_grasp", _capture)
    monkeypatch.setattr(_gp_mod, "solve_arm_ik", _fake_solve_arm_ik)
    monkeypatch.setattr(lifter_mod, "search_lifter_for_box", _no_move_lifter)
    api, env = _api(monkeypatch)
    env.low_level.read_hand_ft = lambda arm: {"ok": True, "fmag": 10.0}
    api.dual_arm_grasp()
    assert seen["check_collision"] is True
    assert seen["package_dir"] == env.cfg.urdf_package_dir


def test_home_safely_recovers_by_homing_one_arm_at_a_time(monkeypatch):
    """When homing both arms together self-collides (forearms cross in front), home_safely
    recovers by homing one arm at a time rather than refusing."""
    from pathlib import Path
    import pytest
    if not Path(_Env.cfg.urdf_path).exists():
        pytest.skip("urdf not present")
    api, env = _api(monkeypatch)
    env.low_level.get_joint_positions = lambda: {
        "lifter_pitch_1_joint": 0.0, "lifter_pitch_2_joint": 0.0,
        "lifter_pitch_3_joint": 0.0, "waist_yaw_joint": 0.0,
        "L_shoulder_pitch_joint": 0.5, "R_shoulder_pitch_joint": 0.5,  # BOTH arms out
    }

    def _both_move(qf, start, goal, n=8):
        # only a simultaneous both-arm sweep self-collides; moving one arm at a time is clear
        moved = {a: any(abs(start.get(j, 0.0) - goal.get(j, 0.0)) > 1e-9 for j in ARM_JOINTS[a])
                 for a in ("left", "right")}
        return moved["left"] and moved["right"]

    monkeypatch.setattr(api, "_arm_home_path_collides", _both_move)
    out = api._home_safely()
    assert out == {"ok": True}
    assert len(env.low_level.moves) >= 2                              # one arm, then the other
    assert all(v == 0.0 for v in env.low_level.moves[-1].values())    # ended at zero
    assert any(not all(v == 0.0 for v in m.values()) for m in env.low_level.moves[:-1])  # staged


def test_home_safely_reroutes_via_ready_on_collision(monkeypatch):
    """If direct AND one-arm-at-a-time both self-collide, home_safely goes via the ready pose."""
    from pathlib import Path
    import pytest
    if not Path(_Env.cfg.urdf_path).exists():
        pytest.skip("urdf not present")
    api, env = _api(monkeypatch)
    env.low_level.get_joint_positions = lambda: {
        "lifter_pitch_1_joint": 0.0, "lifter_pitch_2_joint": 0.0,
        "lifter_pitch_3_joint": 0.0, "waist_yaw_joint": 0.0,
        "L_shoulder_pitch_joint": 0.5,  # arms not home -> would home
    }
    # escalation order: direct(collide), right-first(collide), left-first(collide),
    # then ready leg1(clear), ready leg2(clear) -> ready reroute wins
    calls = iter([True, True, True, False, False])
    monkeypatch.setattr(api, "_arm_home_path_collides", lambda *a, **k: next(calls, False))
    out = api._home_safely()
    assert out["ok"] is True
    # a ready-pose move happened before the final home-to-zero
    assert any(not all(v == 0.0 for v in m.values()) for m in env.low_level.moves[:-1])


def test_home_safely_refuses_when_no_safe_path(monkeypatch):
    """If even the ready route self-collides, home_safely refuses rather than sweeping into itself."""
    from pathlib import Path
    import pytest
    if not Path(_Env.cfg.urdf_path).exists():
        pytest.skip("urdf not present")
    api, env = _api(monkeypatch)
    env.low_level.get_joint_positions = lambda: {
        "lifter_pitch_1_joint": 0.0, "lifter_pitch_2_joint": 0.0,
        "lifter_pitch_3_joint": 0.0, "waist_yaw_joint": 0.0,
        "L_shoulder_pitch_joint": 0.5,
    }
    monkeypatch.setattr(api, "_arm_home_path_collides", lambda *a, **k: True)  # every path collides
    out = api._home_safely()
    assert out == {"ok": False, "reason": "home_path_self_collision"}


def test_home_safely_refuses_on_partial_ready_ik(monkeypatch):
    """Direct path collides and the ready-pose IK converges for only ONE arm ->
    the reroute would command a single arm (other arm's segment unverified), so refuse."""
    from pathlib import Path
    import pytest
    if not Path(_Env.cfg.urdf_path).exists():
        pytest.skip("urdf not present")
    api, env = _api(monkeypatch)
    env.low_level.get_joint_positions = lambda: {
        "lifter_pitch_1_joint": 0.0, "lifter_pitch_2_joint": 0.0,
        "lifter_pitch_3_joint": 0.0, "waist_yaw_joint": 0.0,
        "L_shoulder_pitch_joint": 0.5,
    }
    monkeypatch.setattr(api, "_arm_home_path_collides", lambda *a, **k: True)  # direct path collides
    # only the left arm's ready IK converged -> partial reroute target
    monkeypatch.setattr(api, "_ready_arm_q",
                        lambda chains, qf: {"left": {j: 0.1 for j in ARM_JOINTS["left"]}})
    out = api._home_safely()
    assert out == {"ok": False, "reason": "home_path_self_collision"}
    assert env.low_level.moves == []  # never commanded the partial reroute
