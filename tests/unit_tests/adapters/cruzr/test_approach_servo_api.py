# coding: utf-8
"""Continuous visual-servo terminal (opt-in via ``grasp_servo_enabled``) for approach_for_grasp.

These drive ``_approach_continuous_servo`` with a scripted detection stream and a ``_ServoLL`` stub that
records every start/steer/hold/stop, pinning its contract: creep continuously toward the LATEST centroid
(track a moved box), stop the base EXACTLY ONCE (no per-frame stop-to-look freeze — the reported stutter),
HOLD the wheels on a far loss but commit the final short segment open-loop on a near loss, and NEVER grasp
blind (too_close / never-acquired abort). The seed detection is placed ON the face-normal line at the
standoff entry so ``_approach_single_shot`` plans ZERO L-route legs — ``.calls`` then isolates the servo's
own terminal align move. Mirrors ``test_approach_box_api.py``'s ``_make_cfg`` / ``_det`` / ``_script``."""

import math
from types import SimpleNamespace

from jiuwensymbiosis.adapters.cruzr.api import CruzrApi
from jiuwensymbiosis.motion.base_goal import plan_base_goal_for_grasp


class _ServoLL:
    """Records fine-positioning navs (``.calls``) AND the ordered servo drive ops (``.ops``) so a test can
    assert the steer bearings, that stop fires exactly once, and that a hold sits between two steers."""

    def __init__(self):
        self.calls = []          # ("nav", dx, dy, dyaw) — terminal align move (L-route is 0 in these tests)
        self.kw = []
        self.ops = []            # ordered: ("start",k_fwd,fwd_max) ("steer",b) ("hold",) ("stop",)
        self.start_calls = []
        self.nav_result = {"ok": True}
        self.drive_result = {"ok": True, "dist_traveled": 0.0}
        self._running = True

    def navigate_relative(self, dx_m, dy_m=0.0, dyaw_rad=0.0, *,
                          k_rot=None, k_rot_slow_rad=None, k_fwd=None):
        self.calls.append(("nav", round(float(dx_m), 4), round(float(dy_m), 4), round(float(dyaw_rad), 4)))
        self.kw.append({"k_rot": k_rot, "k_rot_slow_rad": k_rot_slow_rad, "k_fwd": k_fwd})
        return self.nav_result

    def start_base_drive(self, *, k_fwd=None, fwd_max_m=None):
        self.start_calls.append({"k_fwd": k_fwd, "fwd_max_m": fwd_max_m})
        self.ops.append(("start", k_fwd, fwd_max_m))
        return "H"

    def base_drive_running(self, handle):
        return self._running                       # True → loop exits via break / in-band / too_close

    def steer_base_drive(self, handle, bearing_rad):
        self.ops.append(("steer", round(float(bearing_rad), 4)))

    def hold_base_drive(self, handle):
        self.ops.append(("hold",))

    def stop_base_drive(self, handle):
        self.ops.append(("stop",))
        self._running = False
        return dict(self.drive_result)

    @property
    def steers(self):
        return [o[1] for o in self.ops if o[0] == "steer"]

    @property
    def n_holds(self):
        return sum(1 for o in self.ops if o[0] == "hold")

    @property
    def n_stops(self):
        return sum(1 for o in self.ops if o[0] == "stop")

    @property
    def navs(self):
        return [c for c in self.calls if c[0] == "nav"]


def _make_cfg(**over):
    base = {
        "grasp_target_forward_m": 0.40, "grasp_forward_min_m": 0.30, "grasp_forward_max_m": 0.50,
        "base_pos_tol_m": 0.05, "base_yaw_tol_rad": 0.08, "approach_converge_iters": 4,
        "approach_forward_step_m": 0.25, "grasp_approach_forward_reserve_m": 0.15,
        "approach_k_rot": 1.5, "approach_k_rot_slow_rad": 0.5, "approach_k_fwd": 0.8,
        "grasp_square_tol_rad": 0.26,
        "grasp_arc_enabled": False, "grasp_arc_standoff_m": 0.10, "grasp_face_flatness_max": 0.15,
        # continuous visual-servo terminal (opt-in; class defaults mirror config.py)
        "grasp_servo_enabled": False, "grasp_servo_creep_k_fwd": 0.6, "grasp_servo_fwd_max_m": 1.2,
        "grasp_servo_commit_dist_m": 0.30, "grasp_servo_max_polls": 40, "grasp_servo_lost_hold": True,
    }
    base.update(over)
    return SimpleNamespace(**base)


class _Env:
    def __init__(self, cfg):
        self.low_level = _ServoLL()
        self.cfg = cfg


def _det(x_mm, y_mm=0.0):
    return {"ok": True, "center_mm": [x_mm, y_mm, 780.0], "object": "box"}


def _seed():
    """Seed ON the face-normal line at the standoff entry (rng .5, d_e = grasp_d .4 + standoff .1 = .5) so
    the L-route plans ZERO legs, AND with a trustworthy flat normal so ``_approach_single_shot`` does not
    DECLINE — the hand-off reaches the servo with ``.calls`` empty of route legs."""
    return {"ok": True, "center_mm": [500.0, 0.0, 780.0], "object": "box",
            "face_normal": [-1.0, 0.0], "face_flatness": 0.05}


def _script(api, dets):
    it = iter(dets)
    api.locate_for_grasp = lambda object_name="box", reference=None, relation="on": next(it)   # type: ignore[method-assign]


def _servo_api(dets, **cfg_over):
    api = CruzrApi(_Env(_make_cfg(grasp_arc_enabled=True, grasp_servo_enabled=True, **cfg_over)))
    api._last_detection = _seed()
    _script(api, dets)
    return api


def _bearing(det, cfg):
    return round(plan_base_goal_for_grasp(det["center_mm"], cfg)[0], 4)


def test_servo_steers_toward_moving_target():
    """A stream of in-motion detections with a shrinking bearing → the servo feeds each LATEST centroid
    direction to the drive worker (so a box moved during the final stretch is tracked) and stops ONCE."""
    a = _det(450.0, 90.0)                            # in-range distance, off-centre → steer
    b = _det(430.0, 40.0)                            # closer, still off-centre → steer to the new bearing
    c = _det(420.0, 0.0)                             # in band + centred → the one clean stop
    rest = _det(415.0, 0.0)                          # AT-REST refresh after the stop (base static → current coords)
    api = _servo_api([a, b, c, rest])
    out = api.approach_for_grasp()
    ll = api.env.low_level
    assert out["ok"] and out["status"] == "in_band"
    assert ll.steers == [_bearing(a, api.env.cfg), _bearing(b, api.env.cfg)]
    assert ll.n_stops == 1                           # not per-frame stop-to-look (the stutter)
    assert ll.navs == []                             # in_band + square → no terminal move (the refresh is a detect)
    assert api._last_detection is rest               # cache is the AT-REST frame, not the mid-creep c


def test_servo_owns_full_forward_no_discrete_leg():
    """The servo now owns the ENTIRE forward approach: ``_approach_single_shot`` drives the approach leg
    turn-only (``app_dist=0``) so there is no discrete forward-then-stop seam before the creep. The worker's
    forward budget is the adaptive cap ``min(fwd0+reserve, ceiling)`` — for a far seed that is ``fwd0+reserve``,
    exceeding the old fixed 0.40 backstop, proving the cap scales with the seed's distance-to-grasp."""
    api = CruzrApi(_Env(_make_cfg(grasp_arc_enabled=True, grasp_servo_enabled=True)))
    # Seed far on the face-normal line: servo OFF would drive a ~0.5 m approach leg (a dx>0 nav); servo ON
    # must defer it entirely, so NO nav carries forward distance.
    seed = {"ok": True, "center_mm": [900.0, 0.0, 780.0], "object": "box",
            "face_normal": [-1.0, 0.0], "face_flatness": 0.05}
    api._last_detection = seed
    # first in-motion frame in band + centred → one clean stop; second is the at-rest refresh after the stop
    _script(api, [_det(400.0, 0.0), _det(398.0, 0.0)])
    out = api.approach_for_grasp()
    ll = api.env.low_level
    assert out["ok"] and out["status"] == "in_band"
    assert all(n[1] == 0.0 for n in ll.navs)         # no discrete forward leg — servo owns the forward
    assert ll.n_stops == 1
    fwd0 = plan_base_goal_for_grasp(seed["center_mm"], api.env.cfg)[1]
    assert math.isclose(ll.start_calls[0]["fwd_max_m"], fwd0 + 0.15, abs_tol=1e-4)   # adaptive cap = fwd0+reserve
    assert ll.start_calls[0]["fwd_max_m"] > 0.40     # exceeds the old fixed backstop → cap scales with distance


def test_servo_stops_once_on_band_exit():
    """First in-motion frame already in band + centred → break immediately: one stop, no steer, no terminal
    nav, and the at-rest refresh (not the mid-creep frame) is the cached detection."""
    d = _det(400.0, 0.0)
    rest = _det(398.0, 0.0)                           # at-rest refresh after the stop
    api = _servo_api([d, rest])
    out = api.approach_for_grasp()
    ll = api.env.low_level
    assert out["ok"] and out["status"] == "in_band" and out["iters"] == 1
    assert ll.steers == []
    assert ll.n_stops == 1
    assert ll.navs == []
    assert api._last_detection is rest               # the AT-REST frame, not the mid-creep d


def test_servo_aborts_when_rest_refresh_lost():
    """In-band creep stops, but the at-rest refresh misses → abort not-ok rather than clamp the stale
    mid-creep frame (never grasp on a base_link snapshot that lags the final pose)."""
    d = _det(400.0, 0.0)                             # in band + centred → stop
    lost = {"ok": False, "reason": "no_detection"}   # refresh at rest finds nothing
    api = _servo_api([d, lost])
    out = api.approach_for_grasp()
    ll = api.env.low_level
    assert out["ok"] is False and out["reason"] == "no_detection"
    assert ll.n_stops == 1                           # base still stopped exactly once
    assert ll.navs == []                             # aborted before any grasp move


def test_servo_holds_on_lost_then_resumes():
    """A box lost mid-approach while still FAR (remaining forward > commit) HOLDs the wheels (no blind
    creep on a stale bearing) and resumes steering once re-acquired — exactly one hold, between two steers."""
    far = _det(800.0, 100.0)                         # locked but far: fwd_rem .41 > commit .30
    lost = {"ok": False, "reason": "no_detection"}
    resume = _det(500.0, 60.0)                        # re-acquired, still off-centre → steer again
    done = _det(420.0, 0.0)                           # in band + centred → stop
    api = _servo_api([far, lost, resume, done, _det(415.0, 0.0)])   # last is the at-rest refresh after the stop
    out = api.approach_for_grasp()
    ll = api.env.low_level
    assert out["ok"] and out["status"] == "in_band"
    assert ll.n_holds == 1
    assert ll.steers == [_bearing(far, api.env.cfg), _bearing(resume, api.env.cfg)]
    kinds = [o[0] for o in ll.ops if o[0] in ("steer", "hold")]
    assert kinds == ["steer", "hold", "steer"]       # pause sits BETWEEN the two steers
    assert ll.n_stops == 1


def test_servo_commit_open_loop_when_box_leaves_view_near():
    """A box lost within the commit distance (waist camera loses it up close) commits the final short
    segment OPEN-LOOP: no hold, one stop, then exactly one terminal move walking the remaining forward,
    followed by a refresh re-detect that becomes the cached grasp frame."""
    near = _det(650.0, 90.0)                          # locked near: fwd_rem .26 ≤ commit .30, off-centre
    lost = {"ok": False, "reason": "no_detection"}
    after = _det(400.0, 0.0)                          # back in view after the open-loop segment → in band
    api = _servo_api([near, lost, after])
    out = api.approach_for_grasp()
    ll = api.env.low_level
    assert out["ok"] and out["status"] == "in_band"
    assert ll.n_holds == 0                            # near loss commits, never holds
    assert ll.n_stops == 1
    assert len(ll.navs) == 1                          # one open-loop terminal move
    forward = plan_base_goal_for_grasp(near["center_mm"], api.env.cfg)[1]
    assert math.isclose(ll.navs[0][1], round(forward, 4), abs_tol=1e-4)   # ≤2×reserve → walks it in full
    assert api._last_detection is after


def test_servo_too_close_aborts():
    """An in-motion frame nearer than the grasp band → stop and abort not-ok: never grasp blind, and the
    too-close frame does NOT overwrite the cached detection."""
    api = _servo_api([_det(250.0, 0.0)])             # crept inside the band from too close
    out = api.approach_for_grasp()
    ll = api.env.low_level
    assert out["ok"] is False and out["reason"] == "too_close"
    assert ll.n_stops == 1
    assert ll.steers == [] and ll.n_holds == 0
    assert ll.navs == []                             # aborted — no blind grasp move
    assert api._last_detection["center_mm"][0] == 500.0   # seed kept, too-close frame rejected


def test_servo_disabled_uses_single_shot_terminal():
    """``grasp_servo_enabled=False`` → the single-shot's own static re-detect tail runs (unchanged
    baseline); the servo drive worker is NEVER started."""
    api = CruzrApi(_Env(_make_cfg(grasp_arc_enabled=True, grasp_servo_enabled=False)))
    api._last_detection = _seed()
    _script(api, [_det(400.0, 0.0)])                 # one post-route re-detect lands in band → cache
    out = api.approach_for_grasp()
    ll = api.env.low_level
    assert out["ok"] and out["status"] == "in_band"
    assert ll.start_calls == []                      # servo worker never started
    assert ll.ops == []                              # no drive/steer/hold/stop at all


def test_servo_decline_falls_back_to_discrete():
    """An untrustworthy normal makes ``_approach_single_shot`` DECLINE before any motion (even with the
    servo enabled) → the discrete polling loop closes it; the servo worker is never started."""
    api = CruzrApi(_Env(_make_cfg(grasp_arc_enabled=True, grasp_servo_enabled=True)))
    api._last_detection = _det(700.0, 0.0)           # plain: no face_normal / no footprint → decline
    _script(api, [_det(400.0, 0.0)])
    out = api.approach_for_grasp()
    ll = api.env.low_level
    assert out["ok"] and out["status"] == "in_band"
    assert ll.start_calls == []                      # decline routed to the discrete loop, not the servo
    assert any(c[0] == "nav" for c in ll.calls)      # discrete loop still drove via navigate_relative
