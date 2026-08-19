# coding: utf-8
"""Continuous straight-in place-side creep (opt-in via ``place_servo_enabled``) for approach_for_place.

These drive ``_approach_for_place_continuous_servo`` with a scripted ``locate_for_place`` stream and a
``_ServoLL`` stub that records every start/steer/hold/stop, pinning its contract: once the base is SQUARED
to the near edge, creep continuously STRAIGHT to ``place_approach_edge_m`` and stop the base EXACTLY ONCE
(no per-frame stop-to-look freeze — the reported place-side stutter), NEVER feed a non-zero turn (a carried
box must not be flung off the edge normal), HOLD the wheels on a far loss and resume STRAIGHT once
re-acquired, and commit the final short segment open-loop on a near loss. The first ``approach_for_place``
sense is a squared+far edge so the discrete loop hands straight off to the servo; later senses are the
servo's own re-senses. Mirrors ``test_approach_servo_api.py``'s ``_ServoLL`` and ``test_approach_for_place.py``'s
``_surf_edge``."""

import math

from jiuwensymbiosis.adapters.cruzr.api import CruzrApi
from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig


class _ServoLL:
    """Records the servo drive ops (``.ops``) and any terminal ``navigate_relative`` finish (``.nav_calls``)
    so a test can assert stop fires once, no non-zero turn is ever commanded, and a hold sits before its
    straight resume."""

    def __init__(self):
        self.nav_calls = []      # (dx, dy, dyaw) — terminal open-loop finish (empty on the pure-creep path)
        self.ops = []            # ordered: ("start",k_fwd,fwd_max) ("steer",b) ("hold",) ("stop",)
        self.start_calls = []
        self._running = True

    def navigate_relative(self, dx, dy=0.0, dyaw=0.0, *, k_rot=None, k_rot_slow_rad=None, k_fwd=None):
        self.nav_calls.append((round(float(dx), 4), round(float(dy), 4), round(float(dyaw), 4)))
        return {"ok": True, "yaw_reached": dyaw}

    def start_base_drive(self, *, k_fwd=None, fwd_max_m=None):
        self.start_calls.append({"k_fwd": k_fwd, "fwd_max_m": fwd_max_m})
        self.ops.append(("start", k_fwd, fwd_max_m))
        return "H"

    def base_drive_running(self, handle):
        return self._running

    def steer_base_drive(self, handle, bearing_rad):
        self.ops.append(("steer", round(float(bearing_rad), 4)))

    def hold_base_drive(self, handle):
        self.ops.append(("hold",))

    def stop_base_drive(self, handle):
        self.ops.append(("stop",))
        self._running = False
        return {"ok": True, "reason": "stopped", "dist_traveled": 0.0}

    @property
    def steers(self):
        return [o[1] for o in self.ops if o[0] == "steer"]

    @property
    def n_holds(self):
        return sum(1 for o in self.ops if o[0] == "hold")

    @property
    def n_stops(self):
        return sum(1 for o in self.ops if o[0] == "stop")


class _Env:
    def __init__(self, cfg):
        self.low_level = _ServoLL()
        self.cfg = cfg


def _cfg(*, enabled=True, **over):
    cfg = CruzrConfig()
    cfg.place_servo_enabled = enabled
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _api(senses, *, enabled=True, **cfg_over):
    env = _Env(_cfg(enabled=enabled, **cfg_over))
    api = CruzrApi(env)
    it = iter(senses)
    api.locate_for_place = lambda object_name="table", reference=None, relation="on": next(it)  # type: ignore[method-assign]
    # approach_* now folds the search-and-face pass in front of the drive loop. These tests
    # exercise the DRIVE geometry, so mark a surface as already sensed — the same thing a
    # real face pass would leave behind — and the search is skipped (see _ApproachBody).
    api._last_surface = {"ok": True}
    return api, env


def _edge(front_x_mm):
    """Squared near-edge surface at ``front_x_mm`` (normal = −x → square_turn 0, trusted fit)."""
    return {"ok": True, "front_x_mm": front_x_mm, "center_mm": [front_x_mm + 100.0, 0.0, 0.0],
            "surface_z_mm": 600.0, "yaw_rad": 0.0, "long_mm": 800.0, "short_mm": 500.0,
            "edge_normal": [-1.0, 0.0], "edge_quality": 0.05, "edge_len_mm": 600.0,
            "edge_midpoint_mm": [front_x_mm, 0.0]}


def _lost():
    return {"ok": False, "reason": "no_surface"}


# place_approach_edge_m=0.35, base_pos_tol_m=0.05 → reached when front_x ≤ 400mm; far entry at 800mm.
def test_place_servo_continuous_no_per_frame_stop():
    """A shrinking front_x stream creeps straight to the edge with ONE clean stop and NO discrete per-leg
    navigate_relative freeze (the reported place-side stutter); the cached surface is invalidated."""
    api, env = _api([_edge(800.0),                    # approach_for_place hands off (squared + far)
                     _edge(700.0),                    # servo s0 → start worker
                     _edge(600.0), _edge(500.0),      # creep, still far
                     _edge(380.0)])                   # near edge (forward 0.03 ≤ tol) → one stop
    out = api.approach_for_place("table")
    ll = env.low_level
    assert out["ok"] and out["status"] == "in_range"
    assert ll.n_stops == 1                            # not per-frame stop-to-look
    assert len(ll.start_calls) == 1
    assert ll.nav_calls == []                         # continuous creep owns the advance; reached → no finish
    assert ll.steers == [] and ll.n_holds == 0
    assert api._last_surface is None                  # moved → dual_arm_place re-senses
    # fwd_cap = min(forward0 + reserve, grasp_servo_fwd_max_m) = min((0.70-0.35)+0.15, 1.2) = 0.50
    assert math.isclose(ll.start_calls[0]["fwd_max_m"], (0.70 - 0.35) + 0.15, abs_tol=1e-6)
    assert math.isclose(ll.start_calls[0]["k_fwd"], CruzrConfig().grasp_servo_creep_k_fwd, abs_tol=1e-9)


def test_place_servo_no_steer_holds_square():
    """Core carried-box invariant: the base is NEVER commanded a non-zero turn — no steer bearing and no
    terminal nav dyaw is anything but 0.0 — so the squared heading (box on the edge normal) is preserved."""
    api, env = _api([_edge(800.0), _edge(700.0), _edge(600.0), _edge(500.0), _edge(380.0)])
    out = api.approach_for_place("table")
    ll = env.low_level
    assert out["ok"] and out["status"] == "in_range"
    assert all(b == 0.0 for b in ll.steers)           # never a steering turn off the edge normal
    assert all(dyaw == 0.0 for (_dx, _dy, dyaw) in ll.nav_calls)   # any open-loop finish is straight


def test_place_servo_already_at_edge_no_creep():
    """The servo's own first re-sense already at the place gap → return in_range WITHOUT starting the
    worker (no creep, no stop)."""
    api, env = _api([_edge(800.0),                    # approach_for_place hands off (far)
                     _edge(350.0)])                   # servo s0 already at the edge → no creep
    out = api.approach_for_place("table")
    ll = env.low_level
    assert out["ok"] and out["status"] == "in_range" and out["iters"] == 0
    assert ll.start_calls == []                       # worker never started
    assert ll.n_stops == 0 and ll.nav_calls == []


def test_place_servo_commit_open_loop_when_surface_lost_near():
    """A surface lost within the commit distance (waist loses the edge up close) commits the final short
    segment OPEN-LOOP: no hold, one stop, then exactly one STRAIGHT terminal move for the remaining gap."""
    api, env = _api([_edge(800.0),                    # hand off
                     _edge(600.0),                    # s0 → start worker; forward_rem 0.25 ≤ commit 0.30
                     _lost()])                         # lost near → commit, break
    out = api.approach_for_place("table")
    ll = env.low_level
    assert out["ok"] and out["status"] == "in_range"
    assert ll.n_holds == 0                             # near loss commits, never holds
    assert ll.n_stops == 1
    assert len(ll.nav_calls) == 1                      # one open-loop finish
    dx, _dy, dyaw = ll.nav_calls[0]
    assert dyaw == 0.0                                 # straight-in, no turn
    assert math.isclose(dx, 0.60 - 0.35, abs_tol=1e-6)   # remaining forward from the last good surface
    assert ll.steers == []


def test_place_servo_holds_when_lost_far():
    """A surface lost while still FAR (remaining forward > commit) HOLDs the wheels (no blind creep) and,
    once re-acquired, resumes STRAIGHT with a bearing-0 steer (the worker's resume, not a turn)."""
    api, env = _api([_edge(800.0),                    # hand off
                     _edge(800.0),                    # s0 → start worker; forward_rem 0.45 > commit 0.30
                     _lost(),                          # lost far → hold, keep polling
                     _edge(700.0),                     # re-acquired → resume straight
                     _edge(380.0)])                    # near edge → one stop
    out = api.approach_for_place("table")
    ll = env.low_level
    assert out["ok"] and out["status"] == "in_range"
    assert ll.n_holds == 1                             # exactly one pause
    assert ll.steers == [0.0]                          # resume is a straight bearing-0, never a turn
    kinds = [o[0] for o in ll.ops if o[0] in ("hold", "steer")]
    assert kinds == ["hold", "steer"]                  # resume follows the pause
    assert ll.n_stops == 1
    assert ll.nav_calls == []                          # reached under closed loop → no open-loop finish


def test_place_servo_disabled_uses_discrete_straight_in():
    """``place_servo_enabled=False`` → the discrete straight-in tail runs (unchanged baseline); the servo
    drive worker is NEVER started and the base is driven only via navigate_relative."""
    api, env = _api([_edge(800.0),                    # far + squared → discrete straight-in step
                     _edge(350.0)],                    # in range
                    enabled=False)
    out = api.approach_for_place("table")
    ll = env.low_level
    assert out["ok"] and out["status"] == "in_range"
    assert ll.start_calls == [] and ll.ops == []      # servo worker never started
    assert len(ll.nav_calls) >= 1                      # discrete loop drove via navigate_relative
    assert all(dyaw == 0.0 for (_dx, _dy, dyaw) in ll.nav_calls)   # straight-in, no radial turn
