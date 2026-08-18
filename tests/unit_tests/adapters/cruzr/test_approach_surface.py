# coding: utf-8
"""approach_for_place: drive the base to place range only when the surface is out of reach
(no-op when in range); shares _drive_base with approach_for_grasp."""
import math
from jiuwensymbiosis.adapters.cruzr.api import CruzrApi
from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig


class _LL:
    def __init__(self): self.nav_calls = []
    def navigate_relative(self, dx, dy, dyaw, *, k_rot=None, k_rot_slow_rad=None, k_fwd=None):
        self.nav_calls.append((dx, dy, dyaw)); return {"ok": True, "yaw_reached": dyaw}


class _Env:
    def __init__(self): self.low_level = _LL(); self.cfg = CruzrConfig()  # place_approach_edge_m=0.35


def _api(senses):
    env = _Env(); api = CruzrApi(env)
    it = iter(senses)
    api.locate_for_place = lambda object_name="table", reference=None, relation="on": next(it)  # type: ignore[method-assign]
    # approach_* now folds the search-and-face pass in front of the drive loop. These tests
    # exercise the DRIVE geometry, so mark a surface as already sensed — the same thing a
    # real face pass would leave behind — and the search is skipped (see _ApproachBody).
    api._last_surface = {"ok": True}
    return api, env


def _surf(front_x_mm, cy_mm=0.0, yaw_rad=0.0, long_mm=0.0, short_mm=0.0):
    return {"ok": True, "front_x_mm": front_x_mm, "center_mm": [front_x_mm + 100.0, cy_mm, 0.0],
            "surface_z_mm": 600.0, "yaw_rad": yaw_rad, "long_mm": long_mm, "short_mm": short_mm}


def _surf_edge(front_x_mm, cy_mm=0.0, edge_normal=None, edge_quality=0.05, edge_len_mm=600.0,
               yaw_rad=0.0, long_mm=800.0, short_mm=500.0):
    s = _surf(front_x_mm, cy_mm, yaw_rad, long_mm, short_mm)
    if edge_normal is not None:
        s["edge_normal"] = list(edge_normal)
        s["edge_quality"] = edge_quality
        s["edge_len_mm"] = edge_len_mm
        s["edge_midpoint_mm"] = [front_x_mm, cy_mm]
    return s


def test_drives_when_surface_too_far():
    api, env = _api([_surf(1000.0), _surf(350.0)])  # far → drive up → next sense is in range
    out = api.approach_for_place("table")
    assert out["ok"] and out["status"] == "in_range"  # iterates until converged
    assert len(env.low_level.nav_calls) == 1
    dx, dy, dyaw = env.low_level.nav_calls[0]
    assert (dy, ) == (0.0, )
    # forward = front_x - place_approach_edge_m = 0.65; capped to ONE approach_forward_step_m step
    # (0.25), not the full 0.65 lunge — a distant/false frame can only advance one small step.
    assert math.isclose(dx, CruzrConfig().approach_forward_step_m, rel_tol=1e-6)


def test_forward_advance_capped_per_step():
    """A far surface is closed in SMALL capped steps (approach_forward_step_m each), never a single
    full-distance lunge: one detection frame — including a distant false positive — advances only one
    step before the next re-sense corrects it. With resident_workers the extra steps aren't choppy."""
    api, env = _api([_surf(1000.0), _surf(600.0), _surf(350.0)])  # 0.65 → 0.25 → in range
    out = api.approach_for_place("table")
    assert out["ok"] and out["status"] == "in_range"
    step = CruzrConfig().approach_forward_step_m
    dxs = [dx for (dx, _dy, _dyaw) in env.low_level.nav_calls]
    assert dxs and all(math.isclose(d, step, rel_tol=1e-6) for d in dxs)  # every advance capped to one step


def test_noop_when_in_range():
    api, env = _api([_surf(350.0)])           # front edge already at the place gap → don't move
    out = api.approach_for_place("table")
    assert out["ok"] and out["status"] == "in_range"
    assert env.low_level.nav_calls == []


def test_drives_and_centers_when_offset():
    api, env = _api([_surf(1000.0, 300.0), _surf(350.0)])  # offset → turn by bearing → then in range
    out = api.approach_for_place("table")
    assert out["ok"]
    _, _, dyaw = env.low_level.nav_calls[0]
    assert math.isclose(dyaw, math.atan2(0.3, 1.1), rel_tol=1e-6)


def test_squares_to_near_edge_when_oblique_but_positioned():
    # Positioned (at edge distance, centred) but the footprint says the base is 0.3 rad off
    # square to the near edge → rotate in place by 0.3, then the next sense reads squared.
    api, env = _api([_surf(350.0, 0.0, yaw_rad=0.3, long_mm=800.0, short_mm=500.0),
                     _surf(350.0, 0.0, yaw_rad=0.0, long_mm=800.0, short_mm=500.0)])
    out = api.approach_for_place("table")
    assert out["ok"] and out["status"] == "in_range"
    assert len(env.low_level.nav_calls) == 1
    dx, dy, dyaw = env.low_level.nav_calls[0]
    assert (dx, dy) == (0.0, 0.0)                       # rotate in place (position already good)
    assert math.isclose(dyaw, 0.3, abs_tol=1e-6)


def test_near_square_footprint_now_squares():
    # long≈short (near-square table) used to SKIP squaring (aspect gate) → base left oblique. Now the
    # hysteresis-locked footprint normal has no aspect gate, so it squares to a near edge like the
    # rectangular case: rotate in place by 0.3, next sense reads squared → in_range.
    api, env = _api([_surf(350.0, 0.0, yaw_rad=0.3, long_mm=600.0, short_mm=580.0),
                     _surf(350.0, 0.0, yaw_rad=0.0, long_mm=600.0, short_mm=580.0)])
    out = api.approach_for_place("table")
    assert out["ok"] and out["status"] == "in_range"
    assert len(env.low_level.nav_calls) == 1
    dx, dy, dyaw = env.low_level.nav_calls[0]
    assert (dx, dy) == (0.0, 0.0)                       # rotate in place (position already good)
    assert math.isclose(dyaw, 0.3, abs_tol=1e-6)


def test_square_lock_survives_yaw_jitter(monkeypatch):
    # The near-edge normal is hysteresis-locked and frame-compensated across turns: iteration 2 must
    # receive iteration 1's normal rotated by −cmd_turn (into the new base frame) as prev_normal, so a
    # jittery footprint can't 90°-flip the edge. Spy on near_face_normal to assert the wiring.
    import jiuwensymbiosis.motion.approach as api_mod

    seen = []
    # iter1 → a normal needing a 0.4 turn to face; iter2 → already facing (square_turn 0) → in_range.
    rets = iter([(-math.cos(0.4), -math.sin(0.4)), (-1.0, 0.0)])

    def _fake(fp, cfg, prev_normal=None):
        seen.append(prev_normal)
        return next(rets)

    monkeypatch.setattr(api_mod, "near_face_normal", _fake)
    api, env = _api([_surf(350.0, 0.0, yaw_rad=0.9, long_mm=600.0, short_mm=560.0),
                     _surf(350.0, 0.0, yaw_rad=0.9, long_mm=600.0, short_mm=560.0)])
    out = api.approach_for_place("table")
    assert out["ok"] and out["status"] == "in_range"
    assert seen[0] is None                                   # first pick has no lock
    # iter1 normal (−cos0.4,−sin0.4) rotated by −0.4 ≈ (−1, 0): the compensated lock fed to iter2.
    assert seen[1] is not None
    assert math.isclose(seen[1][0], -1.0, abs_tol=1e-6) and math.isclose(seen[1][1], 0.0, abs_tol=1e-6)


def test_in_range_when_positioned_and_squared():
    # Trusted edge fit, base at the edge distance (front_x == gap) and already squared (normal = −x) → no-op.
    api, env = _api([_surf_edge(350.0, 0.0, edge_normal=(-1.0, 0.0))])
    out = api.approach_for_place("table")
    assert out["ok"] and out["status"] == "in_range"
    assert env.low_level.nav_calls == []


def test_squares_to_edge_normal_when_positioned():
    # Positioned (front_x == gap) but the near-edge normal is tilted 0.3 rad → rotate IN PLACE by 0.3.
    theta = 0.3
    api, env = _api([_surf_edge(350.0, 0.0, edge_normal=(-math.cos(theta), -math.sin(theta))),
                     _surf_edge(350.0, 0.0, edge_normal=(-1.0, 0.0))])   # squared next
    out = api.approach_for_place("table")
    assert out["ok"] and out["status"] == "in_range"
    assert len(env.low_level.nav_calls) == 1
    dx, dy, dyaw = env.low_level.nav_calls[0]
    assert (dx, dy) == (0.0, 0.0)                          # rotate in place (no lateral centring)
    assert math.isclose(dyaw, theta, abs_tol=1e-6)


def test_squares_first_then_straight_in_when_far_and_oblique():
    # Far AND off-square → SQUARE FIRST (rotate in place), then straight-in (turn=0) — never a radial turn
    # toward the centroid, and never a lateral swing to the edge midpoint (that flung the box).
    theta = 0.3
    api, env = _api([_surf_edge(800.0, 0.0, edge_normal=(-math.cos(theta), -math.sin(theta))),
                     _surf_edge(800.0, 0.0, edge_normal=(-1.0, 0.0)),    # squared, still far
                     _surf_edge(350.0, 0.0, edge_normal=(-1.0, 0.0))])   # in range
    out = api.approach_for_place("table")
    assert out["ok"] and out["status"] == "in_range"
    dx0, dy0, dyaw0 = env.low_level.nav_calls[0]
    assert (dx0, dy0) == (0.0, 0.0) and math.isclose(dyaw0, theta, abs_tol=1e-6)   # square first
    for _dx, _dy, dyaw in env.low_level.nav_calls[1:]:
        assert dyaw == 0.0                                                          # straight-in, no turn


def test_straight_in_uses_reserve():
    # Squared + far → straight-in advance uses the reserve strategy (forward − reserve), bigger than the
    # old 0.25 cap, no turn (no lateral chase to the edge midpoint).
    api, env = _api([_surf_edge(800.0, 0.0, edge_normal=(-1.0, 0.0)),
                     _surf_edge(350.0, 0.0, edge_normal=(-1.0, 0.0))])
    out = api.approach_for_place("table")
    assert out["ok"] and out["status"] == "in_range"
    dx, dy, dyaw = env.low_level.nav_calls[0]
    assert (dy, dyaw) == (0.0, 0.0)                        # straight-in, no turn
    reserve = CruzrConfig().place_approach_forward_reserve_m
    assert math.isclose(dx, (0.8 - 0.35) - reserve, rel_tol=1e-6)
    assert dx > CruzrConfig().approach_forward_step_m


def test_offset_edge_midpoint_does_not_swing_base_sideways():
    # Edge midpoint 0.3 m to the side but the base is squared: it must NOT turn toward the sideways
    # midpoint (the aim-at-P degeneracy that flung the box). Drive straight in / stay, dyaw stays small.
    api, env = _api([_surf_edge(800.0, 300.0, edge_normal=(-1.0, 0.0)),
                     _surf_edge(350.0, 300.0, edge_normal=(-1.0, 0.0))])
    out = api.approach_for_place("table")
    assert out["ok"]
    for _dx, _dy, dyaw in env.low_level.nav_calls:
        assert abs(dyaw) < 0.2                             # never a ~90° sideways swing


def test_large_square_turn_is_capped():
    # A flipped/wrong near-edge normal would demand a ~86° turn; the per-step cap bounds it so the base
    # never swings wildly with the box held (the fail-safe against the box-fling).
    big = 1.5
    api, env = _api([_surf_edge(350.0, 0.0, edge_normal=(-math.cos(big), -math.sin(big))),
                     _surf_edge(350.0, 0.0, edge_normal=(-1.0, 0.0))])   # squared next → terminate
    out = api.approach_for_place("table")
    assert out["ok"]
    _dx, _dy, dyaw = env.low_level.nav_calls[0]
    assert math.isclose(abs(dyaw), CruzrConfig().place_max_turn_step_rad, rel_tol=1e-6)   # capped, not 1.5


def test_untrusted_edge_falls_back_to_footprint():
    # Edge fit present but quality too high (> place_edge_quality_max) → ignore it, square by the
    # footprint yaw (0.3) instead. Proves the trust gate + fallback.
    api, env = _api([_surf_edge(350.0, 0.0, edge_normal=(-1.0, 0.0), edge_quality=0.9, yaw_rad=0.3),
                     _surf_edge(350.0, 0.0, edge_normal=(-1.0, 0.0), edge_quality=0.9, yaw_rad=0.0)])
    out = api.approach_for_place("table")
    assert out["ok"] and out["status"] == "in_range"
    assert len(env.low_level.nav_calls) == 1
    _, _, dyaw = env.low_level.nav_calls[0]
    assert math.isclose(dyaw, 0.3, abs_tol=1e-6)           # footprint drove the turn (edge ignored)


def test_footprint_fallback_squares_first_then_straight_in():
    # No trusted edge fit (footprint only, no edge midpoint to centre on) → SQUARE FIRST (rotate in place)
    # then straight-in (turn=0), never a radial turn after squaring.
    api, env = _api([_surf(800.0, 0.0, yaw_rad=0.3, long_mm=800.0, short_mm=500.0),   # far + off-square
                     _surf(800.0, 0.0, yaw_rad=0.0, long_mm=800.0, short_mm=500.0),   # squared, still far
                     _surf(350.0, 0.0, yaw_rad=0.0, long_mm=800.0, short_mm=500.0)])  # in range
    out = api.approach_for_place("table")
    assert out["ok"] and out["status"] == "in_range"
    dx0, dy0, dyaw0 = env.low_level.nav_calls[0]
    assert (dx0, dy0) == (0.0, 0.0) and math.isclose(dyaw0, 0.3, abs_tol=1e-6)   # square first
    for _dx, _dy, dyaw in env.low_level.nav_calls[1:]:
        assert dyaw == 0.0                                                        # straight-in, no radial


def test_degenerate_footprint_no_square():
    # Degenerate/absent footprint (long==short==0 → normal is None): nothing to square to → in_range
    # with no rotation (never chase a bogus normal).
    api, env = _api([_surf(350.0, 0.0, yaw_rad=0.3, long_mm=0.0, short_mm=0.0)])
    out = api.approach_for_place("table")
    assert out["ok"] and out["status"] == "in_range"
    assert env.low_level.nav_calls == []


def test_no_surface_propagates():
    api, env = _api([{"ok": False, "reason": "color_mismatch"}])
    out = api.approach_for_place("white table")
    assert out["ok"] is False and out["reason"] == "color_mismatch"
    assert env.low_level.nav_calls == []


def test_grounding_sticky_across_resenses():
    """A grounded surface (locate_for_place with a reference) must stay grounded on every post-move
    re-sense, else the approach could lock onto a different same-class surface mid-drive. The
    reference is captured from the cached surface (survives the per-move invalidation)."""
    env = _Env(); api = CruzrApi(env)
    api._last_surface = {"reference": "water cup", **_surf(1000.0)}   # grounded sense at entry
    hases = []
    it = iter([_surf(1000.0), _surf(350.0)])   # far → drive → next in range

    def _sense(object_name="table", reference=None, relation="on"):
        hases.append(reference)
        return next(it)

    api.locate_for_place = _sense   # type: ignore[method-assign]
    out = api.approach_for_place("table")
    assert out["ok"] and out["status"] == "in_range"
    assert hases and all(h == "water cup" for h in hases)   # every re-sense stayed grounded


def test_ungrounded_resense_passes_no_reference():
    """No cached reference → re-sense must pass has=None so the ungrounded path is unchanged."""
    env = _Env(); api = CruzrApi(env)
    api._last_surface = None
    hases = []
    it = iter([_surf(1000.0), _surf(350.0)])

    def _sense(object_name="table", reference=None, relation="on"):
        hases.append(reference)
        return next(it)

    api.locate_for_place = _sense   # type: ignore[method-assign]
    out = api.approach_for_place("table")
    assert out["ok"]
    assert hases and all(h is None for h in hases)


def test_grounded_resense_degrades_to_plain_while_far_then_regrounds():
    """Symmetric to approach_for_grasp's degrade: the grounded 'surface has reference' sense resolves
    only up close, so while far the re-sense degrades to a PLAIN surface sense (carrying the
    reference) and keeps converging; near the place edge it re-grounds. Without the degrade an early
    plain handoff would abort on iter 1's far grounded miss instead of finishing the positioning."""
    env = _Env(); api = CruzrApi(env)
    api._last_surface = {"reference": "water cup", "relation": "under", **_surf(1000.0)}  # grounded at entry
    fronts = {0: 1000.0}                          # front_x while far; after 1 move (k>=1) → in-range 350
    st = {"k": 0}
    calls = []

    def _sense(object_name="table", reference=None, relation="on"):
        fx = fronts.get(st["k"], 350.0)
        calls.append((reference, round(fx)))
        if reference is not None and fx > 450.0:  # grounded can't resolve the relation while far
            return {"ok": False, "reason": "no_surface_matching_reference"}
        s = _surf(fx)
        if reference is not None:                 # grounded success carries the reference
            s["reference"] = reference
        return s

    def _nav(dx, dy, dyaw, **kw):                 # each drive closes the distance
        st["k"] += 1
        return {"ok": True, "yaw_reached": dyaw}

    api.locate_for_place = _sense                  # type: ignore[method-assign]
    api.env.low_level.navigate_relative = _nav    # type: ignore[method-assign]
    out = api.approach_for_place("table")
    assert out["ok"] and out["status"] == "in_range"
    assert (None, 1000) in calls                              # degraded to plain while far (grounded missed)
    assert any(h == "water cup" and fx <= 450 for (h, fx) in calls)   # re-grounded near the edge


def test_grounded_resense_still_aborts_when_plain_also_lost():
    """Both grounded and plain sense miss → approach_for_place aborts (never drives blind)."""
    env = _Env(); api = CruzrApi(env)
    api._last_surface = {"reference": "water cup", **_surf(700.0)}

    def _sense(object_name="table", reference=None, relation="on"):
        return {"ok": False, "reason": "no_surface"}

    api.locate_for_place = _sense                  # type: ignore[method-assign]
    out = api.approach_for_place("table")
    assert out["ok"] is False and out["reason"] == "no_surface"
    assert env.low_level.nav_calls == []                     # never drove
