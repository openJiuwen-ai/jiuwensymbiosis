# coding: utf-8
"""approach_for_grasp: SQUARE-FIRST converge loop — each iteration re-detects and drives the base until it
faces ONE of the box's vertical FACES at the grasp distance. The dual-arm clamp closes along base-Y and
ignores box yaw, so the base must square to a face for the paddles to land flat. On a differential base
(no strafe) chasing centring after squaring rotates the box back off-square — so the loop squares FIRST
(rotate in place, forward=0, to face a footprint-axis normal) then advances STRAIGHT in (turn=0), which
preserves the square. Which face is HYSTERESIS-LOCKED on the first footprint frame (``prev_normal``) and
only re-picked near that lock afterwards, so a jittery footprint yaw can't 90°-flip it — near-square and
~corner-on boxes now square to a face too (any face is an acceptable grasp face for a near-cube box that
fits the clamp either way), instead of the old radial fallback that could leave the base ~45° corner-on.
The accept tolerance is ``grasp_square_tol_rad`` (must exceed the footprint-yaw jitter, else a jittery box
only ever rotates and never advances). The plain ``_det`` helper below carries no footprint (no yaw_rad),
so it takes the plain radial centre-and-advance path — unchanged."""

import math
from types import SimpleNamespace

from jiuwensymbiosis.adapters.cruzr.api import CruzrApi
from jiuwensymbiosis.motion.approach import Footprint


class _LL:
    def __init__(self):
        self.calls = []
        self.kw = []            # per-call {k_rot, k_rot_slow_rad, k_fwd} overrides (gentle approach gains)
        self.nav_result = {"ok": True}

    def navigate_relative(self, dx_m, dy_m=0.0, dyaw_rad=0.0, *,
                          k_rot=None, k_rot_slow_rad=None, k_fwd=None):
        self.calls.append(("nav", round(float(dx_m), 4), round(float(dy_m), 4), round(float(dyaw_rad), 4)))
        self.kw.append({"k_rot": k_rot, "k_rot_slow_rad": k_rot_slow_rad, "k_fwd": k_fwd})
        return self.nav_result

    def navigate_arc(self, radius, dyaw, *, k_fwd=None):   # arc fast-path primitive (opt-in)
        self.calls.append(("arc", round(float(radius), 4), round(float(dyaw), 4)))
        return self.nav_result


def _make_cfg(**over):
    base = dict(
        grasp_target_forward_m=0.40, grasp_forward_min_m=0.30, grasp_forward_max_m=0.50,
        base_pos_tol_m=0.05, base_yaw_tol_rad=0.08, approach_converge_iters=4,
        approach_forward_step_m=0.25,    # per-step anti-lunge cap (radial fallback / single-shot final align)
        grasp_approach_forward_reserve_m=0.15,  # squared straight-in reserve (≤2 forward moves, gentle landing)
        approach_k_rot=1.5, approach_k_rot_slow_rad=0.5, approach_k_fwd=0.8,  # gentle approach-only gains
        grasp_square_tol_rad=0.26,       # in-place squaring accept: |square_turn| ≤ this ⇒ square enough
        # arc fast-path (OFF by default = current discrete behaviour; on only in the arc tests)
        grasp_arc_enabled=False, grasp_arc_standoff_m=0.10, grasp_arc_min_radius_m=0.35,
        grasp_arc_max_dyaw_rad=1.2, grasp_face_flatness_max=0.15, arc_k_fwd=0.6,
    )
    base.update(over)
    return SimpleNamespace(**base)


class _Env:
    def __init__(self, cfg=None):
        self.low_level = _LL()
        self.cfg = cfg if cfg is not None else _make_cfg()


def _det(x_mm, y_mm=0.0):
    return {"ok": True, "center_mm": [x_mm, y_mm, 780.0], "object": "box"}


def _script(api, dets):
    """Feed `dets` as the successive re-detections (square-up samples + post-move re-detects)."""
    it = iter(dets)
    api.locate_for_grasp = lambda object_name="box", reference=None, relation="on": next(it)   # type: ignore[method-assign]


# --------------------------------------------------------------------------------------------------
# Radial converge loop (plain _det → no footprint → square-up is a no-op; behaviour unchanged)
# --------------------------------------------------------------------------------------------------

def test_no_detection_searches_instead_of_failing_immediately():
    """With nothing cached, approach_for_grasp SEARCHES — it no longer errors out on the spot.

    That is the point of folding the face pass in: "I don't know where it is" is a
    reason to look, not a reason to give up. Only a search that comes up empty fails,
    and it fails with ``object_not_found`` rather than ``no_detection``.
    """
    api = CruzrApi(_Env()); api._last_detection = None
    api._nav._face_object = lambda name="box", reference=None, relation="on": {"ok": False, "reason": "object_not_found"}  # type: ignore[method-assign]
    out = api.approach_for_grasp()
    assert out == {"ok": False, "reason": "object_not_found"}
    assert api.env.low_level.calls == []


def test_cached_detection_skips_the_search_pass():
    """A detection already in hand means no sweep: the drive loop runs straight away."""
    api = CruzrApi(_Env()); api._last_detection = _det(400.0)
    searched = []
    api._nav._face_object = lambda name="box", reference=None, relation="on": searched.append(name) or {"ok": True}  # type: ignore[method-assign]
    out = api.approach_for_grasp()
    assert out["ok"] and searched == []


def test_in_band_first_iter_does_not_move():
    api = CruzrApi(_Env()); api._last_detection = _det(400.0)   # rng .4, centered → in_band
    out = api.approach_for_grasp()
    assert out["ok"] and out["status"] == "in_band" and out["iters"] == 1
    assert api.env.low_level.calls == []            # no move
    assert api._last_detection is not None           # converged detection cached


def test_offcenter_converges_over_iterations():
    api = CruzrApi(_Env())
    api._last_detection = _det(700.0, 200.0)         # far + off to the left
    # each drive brings it closer + more centered until in_band
    _script(api, [_det(500.0, 80.0), _det(420.0, 20.0), _det(400.0, 0.0)])
    out = api.approach_for_grasp()
    assert out["ok"] and out["status"] == "in_band"
    assert out["iters"] >= 2                          # took several correction steps
    assert len(api.env.low_level.calls) >= 2          # drove more than once
    assert api._last_detection is not None


def test_too_close_does_not_move():
    api = CruzrApi(_Env()); api._last_detection = _det(200.0)   # rng .2 < min .3 → too_close
    out = api.approach_for_grasp()
    assert out["ok"] is False and out["reason"] == "too_close"
    assert api.env.low_level.calls == []


def test_nav_failure_propagates_and_keeps_cache():
    api = CruzrApi(_Env())
    api.env.low_level.nav_result = {"ok": False, "reason": "lidar_blocked"}
    api._last_detection = _det(700.0)
    out = api.approach_for_grasp()
    assert out["ok"] is False and out["reason"] == "lidar_blocked"
    assert api._last_detection is not None            # move failed → keep detection


def test_lost_after_move_reports():
    api = CruzrApi(_Env()); api._last_detection = _det(700.0)
    _script(api, [{"ok": False, "reason": "no_detection"}])   # re-detect fails after the drive
    out = api.approach_for_grasp()
    assert out["ok"] is False and out["reason"] == "no_detection"
    assert len(api.env.low_level.calls) == 1          # drove once, then lost it


def test_grounding_sticky_across_redetects():
    """A grounded box (locate_for_grasp with a reference) must stay grounded on every post-move
    re-detect, else the approach could lock onto a different same-class object mid-drive. The
    reference rides in the cached detection; _redetect threads it through each re-detect."""
    api = CruzrApi(_Env())
    api._last_detection = {"ok": True, "center_mm": [700.0, 200.0, 780.0],
                           "object": "white box", "reference": "brown table"}  # far + off-centre
    ons = []
    it = iter([_det(500.0, 80.0), _det(420.0, 20.0), _det(400.0, 0.0)])

    def _detect(object_name="box", reference=None, relation="on"):
        ons.append(reference)
        return next(it)

    api.locate_for_grasp = _detect   # type: ignore[method-assign]
    out = api.approach_for_grasp()
    assert out["ok"] and out["status"] == "in_band"
    assert ons and all(o == "brown table" for o in ons)   # every re-detect stayed grounded


def test_approach_drives_use_gentle_base_gains():
    """Every fine-positioning drive must carry the gentle approach_k_* overrides (so it can't
    overshoot and fling the box out of the waist camera's edge). Search big-turns keep the fast
    global base_k_* — those go through rotate_base, not _drive_base, so they carry no overrides."""
    api = CruzrApi(_Env())
    api._last_detection = _det(700.0, 200.0)         # far + off-centre → at least one drive
    _script(api, [_det(500.0, 80.0), _det(420.0, 20.0), _det(400.0, 0.0)])
    out = api.approach_for_grasp()
    assert out["ok"]
    assert api.env.low_level.kw                        # drove at least once
    assert all(k == {"k_rot": 1.5, "k_rot_slow_rad": 0.5, "k_fwd": 0.8}
               for k in api.env.low_level.kw)          # gentle gains on every approach drive


def test_forward_advance_reserves_final_decel_segment():
    """A far box is closed in AT MOST TWO forward moves — not a single full-distance lunge (which
    would drive INTO the box after the early ~1.2-1.5m handoff) and not many tiny per-0.25m capped
    steps (each an extra wheel_worker subprocess → the reported choppiness). The far move reserves
    the final approach_forward_step_m (drives forward-reserve, stopping that reserve short of the
    grasp band), then the last move drives the whole ≤reserve remainder as a gentle landing."""
    api = CruzrApi(_Env())
    reserve = api.env.cfg.approach_forward_step_m           # 0.25
    api._last_detection = _det(900.0)                       # rng .9, centred → raw forward .5 (far)
    _script(api, [_det(650.0), _det(400.0)])               # .25 forward (near) → in band
    out = api.approach_for_grasp()
    assert out["ok"] and out["status"] == "in_band"
    dxs = [c[1] for c in api.env.low_level.calls if c[0] == "nav"]
    assert len(dxs) == 2                                    # ≤2 forward moves (was up to 3-6 under the cap)
    assert math.isclose(dxs[0], 0.5 - reserve, abs_tol=1e-9)   # far: reserved the final segment, not the full .5
    assert math.isclose(dxs[-1], 0.25, abs_tol=1e-9)          # near: whole remainder as the gentle landing


def test_grounded_redetect_degrades_to_plain_while_far_then_regrounds():
    """After an EARLY plain coarse handoff the grounded 'on surface' re-detect still misses while
    far — approach_for_grasp must NOT abort. It degrades to a plain re-detect (carrying the reference)
    and keeps converging; once near the band the grounded relation resolves again. This restores the
    multi-step waist fine-tuning that a grounded-only handoff (resolving only at grasp distance)
    swallowed."""
    api = CruzrApi(_Env())
    api._last_detection = {"ok": True, "center_mm": [900.0, 0.0, 780.0],
                           "object": "white box", "reference": "brown table"}   # far, centred
    centers = {1: (600.0, 0.0), 2: (430.0, 0.0)}   # box centre after k base-moves (else final 400)
    st = {"k": 0}
    calls = []

    def _detect(object_name="box", reference=None, relation="on"):
        x, y = centers.get(st["k"], (400.0, 0.0))
        calls.append((reference, round(x)))
        if reference is not None and x > 450.0:            # grounded can't resolve the relation while far
            return {"ok": False, "reason": "no_target_on_reference"}
        d = {"ok": True, "center_mm": [x, y, 780.0], "object": object_name}
        if reference is not None:                   # grounded success carries the reference (real _detect_on_surface)
            d["reference"] = reference
        return d

    def _drive(forward, turn, **kw):                # each correction step closes the distance
        st["k"] += 1
        return {"ok": True}

    api.locate_for_grasp = _detect   # type: ignore[method-assign]
    api._nav.drive_base = _drive         # type: ignore[method-assign]
    out = api.approach_for_grasp()
    assert out["ok"] and out["status"] == "in_band"
    assert (None, 600) in calls                          # degraded to plain while far (grounded missed)
    assert any(o == "brown table" and x <= 450 for (o, x) in calls)   # re-grounded near the band
    assert api._last_detection is not None and api._last_detection.get("reference") == "brown table"


def test_grounded_redetect_still_aborts_when_plain_also_lost():
    """The degrade is only a range tolerance, not a blanket 'never fail': if BOTH the grounded and
    the plain re-detect miss (target genuinely gone), the approach still reports lost_after_move."""
    api = CruzrApi(_Env())
    api._last_detection = {"ok": True, "center_mm": [700.0, 0.0, 780.0],
                           "object": "white box", "reference": "brown table"}

    def _detect(object_name="box", reference=None, relation="on"):
        return {"ok": False, "reason": "no_detection"}   # both grounded and plain miss

    api.locate_for_grasp = _detect   # type: ignore[method-assign]
    out = api.approach_for_grasp()
    assert out["ok"] is False and out["reason"] == "no_detection"
    assert len(api.env.low_level.calls) == 1             # drove once, then lost it entirely


def test_ungrounded_redetect_passes_no_reference():
    """A plain (non-grounded) detection has no reference → re-detect must pass on=None so the
    ungrounded path is unchanged (no regression)."""
    api = CruzrApi(_Env())
    api._last_detection = _det(700.0, 200.0)   # no 'reference' key
    ons = []
    it = iter([_det(500.0, 80.0), _det(400.0, 0.0)])

    def _detect(object_name="box", reference=None, relation="on"):
        ons.append(reference)
        return next(it)

    api.locate_for_grasp = _detect   # type: ignore[method-assign]
    out = api.approach_for_grasp()
    assert out["ok"]
    assert ons and all(o is None for o in ons)


# --------------------------------------------------------------------------------------------------
# Square-first branch (footprint near-face → rotate in place to face the front, then straight in)
# --------------------------------------------------------------------------------------------------

def test_footprint_squares_up_in_place_before_grasp():
    """A positioned + centred but yawed footprint makes approach_for_grasp rotate the base IN PLACE to face
    the near-face normal (square up) before declaring in_band. Critically it is a PURE rotation (dx==0) —
    never a forward lunge into the box — and its turn is atan2 over a UNIT normal (no degeneracy)."""
    api = CruzrApi(_Env())
    # in band (rng .4) + centred, but the footprint long axis is yawed ~0.5 rad → near-face normal is
    # tilted → must square up in place (square_turn ≈ 0.5 > tol), not grasp head-on.
    api._last_detection = {"ok": True, "center_mm": [400.0, 0.0, 780.0], "object": "box",
                           "yaw_rad": 0.5, "long_mm": 300.0, "short_mm": 200.0}
    # after the in-place square-up, the re-detect lands square (yaw≈0) + in band → grasp
    _script(api, [{"ok": True, "center_mm": [400.0, 0.0, 780.0], "object": "box",
                   "yaw_rad": 0.0, "long_mm": 300.0, "short_mm": 200.0}])
    out = api.approach_for_grasp()
    assert out["ok"] and out["status"] == "in_band"
    assert api.env.low_level.calls, "expected an in-place square-up turn, not an immediate in_band"
    dx, _dy, dyaw = (api.env.low_level.calls[0][1], api.env.low_level.calls[0][2],
                     api.env.low_level.calls[0][3])
    assert dx == 0.0                                  # squaring is a PURE rotation — never drives into the box
    assert abs(dyaw) > 0.15                           # carried the real footprint square turn (~0.5 rad)


def test_near_square_footprint_now_squares_up():
    """A NEAR-square footprint (aspect 1.05) now ALSO squares to a face — the hysteresis lock removes the
    old ill-conditioned-yaw oscillation (logs: aspect≈1.05 yaw 18°→−11°→14°→−13°) that the aspect gate
    used to dodge by falling back to radial. The old radial fallback could leave the base ~45° corner-on;
    now a centred in-band near-square box rotates IN PLACE to face a footprint axis (dx==0, turn≠0)."""
    api = CruzrApi(_Env())
    api._last_detection = {"ok": True, "center_mm": [400.0, 0.0, 780.0], "object": "box",
                           "yaw_rad": 0.5, "long_mm": 210.0, "short_mm": 200.0}   # aspect 1.05 (near-square)
    _script(api, [{"ok": True, "center_mm": [400.0, 0.0, 780.0], "object": "box",
                   "yaw_rad": 0.0, "long_mm": 210.0, "short_mm": 200.0}])         # squared after the turn
    out = api.approach_for_grasp()
    assert out["ok"] and out["status"] == "in_band"
    assert api.env.low_level.calls, "near-square should now square up, not fall back to a no-move radial grasp"
    assert api.env.low_level.calls[0][1] == 0.0 and abs(api.env.low_level.calls[0][3]) > 0.15  # pure in-place turn


def test_corner_on_view_now_squares_to_a_face():
    """A ~corner-on view (box→robot line bisects two adjacent faces) used to fall back to radial (leaving
    the base facing the CORNER, ~45° off any face). Now it picks ONE of the two faces and squares to it;
    the hysteresis lock keeps it from flipping between the two under noise. Any face is an acceptable grasp
    face, so squaring to a face beats grasping a corner."""
    api = CruzrApi(_Env())
    api._last_detection = {"ok": True, "center_mm": [400.0, 0.0, 780.0], "object": "box",
                           "yaw_rad": math.pi / 4, "long_mm": 300.0, "short_mm": 200.0}  # 45° corner-on
    _script(api, [{"ok": True, "center_mm": [400.0, 0.0, 780.0], "object": "box",
                   "yaw_rad": 0.0, "long_mm": 300.0, "short_mm": 200.0}])         # squared after the turn
    out = api.approach_for_grasp()
    assert out["ok"] and out["status"] == "in_band"
    assert api.env.low_level.calls, "corner-on should now square to a face, not fall back to a no-move radial grasp"
    assert api.env.low_level.calls[0][1] == 0.0 and abs(api.env.low_level.calls[0][3]) > 0.15  # pure in-place turn


def test_square_first_then_straight_in_preserves_squareness():
    """The ordering guarantee: a FAR + yawed box squares to the front IN PLACE first (dx==0, turn≠0),
    THEN advances STRAIGHT in (dx>0, turn==0) — never a turn+forward lunge (the collision mode), and the
    straight-in step never re-introduces yaw so the square is preserved down to the grasp distance."""
    api = CruzrApi(_Env())
    api._last_detection = {"ok": True, "center_mm": [800.0, 0.0, 780.0], "object": "box",
                           "yaw_rad": 0.5, "long_mm": 300.0, "short_mm": 200.0}   # far (rng .8) + yawed
    # re-detect 1: square (yaw 0) but still far → straight-in; re-detect 2: at the band → grasp
    _script(api, [{"ok": True, "center_mm": [800.0, 0.0, 780.0], "object": "box",
                   "yaw_rad": 0.0, "long_mm": 300.0, "short_mm": 200.0},
                  {"ok": True, "center_mm": [400.0, 0.0, 780.0], "object": "box",
                   "yaw_rad": 0.0, "long_mm": 300.0, "short_mm": 200.0}])
    out = api.approach_for_grasp()
    assert out["ok"] and out["status"] == "in_band"
    assert len(api.env.low_level.calls) == 2
    assert api.env.low_level.calls[0][1] == 0.0 and abs(api.env.low_level.calls[0][3]) > 0.15  # square in place
    assert api.env.low_level.calls[1][1] > 0.0 and api.env.low_level.calls[1][3] == 0.0        # straight in (turn=0)


def test_squared_straight_in_reserves_final_segment_no_per_step_chop():
    """After squaring, a LONG straight-in leg closes in AT MOST TWO forward moves (reserve strategy),
    not the old per-``approach_forward_step_m`` chopping (each cap step ends in a full base STOP + a waist
    re-detect → the reported straight-line stutter). Here the squared box starts rng 1.3 → raw forward .9;
    the reserve drives forward−reserve in one continuous move then a ≤reserve landing. Under the old 0.25
    hard cap this same .9 would take ~4 stop-and-redetect steps."""
    api = CruzrApi(_Env())
    reserve = api.env.cfg.grasp_approach_forward_reserve_m       # 0.15
    api._last_detection = {"ok": True, "center_mm": [1300.0, 0.0, 780.0], "object": "box",
                           "yaw_rad": 0.0, "long_mm": 300.0, "short_mm": 200.0}   # far (rng 1.3) + already square
    _script(api, [{"ok": True, "center_mm": [550.0, 0.0, 780.0], "object": "box",
                   "yaw_rad": 0.0, "long_mm": 300.0, "short_mm": 200.0},          # after the reserved move
                  {"ok": True, "center_mm": [400.0, 0.0, 780.0], "object": "box",
                   "yaw_rad": 0.0, "long_mm": 300.0, "short_mm": 200.0}])         # in band
    out = api.approach_for_grasp()
    assert out["ok"] and out["status"] == "in_band"
    dxs = [c[1] for c in api.env.low_level.calls if c[1] > 0.0]   # forward moves (square turns are dx==0)
    assert len(dxs) == 2                                          # ≤2 moves (was ~4 under the 0.25 cap = stutter)
    assert math.isclose(dxs[0], 0.9 - reserve, abs_tol=1e-9)      # far: one continuous move, reserved the final bit
    assert math.isclose(dxs[-1], 0.15, abs_tol=1e-9)             # near: whole ≤reserve remainder, gentle landing


def test_near_square_jittery_yaw_advances_without_deadlock():
    """A FAR near-square box with a jittery footprint yaw must still make forward progress. Because the
    accept tolerance ``grasp_square_tol_rad`` (0.26) exceeds the ~0.26 yaw jitter, a roughly-squared frame
    ADVANCES instead of re-rotating forever — the deadlock the old tight place-tol (0.15) would cause once
    near-square boxes are squared (|square_turn| ≈ jitter > tol every frame ⇒ only ever rotate)."""
    api = CruzrApi(_Env())
    api._last_detection = {"ok": True, "center_mm": [800.0, 0.0, 780.0], "object": "box",
                           "yaw_rad": 0.3, "long_mm": 210.0, "short_mm": 200.0}        # far + near-square
    _script(api, [{"ok": True, "center_mm": [800.0, 0.0, 780.0], "object": "box",
                   "yaw_rad": -0.25, "long_mm": 210.0, "short_mm": 200.0},            # jitter, still far
                  {"ok": True, "center_mm": [500.0, 0.0, 780.0], "object": "box",
                   "yaw_rad": 0.2, "long_mm": 210.0, "short_mm": 200.0},              # advanced
                  {"ok": True, "center_mm": [400.0, 0.0, 780.0], "object": "box",
                   "yaw_rad": 0.0, "long_mm": 210.0, "short_mm": 200.0}])             # in band
    out = api.approach_for_grasp()
    assert out["ok"] and out["status"] == "in_band"
    forward_moves = [c for c in api.env.low_level.calls if c[1] > 0.0]
    assert forward_moves, "far near-square box must advance, not rotate-in-place forever (deadlock)"


def test_grasp_near_face_normal_hysteresis_prevents_face_flip():
    """Unit test of the hysteresis directly. At a ~corner-on view (box→robot ~45° to the footprint axes)
    the two nearest faces nearly tie, so a tiny yaw sign flip makes the unlocked (most-aligned) pick jump
    to the PERPENDICULAR face. Passing ``prev_normal`` locks the pick to the face nearest the previous
    normal, so the same jittered frame stays on the original face — no 90° flip."""
    from jiuwensymbiosis.adapters.cruzr.api import _grasp_near_face_normal

    class _cfg:                      # _grasp_near_face_normal no longer reads any cfg attribute
        pass

    cfg = _cfg()
    center = [400.0, 400.0, 780.0]                          # box→robot ~45° to the footprint axes
    n0 = _grasp_near_face_normal(Footprint(center, 0.05, 300.0, 200.0), cfg, prev_normal=None)
    n_flip = _grasp_near_face_normal(Footprint(center, -0.05, 300.0, 200.0), cfg, prev_normal=None)
    assert n0 is not None and n_flip is not None
    assert n0[0] * n_flip[0] + n0[1] * n_flip[1] < 0.2      # unlocked → ~90° flip between the two faces
    n_lock = _grasp_near_face_normal(Footprint(center, -0.05, 300.0, 200.0), cfg, prev_normal=n0)
    assert n0[0] * n_lock[0] + n0[1] * n_lock[1] > 0.9      # locked → same face, no flip


# --------------------------------------------------------------------------------------------------
# Single-shot right-angle approach (opt-in via grasp_arc_enabled): L-route onto the face-normal line,
# then ONE final re-detect + a single alignment — NO per-step re-detect polling.
# --------------------------------------------------------------------------------------------------

def test_single_shot_on_line_straight_in():
    """Box already ON the normal line (its face normal points straight back at the robot) → no lateral
    leg; the single-shot drives the approach leg straight in, then ONE re-detect → in_band (no polling)."""
    api = CruzrApi(_Env(_make_cfg(grasp_arc_enabled=True)))
    api._last_detection = {"ok": True, "center_mm": [1000.0, 0.0, 780.0], "object": "box",
                           "face_normal": [-1.0, 0.0], "face_flatness": 0.05}
    _script(api, [{"ok": True, "center_mm": [400.0, 0.0, 780.0], "object": "box",   # on line, in band
                   "yaw_rad": 0.0, "long_mm": 300.0, "short_mm": 200.0}])
    out = api.approach_for_grasp()
    assert out["ok"] and out["status"] == "in_band"
    calls = api.env.low_level.calls
    assert not any(c[0] == "arc" for c in calls)                  # no arc — right-angle route
    navs = [c for c in calls if c[0] == "nav"]
    assert len(navs) == 1 and navs[0][1] > 0.0 and abs(navs[0][3]) < 1e-6   # one straight approach, no turn


def test_single_shot_right_angle_off_line():
    """Off the line (corner-on): the single-shot drives a RIGHT-ANGLE route — a LATERAL leg (⊥ to the
    normal, onto the line) then an APPROACH leg (turn to face the box, straight in) — no arc. Then ONE
    final re-detect lands it in the band → cache, done."""
    api = CruzrApi(_Env(_make_cfg(grasp_arc_enabled=True)))
    api._last_detection = {"ok": True, "center_mm": [1000.0, 0.0, 780.0], "object": "box",
                           "face_normal": [-0.866, 0.5], "face_flatness": 0.05}   # 30° corner-on
    _script(api, [{"ok": True, "center_mm": [400.0, 0.0, 780.0], "object": "box",   # on line, in band
                   "yaw_rad": 0.0, "long_mm": 300.0, "short_mm": 200.0}])
    out = api.approach_for_grasp()
    assert out["ok"] and out["status"] == "in_band"
    calls = api.env.low_level.calls
    assert not any(c[0] == "arc" for c in calls)
    navs = [c for c in calls if c[0] == "nav"]
    assert len(navs) == 2                                  # right-angle: lateral leg + approach leg
    assert navs[0][1] > 0.0 and abs(navs[0][3]) > 0.15     # lateral: turn ⊥ to normal (~60°) + straight
    assert navs[1][1] > 0.0 and abs(navs[1][3]) > 0.15     # approach: turn to face the box + straight in


def test_single_shot_one_final_alignment():
    """When the open-loop route lands a bit off, the single-shot does exactly ONE alignment move (square +
    straight-in) from the final re-detect, then caches — never a polling loop of drive→re-detect."""
    api = CruzrApi(_Env(_make_cfg(grasp_arc_enabled=True)))
    api._last_detection = {"ok": True, "center_mm": [1000.0, 0.0, 780.0], "object": "box",
                           "face_normal": [-0.866, 0.5], "face_flatness": 0.05}   # corner-on → 2-leg route
    _script(api, [{"ok": True, "center_mm": [600.0, 0.0, 780.0], "object": "box",   # post-route: short + tilted
                   "face_normal": [-0.95, 0.31], "face_flatness": 0.05},
                  {"ok": True, "center_mm": [400.0, 0.0, 780.0], "object": "box",   # after align → cache
                   "face_normal": [-1.0, 0.0], "face_flatness": 0.05}])
    out = api.approach_for_grasp()
    assert out["ok"] and out["status"] == "in_band"
    navs = [c for c in api.env.low_level.calls if c[0] == "nav"]
    assert len(navs) == 3                                  # 2 route legs + exactly ONE alignment (no polling)
    assert navs[2][1] > 0.0 and abs(navs[2][3]) > 0.0      # the align: square (turn) + straight-in (forward)


def test_single_shot_too_close_after_route_aborts():
    """If the open-loop route overshoots INSIDE the grasp band (what grasp_arc_standoff_m reserves against),
    the ONE final re-detect reports too_close → single-shot aborts instead of grasping blind."""
    api = CruzrApi(_Env(_make_cfg(grasp_arc_enabled=True)))
    api._last_detection = {"ok": True, "center_mm": [1000.0, 0.0, 780.0], "object": "box",
                           "face_normal": [-0.866, 0.5], "face_flatness": 0.05}
    _script(api, [{"ok": True, "center_mm": [200.0, 0.0, 780.0], "object": "box",   # overshot: rng .2 < min .30
                   "face_normal": [-1.0, 0.0], "face_flatness": 0.05}])
    out = api.approach_for_grasp()
    assert out["ok"] is False and out["reason"] == "too_close"
    navs = [c for c in api.env.low_level.calls if c[0] == "nav"]
    assert len(navs) == 2                                  # route legs ran, then aborted — no align move


def test_single_shot_declines_on_untrusted_normal():
    """No trustworthy normal (plain detection: no point-cloud face_normal, no footprint yaw) → single-shot
    DECLINES before any motion → falls back to the discrete polling loop, which closes it step by step."""
    api = CruzrApi(_Env(_make_cfg(grasp_arc_enabled=True)))
    api._last_detection = _det(700.0)                            # plain → _select_grasp_normal returns None
    _script(api, [_det(400.0)])
    out = api.approach_for_grasp()
    assert out["ok"]
    assert not any(c[0] == "arc" for c in api.env.low_level.calls)


def test_single_shot_grounded_requires_confirmation_before_moving():
    """Grounded 'X on Y': the open-loop single-shot must CONFIRM the target is on its reference (a strict
    grounded re-detect: target + reference detected + the on-surface relation) BEFORE any move. If that
    fails, it aborts — never drives toward an unconfirmed target (e.g. the bare reference picked as it)."""
    api = CruzrApi(_Env(_make_cfg(grasp_arc_enabled=True)))
    api._last_detection = {"ok": True, "center_mm": [1000.0, 0.0, 780.0], "object": "white bin",
                           "reference": "brown box", "face_normal": [-0.866, 0.5], "face_flatness": 0.05}
    _script(api, [{"ok": False, "reason": "no_target_on_reference"}])   # grounded confirm fails
    out = api.approach_for_grasp()
    assert out["ok"] is False and out["reason"] == "no_target_on_reference"
    assert api.env.low_level.calls == []                          # NO move — target not confirmed on reference


def test_single_shot_grounded_confirmed_then_moves():
    """When the grounded confirm succeeds, the single-shot proceeds to plan + drive the right-angle route
    from the CONFIRMED detection."""
    api = CruzrApi(_Env(_make_cfg(grasp_arc_enabled=True)))
    api._last_detection = {"ok": True, "center_mm": [1000.0, 0.0, 780.0], "object": "white bin",
                           "reference": "brown box", "face_normal": [-0.866, 0.5], "face_flatness": 0.05}
    _script(api, [
        {"ok": True, "center_mm": [1000.0, 0.0, 780.0], "object": "white bin", "reference": "brown box",
         "face_normal": [-0.866, 0.5], "face_flatness": 0.05},    # grounded confirm OK (corner-on)
        {"ok": True, "center_mm": [400.0, 0.0, 780.0], "object": "white bin", "reference": "brown box",
         "yaw_rad": 0.0, "long_mm": 300.0, "short_mm": 200.0},    # post-route re-detect → in_band
    ])
    out = api.approach_for_grasp()
    assert out["ok"] and out["status"] == "in_band"
    assert any(c[0] == "nav" for c in api.env.low_level.calls)    # moved (right-angle) only after confirmation

