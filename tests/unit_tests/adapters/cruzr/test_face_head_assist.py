# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Head+waist coordinated search for _face_by_sweep (shared by face_object / face_surface).

The narrow, short-range waist RGBD can't resolve a far target; the wide-FOV head camera gives a
bearing only. So the search tries the waist first (precise 3-D → align now) and, on a miss, PANS the
head to scan the current facing (look left + right once each). A head hit hands off to a coarse
approach that turns to the bearing and CREEPS FORWARD continuously (``--forward`` worker) while
steering by the target's true base-frame bearing (live head yaw + in-image bearing) and yaw-tracking
the head to keep the target centred, until the waist acquires. If the head scan finds nothing, the base rotates 180° ONCE and the head
re-scans the half behind us. Fully mocked: real rotate_base path (fake LL) + mock drive worker +
monkeypatched waist/head detectors. The head is set to 2-look (``[0.6, -0.6]``) so per-scan
head-call counts are deterministic (the deployed "left + right once" config).
"""

import math

from jiuwensymbiosis.adapters.cruzr.api import CruzrApi
from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig


class _LL:
    """Fake low-level: records nav commands + a mock continuous-drive worker (reports ``running``
    for ``drive_polls`` polls, then finishes on its own). ``start_base_spin`` is present but must
    NEVER be called — the head pan-scan replaced the old continuous base spin."""

    def __init__(self, drive_polls=0):
        self.nav_calls = []
        self.spin_started = 0
        self.frames_asked = []          # cameras the look-around actually read
        self._drive_polls = drive_polls
        self._drive_count = 0
        self.drive_started = 0
        self.drive_stopped = 0
        self.steer_calls = []   # head bearings fed to the forward worker during the creep
        self.hold_calls = 0     # times the creep was paused because the head lost the anchor

    def grab_frames(self, camera="waist"):
        """No image: these tests script the SEARCH RESULT, so the look-around must fail open
        to whatever the test stubbed rather than depend on a rendered frame."""
        self.frames_asked.append(camera)
        return None

    def navigate_relative(self, dx, dy, dyaw):
        self.nav_calls.append((round(dx, 4), round(dy, 4), round(dyaw, 4)))
        return {"ok": True, "yaw_reached": dyaw}

    def start_base_spin(self, direction=1.0):
        self.spin_started += 1        # asserted 0 everywhere: pan-scan must not spin
        return object()

    def start_base_drive(self):
        self.drive_started += 1
        self._drive_count = 0
        return object()

    def base_drive_running(self, handle):
        self._drive_count += 1
        return self._drive_count <= self._drive_polls

    def steer_base_drive(self, handle, bearing_rad):
        self.steer_calls.append(round(float(bearing_rad), 4))

    def hold_base_drive(self, handle):
        self.hold_calls += 1

    def stop_base_drive(self, handle):
        self.drive_stopped += 1
        return {"ok": True, "dist_traveled": 1.0}


class _Env:
    def __init__(self, drive_polls=0):
        self.low_level = _LL(drive_polls=drive_polls)
        self.cfg = CruzrConfig()   # real defaults: face tol 0.10, lost_max 3
        self.cfg.head_search_yaw_positions_rad = (0.6, -0.6)   # deployed 2-look (left + right once)


def _api(waist_seq, head_seq, *, side="object", drive_polls=0):
    """CruzrApi with the waist detector (locate_for_grasp / locate_for_place) and the head detector
    (search_target) each yielding a fixed sequence; nav + drive are the fake LL."""
    env = _Env(drive_polls=drive_polls)
    api = CruzrApi(env)
    wit = iter(waist_seq)
    hit = iter(head_seq)
    if side == "object":
        api.locate_for_grasp = lambda object_name="box", reference=None, relation="on": next(wit)  # type: ignore[method-assign]
    else:
        api.locate_for_place = lambda object_name="table", reference=None, relation="on": next(wit)  # type: ignore[method-assign]
    api._nav._look_for = lambda object_name="box", on=None, camera=None: next(hit)           # type: ignore[method-assign]
    api.set_head = lambda yaw_rad, pitch_rad: {"ok": True}                     # type: ignore[method-assign]
    return api, env


def _wok(cx, cy):
    return {"ok": True, "center_mm": [cx, cy, 0.0], "surface_z_mm": 700.0}


_WNF = {"ok": False, "reason": "no_detection"}


def _hfound(bearing=0.0):
    return {"ok": True, "found": True, "bearing_rad": bearing, "image_h": 100, "image_w": 200,
            "bbox": [0, 0, 50, 50], "score": 0.9}


_HNF = {"ok": True, "found": False, "reason": "no_detection", "image_h": 100, "image_w": 200}

# Grounded head reject: reference resolves but no target overlaps it. ``_grnd_miss_seen`` additionally
# carries where the (colour-verified) target WAS seen, off the reference, so the drive loop re-aims
# the head toward it (base held) to recover the on-reference overlap next poll.
_GRND_MISS = {"ok": True, "found": False, "verified": True, "reason": "no_target_on_reference",
              "image_h": 100, "image_w": 200}


def _grnd_miss_seen(seen_bearing):
    return {**_GRND_MISS, "seen_bearing_rad": seen_bearing}

_PI4 = round(math.pi, 4)   # the 180° fallback as recorded by the (4-dp-rounded) fake nav


def test_waist_current_view_hits_no_head_scan():
    # The waist resolves the box centred in the current view → align immediately; the head is never
    # panned and the base never rotates 180°.
    api, env = _api(waist_seq=[_wok(1000, 0)], head_seq=[])
    out = api._nav._face_object("box")
    assert out["ok"] and out["status"] == "acquired"
    assert env.low_level.spin_started == 0
    assert env.low_level.drive_started == 0
    assert env.low_level.nav_calls == []          # acquire only → never rotates (approach owns alignment)


def test_front_panscan_found_then_coarse_approach():
    # Waist blind up front; the FRONT head pan-scan sees the box (bearing 0.3) → coarse-approach
    # (turn once, drive forward) until the waist acquires → align. No 180°.
    api, env = _api(
        waist_seq=[_WNF,            # step1 current-view waist miss
                   _WNF,            # drive poll1 waist miss → keep creeping
                   _wok(1000, 0),   # drive poll2 waist ACQUIRED → handoff (drive halts)
                   _wok(1000, 200)],  # fresh re-detect for align
        head_seq=[_hfound(bearing=0.3),   # front scan look1: found → coarse approach
                  _hfound()],              # drive poll1 head: still sees anchor
        side="object", drive_polls=5)
    out = api._nav._face_object("box")
    assert out["ok"] and out["status"] == "acquired"
    assert env.low_level.spin_started == 0
    assert env.low_level.drive_started == 1           # one continuous forward creep
    navs = env.low_level.nav_calls
    assert navs == [(0.0, 0.0, 0.9)]                   # ONLY the coarse turn (head_yaw 0.6 + bearing 0.3):
    assert (0.0, 0.0, _PI4) not in navs               # no 180°, and NO centroid-align turn after acquire


def test_coarse_approach_is_waist_only_no_head_steer():
    # Waist-only creep (head detection removed 2026-07-28): after the initial turn the base drives
    # STRAIGHT and polls ONLY the waist — the worker is never steered, and the head is not consulted
    # mid-creep. head_seq carries JUST the one pan-scan hit; if the creep still polled the head it would
    # exhaust the iterator (StopIteration) — this test passing proves the creep does not.
    api, env = _api(
        waist_seq=[_WNF,             # step1 current-view waist miss
                   _WNF,             # drive poll1 waist miss → keep creeping (no head consulted)
                   _wok(1000, 0),    # drive poll2 waist ACQUIRED → handoff
                   _wok(1000, 200)],  # fresh re-detect at handoff
        head_seq=[_hfound(bearing=0.3)],   # ONLY the pan-scan uses the head; the creep never does
        side="object", drive_polls=5)
    out = api._nav._face_object("box")
    assert out["ok"] and out["status"] == "acquired"
    assert env.low_level.drive_started == 1 and env.low_level.drive_stopped == 1
    assert env.low_level.steer_calls == []          # drives straight on the initial bearing — never steered


def test_front_miss_then_180_then_back_found():
    # Front pan-scan finds nothing → rotate 180° ONCE → the BACK pan-scan finds it (bearing 0.4) →
    # coarse-approach until the waist acquires → align.
    api, env = _api(
        waist_seq=[_WNF,            # step1 current-view waist miss
                   _WNF,            # drive poll1 waist miss
                   _wok(1000, 0),   # drive poll2 waist ACQUIRED
                   _wok(1000, 150)],  # fresh for align
        head_seq=[_HNF, _HNF,             # front scan: 2 looks, both miss
                  _hfound(bearing=0.4),   # back scan look1: found
                  _hfound()],              # drive poll1 head keep
        side="surface", drive_polls=5)
    out = api._nav._face_surface("table")
    assert out["ok"] and out["status"] == "acquired"
    assert env.low_level.spin_started == 0
    assert env.low_level.drive_started == 1
    navs = env.low_level.nav_calls
    assert navs[0] == (0.0, 0.0, _PI4)                # the 180° fallback comes first
    assert navs[1] == (0.0, 0.0, 1.0)                 # coarse turn = head_yaw 0.6 + in-image bearing 0.4
    assert len(navs) == 2                             # ...and nothing after — no centroid-align turn
    assert out["turned_rad"] >= math.pi               # includes the 180° turn


def test_all_miss_returns_not_found():
    # Waist blind and both head pan-scans (front + 180° + back) find nothing → fail safe with
    # ``panscan_exhausted``; caller must NOT grasp.
    api, env = _api(
        waist_seq=[_WNF],                       # step1 waist miss; no coarse approach (head misses)
        head_seq=[_HNF, _HNF, _HNF, _HNF],      # 2 front looks + 2 back looks, all miss
        side="object", drive_polls=0)
    out = api._nav._face_object("box")
    assert out["ok"] is False
    assert out["reason"] == "object_not_found"
    assert out.get("note") == "panscan_exhausted"
    assert math.isclose(out["turned_rad"], math.pi, rel_tol=1e-6)
    assert env.low_level.spin_started == 0
    assert env.low_level.drive_started == 0
    assert env.low_level.nav_calls == [(0.0, 0.0, _PI4)]   # only the 180° fallback


def test_front_head_seen_but_waist_never_acquires_fails_safe():
    # The FRONT head-scan sees the box, but the waist-only creep can't get it into the waist view (bad
    # open-loop turn / target beyond reach) → fail safe with a note; NO 180° (we already drove toward
    # it). Waist-only: the base drives straight until the worker self-stops (no head-lost hold/abort),
    # then ONE final post-stop waist look also misses → give up.
    api, env = _api(
        waist_seq=[_WNF,                   # step1 waist miss
                   _WNF, _WNF, _WNF,        # 3 drive polls: waist never acquires
                   _WNF],                   # final post-stop look: still misses
        head_seq=[_hfound(bearing=0.2)],   # front scan look1: found → coarse (creep consults head no more)
        side="object", drive_polls=3)
    out = api._nav._face_object("box")
    assert out["ok"] is False
    assert out["reason"] == "object_not_found"
    assert out.get("note") == "head_seen_no_waist_acquire"
    assert math.isclose(out["turned_rad"], 0.0, abs_tol=1e-9)
    assert env.low_level.spin_started == 0
    assert env.low_level.drive_started == 1 and env.low_level.drive_stopped == 1
    assert env.low_level.hold_calls == 0                     # waist-only: never pauses on a head-lost poll
    assert env.low_level.steer_calls == []                  # drives straight — never steered
    assert (0.0, 0.0, _PI4) not in env.low_level.nav_calls   # no 180° after a front head hit


def test_final_waist_detect_rescues_after_worker_selfstop():
    # The worker self-stops (lidar standoff / generous safety cap) with the head STILL tracking but
    # the in-loop waist polls all missed. The base closed the last stretch after the final poll, so
    # ONE final waist look after stopping now acquires → handoff succeeds instead of a false give-up.
    api, env = _api(
        waist_seq=[_WNF,            # step1 current-view waist miss
                   _WNF, _WNF,      # 2 drive polls: waist misses (head keeps tracking)
                   _wok(1000, 0),   # FINAL post-stop waist look: ACQUIRED
                   _wok(1000, 120)],  # fresh re-detect for align
        head_seq=[_hfound(bearing=0.2),   # front scan: found → coarse approach
                  _hfound(), _hfound()],   # both drive polls: head still tracking (never lost)
        side="object", drive_polls=2)      # worker self-stops after 2 polls
    out = api._nav._face_object("box")
    assert out["ok"] and out["status"] == "acquired"
    assert env.low_level.drive_started == 1 and env.low_level.drive_stopped == 1
    assert env.low_level.hold_calls == 0                     # head never lost → never paused
    assert env.low_level.nav_calls == [(0.0, 0.0, 0.8)]      # ONLY the coarse turn; no centroid-align turn


def test_grounded_grasp_head_searches_target_verified_on_reference():
    # _face_object(target, on=reference), S2: the head now has a point cloud, so its COARSE search
    # targets the REAL target (white box) and 3-D-verifies it on the reference surface (threaded as
    # on="brown table") — no longer a reference-surface proxy. The waist keeps testing the full
    # grounded relation → acquisition = target. BOTH the head pan-scan AND every coarse-approach
    # drive poll re-detect target + reference and depth-verify target-on-reference before advancing
    # (on="brown table" threaded all the way into the creep) — never a bare bearing-only head check.
    env = _Env(drive_polls=5)
    api = CruzrApi(env)
    wit = iter([_WNF,             # step1 grounded waist miss (ref color-rejected up close)
                _WNF,             # drive poll1 waist miss → head-steered creep
                _wok(1000, 0),    # drive poll2 waist: target-on-surface ACQUIRED
                _wok(1000, 0)])   # fresh for align
    api.locate_for_grasp = lambda object_name="white box", reference=None, relation="on": next(wit)  # type: ignore[method-assign]
    head_calls = []

    def _head(object_name="x", on=None, camera=None):
        head_calls.append((object_name, on))
        return _hfound(bearing=0.2)

    api._nav._look_for = _head          # type: ignore[method-assign]
    api.set_head = lambda yaw_rad, pitch_rad: {"ok": True}   # type: ignore[method-assign]
    out = api._nav._face_object("white box", reference="brown table")
    assert out["ok"]
    names = [n for (n, _o) in head_calls]
    ons = [o for (_n, o) in head_calls]
    assert names                                             # the head was actually consulted
    assert all(n == "white box" for n in names)              # S2: head searches the REAL target
    assert "brown table" not in names                        # never chased the bare surface
    assert env.low_level.drive_started == 1                  # the coarse-approach creep ran
    assert len(ons) >= 1                                     # head consulted in the pan-scan (creep is waist-only)
    assert all(o == "brown table" for o in ons)              # ...grounded-verified on the reference (on set)


def test_grounded_grasp_hands_off_only_on_grounded_confirm():
    # Grounded handoff (grounded grasp): the coarse-creep WAIST trigger is the GROUNDED "on surface"
    # detection — never a plain one, which at range can land on the bare reference box (the detector
    # can't separate the target noun from the reference noun without depth). So the base head-steers
    # closer until the target is confirmed ON its reference. Here the grounded detection misses the
    # first polls and resolves mid-creep → handoff; the grounded result carries the reference.
    env = _Env(drive_polls=5)
    api = CruzrApi(env)

    def _detect(object_name="white box", reference=None, relation="on"):
        assert reference is not None, "coarse handoff must use the GROUNDED detection, never plain"
        _detect.n += 1                         # grounded acquires on the 3rd call (2nd creep poll)
        return {**_wok(1000, 0), "reference": reference} if _detect.n >= 3 else _WNF
    _detect.n = 0

    api.locate_for_grasp = _detect                                         # type: ignore[method-assign]
    api._nav._look_for = lambda object_name="x", on=None, camera=None: _hfound(bearing=0.2)  # type: ignore[method-assign]
    api.set_head = lambda yaw_rad, pitch_rad: {"ok": True}                 # type: ignore[method-assign]
    out = api._nav._face_object("white box", reference="brown table")
    assert out["ok"]
    assert env.low_level.drive_started == 1 and env.low_level.drive_stopped == 1
    # the grounded handoff detection is cached (carries its reference) for approach_for_grasp.
    assert api._last_detection is not None
    assert api._last_detection.get("reference") == "brown table"


def test_grounded_place_head_searches_reference_object_verified_on_surface():
    # _face_surface(surface, has=reference), S2: the head coarse-searches the reference OBJECT (has)
    # — which sits ON the surface, so its bearing ≈ the surface's — and 3-D-verifies it on the
    # surface (threaded as on="white table"). Symmetric to the grasp side; the waist does the final
    # "surface holds reference" grounding. Never chases the colour-qualified surface noun directly.
    # BOTH the head pan-scan AND every drive-loop poll carry on="white table" (grounded to the creep).
    env = _Env(drive_polls=5)
    api = CruzrApi(env)
    wit = iter([_WNF, _WNF, _wok(1000, 0), _wok(1000, 0)])
    api.locate_for_place = lambda object_name="white table", reference=None, relation="on": next(wit)  # type: ignore[method-assign]
    head_calls = []

    def _head(object_name="x", on=None, camera=None):
        head_calls.append((object_name, on))
        return _hfound(bearing=0.2)

    api._nav._look_for = _head          # type: ignore[method-assign]
    api.set_head = lambda yaw_rad, pitch_rad: {"ok": True}   # type: ignore[method-assign]
    out = api._nav._face_surface("white table", reference="water cup", relation="under")
    assert out["ok"]
    names = [n for (n, _o) in head_calls]
    ons = [o for (_n, o) in head_calls]
    assert names                                             # the head was actually consulted
    assert all(n == "water cup" for n in names)              # head searches the reference OBJECT
    assert "white table" not in names                        # never chased the bare surface noun
    assert env.low_level.drive_started == 1                  # the coarse-approach creep ran
    assert len(ons) >= 1                                     # head consulted in the pan-scan (creep is waist-only)
    assert all(o == "white table" for o in ons)              # ...grounded-verified on the surface (on set)


def test_grounded_place_hands_off_only_on_grounded_confirm():
    # Symmetric to the grasp grounded-handoff: the coarse-creep WAIST trigger is the GROUNDED "surface
    # has reference" sense — never a plain surface sense, which at range can face a bare wrong surface.
    # So the base head-steers closer until the surface is confirmed to hold the reference. Here the
    # grounded sense misses the first polls and resolves mid-creep → handoff; the grounded surface
    # carries the reference and is routed to the PLACE cache (_last_surface), not the grasp cache.
    env = _Env(drive_polls=5)
    api = CruzrApi(env)

    def _sense(object_name="white table", reference=None, relation="on"):
        assert reference is not None, "coarse handoff must use the GROUNDED surface sense, never plain"
        _sense.n += 1                          # grounded acquires on the 3rd call (2nd creep poll)
        return {**_wok(1000, 0), "reference": reference} if _sense.n >= 3 else _WNF
    _sense.n = 0

    api.locate_for_place = _sense                                          # type: ignore[method-assign]
    api._nav._look_for = lambda object_name="x", on=None, camera=None: _hfound(bearing=0.2)  # type: ignore[method-assign]
    api.set_head = lambda yaw_rad, pitch_rad: {"ok": True}                # type: ignore[method-assign]
    out = api._nav._face_surface("white table", reference="water cup", relation="under")
    assert out["ok"]
    assert env.low_level.drive_started == 1 and env.low_level.drive_stopped == 1
    # the grounded handoff surface is cached (carries its reference) for approach_for_place, routed to
    # the PLACE cache (_last_surface), leaving the grasp cache untouched.
    assert api._last_surface is not None
    assert api._last_surface.get("reference") == "water cup"
    assert api._last_detection is None
