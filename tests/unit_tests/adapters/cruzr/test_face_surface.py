# coding: utf-8
"""face_surface: sensor-guided base reorientation — turn by the PERCEIVED surface bearing
(no-op when facing). When not in the current waist view, the head PANS to scan the facing (look
left + right once); on a miss the base rotates 180° ONCE and the head re-scans behind — no
continuous base spin. These tests drive the pure-waist paths (head detector always not-found);
the head+waist coordinated search is covered in test_face_head_assist.py."""

import math

from jiuwensymbiosis.adapters.cruzr.api import CruzrApi
from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig


class _LL:
    """Fake low-level: real rotate_base path (navigate_relative, recorded raw). ``start_base_spin``
    is present but must NEVER be called — the head pan-scan replaced the old continuous base spin."""

    def __init__(self):
        self.nav_calls = []
        self.spin_started = 0

    def navigate_relative(self, dx, dy, dyaw):
        self.nav_calls.append((dx, dy, dyaw))
        return {"ok": True, "yaw_reached": dyaw}

    def start_base_spin(self, direction=1.0):
        self.spin_started += 1        # asserted 0: the pan-scan must not spin the base
        return object()


class _Env:
    def __init__(self):
        self.low_level = _LL()
        self.cfg = CruzrConfig()   # real defaults: tol=0.10


def _api_with_senses(senses):
    """CruzrApi whose locate_for_place yields `senses` in order (rotate_base is the fake LL).

    The head detector (``search_target``) always reports not-found here so these tests exercise the
    pure waist path; head+waist coordination is covered in test_face_head_assist.py."""
    env = _Env()
    api = CruzrApi(env)
    it = iter(senses)
    api.locate_for_place = lambda object_name="table": next(it)   # type: ignore[method-assign]
    api._nav.look_for = lambda object_name="table", on=None, camera=None: {"ok": True, "found": False,  # type: ignore[method-assign]
                                                              "reason": "no_detection", "image_h": 100, "image_w": 200}
    api.set_head = lambda yaw_rad, pitch_rad: {"ok": True}       # type: ignore[method-assign]
    return api, env


def _ok(cx, cy):
    return {"ok": True, "center_mm": [cx, cy, 0.0], "surface_z_mm": 700.0}


_NF = {"ok": False, "reason": "no_detection"}


def test_acquire_in_view_does_not_turn():
    # surface centred ahead, already in view → acquire + cache, no rotation, no head scan, no spin.
    api, env = _api_with_senses([_ok(1000.0, 50.0)])
    out = api._nav._face_surface("table")
    assert out["ok"] and out["status"] == "acquired"
    assert out["turned_rad"] == 0.0
    assert env.low_level.nav_calls == []          # never rotated
    assert env.low_level.spin_started == 0         # in view → never spun


def test_offcentre_surface_in_view_does_not_align_turn():
    # Centroid facing removed: an off-centre surface already in view is acquired + cached WITHOUT a
    # rotate — approach_for_place owns all base alignment. The bearing is still reported for info.
    api, env = _api_with_senses([_ok(1000.0, 500.0)])
    out = api._nav._face_surface("table")
    assert out["ok"] and out["status"] == "acquired"
    assert math.isclose(out["bearing_rad"], math.atan2(500.0, 1000.0), rel_tol=1e-6)
    assert env.low_level.spin_started == 0
    assert env.low_level.nav_calls == []          # off-centre, but no centroid-align turn


def test_not_found_after_panscan_and_180_fails_safe():
    # never in the waist view and the head never sees it (front pan-scan miss → 180° → back
    # pan-scan miss) → fail safe; caller must not blind-place. The ONLY base motion is the single
    # 180° fallback (no continuous spin).
    api, env = _api_with_senses([_NF])
    out = api._nav._face_surface("table")
    assert out["ok"] is False and out["reason"] == "surface_not_found"
    assert out.get("note") == "panscan_exhausted"
    assert env.low_level.spin_started == 0
    assert env.low_level.nav_calls == [(0.0, 0.0, math.pi)]   # only the 180° fallback
    assert math.isclose(out["turned_rad"], math.pi, rel_tol=1e-6)
