# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Fine-grained 'X on Y' grounding: ``locate_for_grasp(object_name, reference=..., relation="on")`` keeps only
the target that sits ON the reference surface (footprint + above its top), disambiguating
same-class targets. Reference sensing + candidate geometry + colour check are mocked."""

from __future__ import annotations

import numpy as np
import pytest

from jiuwensymbiosis.adapters.cruzr.api import CruzrApi
from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
from jiuwensymbiosis.adapters.cruzr.env import CruzrEnv
from jiuwensymbiosis.perception import object_geometry as og
from jiuwensymbiosis.perception import vision

from jiuwensymbiosis.perception.frame import CameraFrame

_EMPTY_FRAME = CameraFrame(rgb=None)   # these tests stub the detector; no pixels are read


def _api():
    env = CruzrEnv.__new__(CruzrEnv)
    env.cfg = CruzrConfig()
    api = CruzrApi(env)
    api._ensure_detector = lambda: None  # type: ignore[method-assign]
    api._seg_fn = lambda rgb, text_prompt=None: [{"mask": np.ones((4, 4), bool), "score": 0.8},
                                                 {"mask": np.ones((4, 4), bool), "score": 0.7}]
    return api


# reference: brown table footprint x∈[400,900], y-centre 0, width 600, top z 600mm
_REF = {"ok": True, "surface_z_mm": 600.0, "center_mm": [650.0, 0.0, 600.0],
        "front_x_mm": 400.0, "back_x_mm": 900.0, "width_mm": 600.0}


def _geo(center):
    return og.ObjectGeometry3D(ok=True, reason="", center_mm=center, width_mm=200, height_mm=100,
                               front_x_mm=center[0] - 50, top_z_mm=center[2] + 50, n_points=500,
                               back_x_mm=center[0] + 50)


def _patch_geoms(monkeypatch, geoms):
    it = iter(geoms)
    monkeypatch.setattr(og, "object_geometry_from_mask", lambda *a, **k: next(it))
    monkeypatch.setattr(vision, "region_color_matches", lambda *a, **k: True)
    monkeypatch.setattr(vision, "extract_color_word", lambda n: "white")


def test_picks_the_target_on_the_reference(monkeypatch):
    api = _api()
    api._scene._sense_surface_plain = lambda name, *a, **k: _REF  # type: ignore[method-assign]
    # cand A on the table (x,y inside footprint, above top); cand B off the table (x too far)
    _patch_geoms(monkeypatch, [_geo((650.0, 50.0, 650.0)), _geo((1500.0, 800.0, 650.0))])
    r = api._scene._detect_related("white box", "brown table", "on", _EMPTY_FRAME)
    assert r["ok"] is True
    assert r["center_mm"] == [650.0, 50.0, 650.0]  # the on-surface one
    assert api._last_detection is r  # cached for dual_arm_grasp


def test_no_candidate_on_reference_fails(monkeypatch):
    api = _api()
    api._scene._sense_surface_plain = lambda name, *a, **k: _REF  # type: ignore[method-assign]
    # both candidates are off the table (x beyond back_x + margin)
    _patch_geoms(monkeypatch, [_geo((1500.0, 0.0, 650.0)), _geo((1600.0, 0.0, 650.0))])
    r = api._scene._detect_related("white box", "brown table", "on", _EMPTY_FRAME)
    assert r["ok"] is False and r["reason"] == "no_target_matching_reference"


def test_reference_not_found_fails():
    api = _api()
    api._scene._sense_surface_plain = lambda name, *a, **k: {"ok": False, "reason": "surface_not_found"}  # type: ignore[method-assign]
    r = api._scene._detect_related("white box", "brown table", "on", _EMPTY_FRAME)
    assert r["ok"] is False and r["reason"].startswith("reference_not_found")


def test_detect_on_surface_grabs_frame_once(monkeypatch):
    """Grounded locate_for_grasp(reference=...) reuses the single grabbed frame for the reference surface
    (via _sense_surface_plain) instead of re-grabbing through locate_for_place — one grab, not two."""
    api = _api()
    grabs = {"n": 0}

    class _LL:
        def grab_frames(self, camera="waist"):
            grabs["n"] += 1
            return (np.ones((4, 4, 3), np.uint8), np.ones((4, 4)), np.eye(3), np.eye(4))

    api._ll = lambda: _LL()  # type: ignore[method-assign]
    api._scene._sense_surface_plain = lambda name, *a, **k: _REF  # type: ignore[method-assign]
    # _seg_fn yields two masks, so provide one geometry per mask (both on-surface).
    _patch_geoms(monkeypatch, [_geo((650.0, 50.0, 650.0)), _geo((650.0, 50.0, 650.0))])
    r = api.locate_for_grasp("white box", reference="brown table")
    assert r["ok"] is True
    assert grabs["n"] == 1  # exactly one grab (was 2 before the dedup)


# ---- place-side mirror: locate_for_place(relation="under") picks the surface holding the reference ----

def _surf(center, front_x, back_x, width):
    return og.ObjectGeometry3D(ok=True, reason="", center_mm=center, width_mm=width, height_mm=50,
                               front_x_mm=front_x, top_z_mm=center[2] + 25, n_points=800, back_x_mm=back_x)


def test_picks_the_surface_holding_the_reference(monkeypatch):
    api = _api()
    cup = _geo((600.0, 0.0, 650.0))  # water cup at x=600, y=0, above the surfaces
    table_a = _surf((650.0, 0.0, 600.0), 400.0, 900.0, 600.0)   # footprint contains the cup
    table_b = _surf((650.0, 900.0, 600.0), 400.0, 900.0, 600.0)  # y-centre 900 → cup not on it
    # _candidate_geometries returns (geometry, detection) pairs; _sense_surface_related uses the geometry.
    monkeypatch.setattr(api._scene, "_candidate_geometries",
                        lambda name, *a, **k: [(cup, {})] if name == "water cup" else [(table_a, {}), (table_b, {})])
    r = api._scene._sense_surface_related("table", "water cup", "under", _EMPTY_FRAME)
    assert r["ok"] is True
    assert r["center_mm"][1] == 0.0  # the table that has the cup
    assert api._last_surface is r


def test_no_surface_has_the_reference_fails(monkeypatch):
    api = _api()
    cup = _geo((600.0, 0.0, 650.0))
    far_table = _surf((650.0, 2000.0, 600.0), 400.0, 900.0, 600.0)  # cup nowhere near it
    monkeypatch.setattr(api._scene, "_candidate_geometries",
                        lambda name, *a, **k: [(cup, {})] if name == "water cup" else [(far_table, {})])
    r = api._scene._sense_surface_related("table", "water cup", "under", _EMPTY_FRAME)
    assert r["ok"] is False and r["reason"] == "no_surface_matching_reference"


def test_reference_object_not_detected_fails(monkeypatch):
    api = _api()
    monkeypatch.setattr(api._scene, "_candidate_geometries",
                        lambda name, *a, **k: [] if name == "water cup" else [(_surf((650.0, 0.0, 600.0), 400.0, 900.0, 600.0), {})])
    r = api._scene._sense_surface_related("table", "water cup", "under", _EMPTY_FRAME)
    assert r["ok"] is False and r["reason"].startswith("reference_not_found")


# ---- the relation is a parameter, not a hard-coded "on" ----------------------------------
def test_beside_picks_the_neighbour_not_the_stack(monkeypatch):
    """"the box beside the hat": the reference is an ORDINARY object, so it is measured as one —
    a surface record states no thickness and `beside` compares z-extents."""
    api = _api()
    hat = _geo((650.0, 500.0, 650.0))
    api._scene._candidate_geometries = lambda name, *a, **k: (  # type: ignore[method-assign]
        [(hat, {})] if name == "hat"
        else [(_geo((650.0, 350.0, 650.0)), {}), (_geo((650.0, -900.0, 650.0)), {})]
    )
    r = api._scene._detect_related("box", "hat", "beside", _EMPTY_FRAME)
    assert r["ok"] is True
    assert r["center_mm"] == [650.0, 350.0, 650.0]      # the neighbour, not the far one
    assert (r["reference"], r["relation"]) == ("hat", "beside")


def test_a_relation_outside_the_closed_set_is_refused_before_sensing():
    """The schema advertises the enum but nothing enforces it at dispatch, so the action checks —
    an unknown relation must fail by name, never quietly behave like `on`."""
    api = _api()
    api._grab_calibrated_frame = lambda camera=None: pytest.fail("must not sense on a bad relation")
    r = api.locate_for_grasp("box", reference="hat", relation="left_of")
    assert r["ok"] is False and r["reason"] == "unknown_relation:left_of"
    assert r["known_relations"] == ["on", "under", "in", "beside", "near"]


def test_the_place_side_takes_the_relation_too():
    api = _api()
    cup = _geo((650.0, 0.0, 650.0))
    tables = [_surf((650.0, 0.0, 600.0), 400.0, 900.0, 600.0),
              _surf((650.0, 900.0, 600.0), 400.0, 900.0, 600.0)]
    api._scene._candidate_geometries = lambda name, *a, **k: (  # type: ignore[method-assign]
        [(cup, {})] if name == "water cup" else [(t, {}) for t in tables]
    )
    r = api._scene._sense_surface_related("table", "water cup", "under", _EMPTY_FRAME)
    assert r["ok"] is True and r["relation"] == "under"
    assert r["center_mm"][1] == 0.0        # the table the cup is actually on
