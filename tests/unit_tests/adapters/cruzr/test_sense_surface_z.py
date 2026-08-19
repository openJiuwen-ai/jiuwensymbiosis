# coding: utf-8
"""locate_for_place: detect the tabletop and return its base-frame surface z (top_z_mm)
plus the XY footprint (center/front_x/back_x/width) that dual_arm_place lands the box within."""

import numpy as np

from jiuwensymbiosis.adapters.cruzr.api import CruzrApi


class _LL:
    def __init__(self, depth_val=0.5):
        self._depth_val = depth_val

    def grab_frames(self, camera="waist"):
        h, w = 360, 640
        depth = np.zeros((h, w), dtype=np.float32)
        depth[150:210, 290:350] = self._depth_val          # a flat patch = the tabletop
        k = np.array([[345.0, 0, 320.0], [0, 345.0, 180.0], [0, 0, 1]])
        return np.zeros((h, w, 3), np.uint8), depth, k, np.eye(4)   # tf=eye -> base==cam


class _EnvCfg:
    urdf_path = "/nonexistent.urdf"


class _Env:
    def __init__(self, ll):
        self.low_level = ll
        self.cfg = _EnvCfg()


def _api(monkeypatch, ll):
    env = _Env(ll)
    api = CruzrApi(env)
    monkeypatch.setattr(api, "_ensure_detector", lambda: None)
    mask = np.zeros((360, 640), dtype=bool)
    mask[150:210, 290:350] = True
    api._seg_fn = lambda rgb, text_prompt="table": [
        {"score": 0.9, "label": "table", "box": [290, 150, 350, 210], "mask": mask}]
    return api, env


def test_locate_for_place_returns_backprojected_top_z(monkeypatch):
    api, env = _api(monkeypatch, _LL(depth_val=0.5))
    out = api.locate_for_place()
    assert out["ok"] is True
    # tf=eye, depth 0.5 m -> all masked points at z=500 mm -> surface_z_mm ~ 500
    assert abs(out["surface_z_mm"] - 500.0) < 1.0
    assert out["n_points"] > 0
    # also returns the base-frame XY footprint dual_arm_place uses to land ON the table
    assert len(out["center_mm"]) == 3
    assert out["front_x_mm"] <= out["back_x_mm"]        # near edge <= far edge
    assert out["width_mm"] >= 0.0
    assert api._last_surface is out                     # cached for a no-arg dual_arm_place()


def test_locate_for_place_includes_footprint_and_edge_fields(monkeypatch):
    # approach_for_place squares to the table edge using yaw_rad / edge_normal — the sense output MUST
    # carry them (the grounded path dropping these silently disabled squaring, the bug this fixes).
    api, env = _api(monkeypatch, _LL(depth_val=0.5))
    out = api.locate_for_place()
    assert out["ok"] is True
    for k in ("yaw_rad", "long_mm", "short_mm", "face_normal",
              "edge_midpoint_mm", "edge_normal", "edge_quality", "edge_len_mm"):
        assert k in out, f"missing {k}"
    assert len(out["edge_normal"]) == 2 and len(out["edge_midpoint_mm"]) == 2


def test_surface_footprint_fields_parity_carries_edge_and_yaw():
    # Both sense paths (plain + grounded) spread _surface_footprint_fields, so they can never drift.
    from jiuwensymbiosis.adapters.cruzr.api import _surface_footprint_fields
    from jiuwensymbiosis.perception.object_geometry import ObjectGeometry3D

    surf = ObjectGeometry3D(
        True, "", (700.0, 50.0, 600.0), 800.0, 20.0, 400.0, 620.0, 5000, back_x_mm=900.0,
        yaw_rad=0.1, long_mm=800.0, short_mm=500.0, face_normal_x=0.0, face_normal_y=0.0,
        face_flatness=1.0, edge_mid_x_mm=400.0, edge_mid_y_mm=50.0, edge_normal_x=-1.0,
        edge_normal_y=0.0, edge_quality=0.03, edge_len_mm=760.0)
    f = _surface_footprint_fields(surf)
    assert f["yaw_rad"] == 0.1                          # the grounded-path omission that broke squaring
    assert f["edge_normal"] == [-1.0, 0.0]
    assert f["edge_midpoint_mm"] == [400.0, 50.0]
    assert f["edge_quality"] == 0.03 and f["edge_len_mm"] == 760.0


def test_locate_for_place_no_depth(monkeypatch):
    class _NoDepth(_LL):
        def grab_frames(self, camera="waist"):
            return np.zeros((360, 640, 3), np.uint8), None, np.eye(3), np.eye(4)
    api, env = _api(monkeypatch, _NoDepth())
    assert api.locate_for_place()["reason"] == "no_depth"


def test_locate_for_place_no_detection(monkeypatch):
    api, env = _api(monkeypatch, _LL())
    api._seg_fn = lambda rgb, text_prompt="table": []        # detector finds nothing
    assert api.locate_for_place()["ok"] is False
