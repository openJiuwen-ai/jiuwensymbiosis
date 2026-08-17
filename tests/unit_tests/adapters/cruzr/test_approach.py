# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Unit tests for the Cruzr approach loop (fully mocked api; no ROS/hardware)."""

from types import SimpleNamespace

from jiuwensymbiosis.adapters.cruzr.api import run_approach


def _cfg(**over):
    base = dict(
        head_yaw_joint="head_yaw_joint", head_pitch_joint="head_pitch_joint",
        head_search_yaw_positions_rad=(0.0, 0.6, -0.6),
        head_search_pitch_rad=-0.35, head_forward_yaw_rad=0.0, head_forward_pitch_rad=0.0,
        center_tol_frac=0.08, approach_step_m=0.4, approach_max_iterations=5,
        probe_bbox_frac=0.25, grasp_forward_min_m=0.30, grasp_forward_max_m=0.50,
        lost_target_max=3,
    )
    base.update(over)
    return SimpleNamespace(**base)


class FakeApi:
    def __init__(self, search_results, detect_results=None, nav_result=None, **cfg_over):
        self._search = list(search_results)
        self._detect = list(detect_results or [])
        self._nav_result = nav_result or {"ok": True, "arrived": True}
        self.calls = []
        self.env = SimpleNamespace(cfg=_cfg(**cfg_over))

    def move_named_joint(self, joint_name, position_rad):
        self.calls.append(("move_named_joint", joint_name, round(float(position_rad), 4)))
        return {"ok": True}

    def set_head(self, yaw_rad, pitch_rad):
        self.calls.append(("set_head", round(float(yaw_rad), 4), round(float(pitch_rad), 4)))
        return {"ok": True}

    @property
    def _nav(self):
        """A body holds its approach component; the head pan-scan looks through it."""
        return self

    def _look_for(self, object_name="box", on=None, camera=None):
        """One look through one camera — the seam the head pan-scan drives."""
        self.calls.append(("search_target", object_name))
        return self._search.pop(0)

    def navigate_relative(self, dx_m, dy_m=0.0, dyaw_rad=0.0):
        self.calls.append(("navigate_relative", round(dx_m, 4), round(dy_m, 4), round(dyaw_rad, 4)))
        return self._nav_result

    def locate_for_grasp(self, object_name="box", reference=None, relation="on"):
        self.calls.append(("locate_for_grasp", object_name))
        return self._detect.pop(0) if self._detect else {"ok": False, "reason": "no_detection"}


def _found(bearing=0.0, u_err=0.0, bbox=(0, 0, 50, 50), image_h=100, image_w=200, v_center=None):
    return {"ok": True, "found": True, "bearing_rad": bearing, "u_error_frac": u_err,
            "bbox": list(bbox), "image_h": image_h, "image_w": image_w, "score": 0.9,
            "v_center": image_h / 2.0 if v_center is None else v_center}


def _missing():
    return {"ok": True, "found": False, "reason": "no_detection", "image_h": 100, "image_w": 200}


def _det_ok(center_x_m):
    return {"ok": True, "center_mm": [center_x_m * 1000.0, 0.0, 780.0],
            "width_mm": 300.0, "height_mm": 200.0, "front_x_mm": center_x_m * 1000.0 - 100,
            "back_x_mm": center_x_m * 1000.0 + 100, "top_z_mm": 880.0, "n_points": 500}


def test_acquire_by_head_then_handoff():
    # Head sweep: not found at yaw 0.0 and 0.6; found at -0.6 with in-image bearing 0.1.
    # Then approach iter 1: centered, big bbox -> waist detect in band -> handoff.
    api = FakeApi(
        search_results=[_missing(), _missing(), _found(bearing=0.1),
                        _found(bbox=(0, 0, 60, 100))],   # bbox height 100/100 -> probes
        detect_results=[_det_ok(0.4)],
    )
    out = run_approach(api, "box")
    assert out["ok"] is True and out["handoff"] is True
    assert out["detection"]["center_mm"][0] == 400.0
    # base turned by total bearing theta+beta = -0.6 + 0.1 = -0.5 AFTER head reset
    navs = [c for c in api.calls if c[0] == "navigate_relative"]
    assert navs[0] == ("navigate_relative", 0.0, 0.0, -0.5)
    # head was reset to forward (yaw 0.0) before turning the base
    assert ("set_head", 0.0, 0.0) in api.calls


def test_center_then_step_then_handoff():
    # Acquire ahead (found at yaw 0.0, bearing 0). Then: off-center small bbox -> turn;
    # centered small bbox -> step; centered big bbox -> detect in band -> handoff.
    api = FakeApi(
        search_results=[
            _found(bearing=0.0),                              # acquisition (yaw 0.0)
            _found(u_err=0.3, bearing=-0.36, bbox=(0, 0, 10, 10)),  # off-center, small
            _found(u_err=0.0, bbox=(0, 0, 10, 10)),           # centered, small
            _found(u_err=0.0, bbox=(0, 0, 60, 100)),          # centered, big -> probe
        ],
        detect_results=[_det_ok(0.35)],
    )
    out = run_approach(api, "box")
    assert out["ok"] and out["handoff"]
    navs = [c for c in api.calls if c[0] == "navigate_relative"]
    # first loop nav = turn (dyaw), second = forward step (dx)
    assert navs[0] == ("navigate_relative", 0.0, 0.0, -0.36)
    assert navs[1] == ("navigate_relative", 0.4, 0.0, 0.0)


def test_target_not_found_resets_head():
    api = FakeApi(search_results=[_missing(), _missing(), _missing()])
    out = run_approach(api, "box")
    assert out["ok"] is False
    assert out["reason"] == "target_not_found"
    # head reset to forward on the way out
    assert ("set_head", 0.0, 0.0) in api.calls


def test_nav_failed():
    api = FakeApi(
        search_results=[_found(bearing=0.0), _found(u_err=0.3, bearing=-0.36, bbox=(0, 0, 10, 10))],
        nav_result={"ok": False, "arrived": False, "reason": "nav_unavailable"},
    )
    out = run_approach(api, "box")
    assert out["ok"] is False
    assert out["reason"] == "nav_failed"


def test_lost_target():
    api = FakeApi(
        search_results=[_found(bearing=0.0), _missing(), _missing(), _missing()],
        lost_target_max=3,
    )
    out = run_approach(api, "box")
    assert out["ok"] is False
    assert out["reason"] == "lost_target"


def test_too_close_does_not_step_forward():
    # Acquire ahead (bearing 0); a big-bbox probe reports the box too close
    # (center_x < grasp_forward_min_m=0.30) -> too_close, and NO forward creep.
    api = FakeApi(
        search_results=[_found(bearing=0.0), _found(u_err=0.0, bbox=(0, 0, 60, 100))],
        detect_results=[_det_ok(0.20)],
    )
    out = run_approach(api, "box")
    assert out["ok"] is False
    assert out["reason"] == "too_close"
    navs = [c for c in api.calls if c[0] == "navigate_relative"]
    assert all(dx == 0.0 for (_, dx, _dy, _dyaw) in navs)  # no forward step issued


def test_head_miss_waist_handoff():
    # Head lost the box (too close / below its high FOV), but the waist sees it in the
    # grasp band -> handoff, NOT counted toward lost_target.
    api = FakeApi(
        search_results=[_found(bearing=0.0), _missing()],
        detect_results=[_det_ok(0.40)],
    )
    out = run_approach(api, "box")
    assert out["ok"] and out["handoff"]
    assert out["detection"]["center_mm"][0] == 400.0


def test_head_pitch_tracks_low_box():
    # Box low in the head frame (v_center=80 of 100) -> head looks down to keep it in
    # view: v_err=(80-50)/100=0.3, +gain(-0.5)*0.3 = -0.15 from forward (0.0).
    api = FakeApi(
        search_results=[_found(bearing=0.0),
                        _found(u_err=0.0, bbox=(0, 0, 60, 100), v_center=80)],
        detect_results=[_det_ok(0.40)],
    )
    out = run_approach(api, "box")
    assert out["ok"] and out["handoff"]
    pitch_moves = [c for c in api.calls if c[0] == "set_head"]
    assert any(abs(m[2] - (-0.15)) < 1e-6 for m in pitch_moves)  # box low -> look down (-pitch)


def test_beyond_band_clamps_forward_step():
    # Big-bbox probe beyond the band (center_x=0.70 > max=0.50) -> forward step
    # clamped to center_x - max = 0.20 (not the full approach_step_m=0.4);
    # next probe (0.40) is in-band -> handoff.
    api = FakeApi(
        search_results=[_found(bearing=0.0),
                        _found(u_err=0.0, bbox=(0, 0, 60, 100)),
                        _found(u_err=0.0, bbox=(0, 0, 60, 100))],
        detect_results=[_det_ok(0.70), _det_ok(0.40)],
    )
    out = run_approach(api, "box")
    assert out["ok"] and out["handoff"]
    navs = [c for c in api.calls if c[0] == "navigate_relative"]
    fwd = [dx for (_, dx, _dy, _dyaw) in navs if dx > 0.0]
    assert fwd == [0.20]
