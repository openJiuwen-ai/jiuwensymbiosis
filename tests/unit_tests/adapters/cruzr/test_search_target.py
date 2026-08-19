# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Unit tests for CruzrApi.search_target (head bearing search; stubs grab + seg_fn)."""

import numpy as np

from jiuwensymbiosis.adapters.cruzr.api import CruzrApi


class _Cfg:
    default_arm = "left"
    head_hfov_rad = 1.2


class _HeadLL:
    def __init__(self, rgb):
        self._rgb = rgb

    def grab_frames(self, camera="waist"):
        # search_target looks through EVERY camera now; both of cruzr's serve the same
        # rendered scene here, so which one hits is not what these tests are about.
        assert camera in ("head", "waist")
        return self._rgb, None, None, None


class _Env:
    cameras = ("head",)

    def __init__(self, rgb):
        self.low_level = _HeadLL(rgb)
        self.cfg = _Cfg()


def _api(rgb):
    return CruzrApi(_Env(rgb))


def test_search_target_found_centered(monkeypatch):
    rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    api = _api(rgb)
    monkeypatch.setattr(api, "_ensure_detector", lambda: None)
    # box centered horizontally: x in [80,120] -> u=100 == w/2
    # mask is a dummy non-empty array: search_target reads only box/score, but
    # the shared _run_detect_pick_best logs mask.shape/.sum() for every
    # candidate, so a None mask would crash that (unrelated) logging line.
    api._seg_fn = lambda rgb, text_prompt="box": [
        {"score": 0.9, "label": "box", "box": [80, 40, 120, 60], "mask": np.ones((20, 40), dtype=bool)}
    ]
    out = api.search_target("box")
    assert out["ok"] and out["found"]
    assert out["camera"] == "head"   # the result names the camera that answered
    assert out["image_w"] == 200 and out["image_h"] == 100
    assert abs(out["u_center"] - 100.0) < 1e-6
    assert abs(out["u_error_frac"]) < 1e-6
    assert abs(out["bearing_rad"]) < 1e-6


def test_search_target_found_left_turns_left(monkeypatch):
    rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    api = _api(rgb)
    monkeypatch.setattr(api, "_ensure_detector", lambda: None)
    # box on the LEFT: x in [20,60] -> u=40, u_error_frac=(40-100)/200=-0.3
    api._seg_fn = lambda rgb, text_prompt="box": [
        {"score": 0.8, "label": "box", "box": [20, 40, 60, 60], "mask": np.ones((20, 40), dtype=bool)}
    ]
    out = api.search_target("box")
    assert out["found"]
    assert abs(out["u_error_frac"] + 0.3) < 1e-6
    # target left -> positive bearing (turn left): -(-0.3)*1.2 = 0.36
    assert abs(out["bearing_rad"] - 0.36) < 1e-6


def test_search_target_not_found(monkeypatch):
    rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    api = _api(rgb)
    monkeypatch.setattr(api, "_ensure_detector", lambda: None)
    api._seg_fn = lambda rgb, text_prompt="box": []
    out = api.search_target("box")
    assert out["ok"] is True
    assert out["found"] is False
    assert out["reason"] == "no_detection"


def test_search_target_no_camera(monkeypatch):
    class _NoCamEnv:
        class low_level:
            @staticmethod
            def grab_frames(camera="waist"):
                return None
        cfg = _Cfg()

    api = CruzrApi(_NoCamEnv())
    monkeypatch.setattr(api, "_ensure_detector", lambda: None)
    out = api.search_target("box")
    assert out["ok"] is False
    assert out["found"] is False
    assert out["reason"] == "no_camera"


def test_search_target_grayscale_is_stacked(monkeypatch):
    # A 2-D (mono) head frame must be promoted to 3-channel before detection.
    rgb = np.zeros((100, 200), dtype=np.uint8)
    api = _api(rgb)
    monkeypatch.setattr(api, "_ensure_detector", lambda: None)
    seen = {}

    def _seg(img, text_prompt="box"):
        seen["ndim"] = img.ndim
        return [{"score": 0.7, "label": "box", "box": [90, 40, 110, 60], "mask": np.ones((20, 20), dtype=bool)}]

    api._seg_fn = _seg
    out = api.search_target("box")
    assert out["found"]
    assert seen["ndim"] == 3


def test_search_target_tool_tagged_vision():
    from jiuwensymbiosis.tools.builder import list_tool_meta
    api = _api(np.zeros((100, 200, 3), dtype=np.uint8))
    meta = {m["name"]: m for m in list_tool_meta(api)}
    assert "search_target" in meta
    assert "vision" in meta["search_target"].get("tags", [])
