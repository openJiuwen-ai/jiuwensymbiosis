# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Unit tests for CruzrApi.locate_for_grasp (stubs grab_frames + seg_fn)."""

import numpy as np

from jiuwensymbiosis.adapters.cruzr.api import CruzrApi


class _LL:
    def grab_frames(self, camera="waist"):
        h, w = 360, 640
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        depth = np.zeros((h, w), dtype=np.float32)
        depth[160:200, 300:340] = 0.5
        K = np.array([[345.0, 0, 320.0], [0, 345.0, 180.0], [0, 0, 1]])
        return rgb, depth, K, np.eye(4)


class _Env:
    low_level = _LL()

    class cfg:
        default_arm = "left"


def test_locate_for_grasp_ok(monkeypatch):
    api = CruzrApi(_Env())
    mask = np.zeros((360, 640), dtype=bool)
    mask[160:200, 300:340] = True
    monkeypatch.setattr(api, "_ensure_detector", lambda: None)
    api._seg_fn = lambda rgb, text_prompt="box": [
        {"score": 0.9, "label": "box", "box": [300, 160, 340, 200], "mask": mask}
    ]
    out = api.locate_for_grasp("box")
    assert out["ok"]
    assert out["width_mm"] > 0   # base-Y extent of the 40-pixel patch
    # height_mm (base-Z extent) ≈ 0 for a fronto-parallel patch at constant depth
    assert out["center_mm"][2] > 0
    assert abs(out["center_mm"][2] - 500.0) < 2.0


def test_locate_for_grasp_no_camera(monkeypatch):
    class _NoCamLL:
        def grab_frames(self, camera="waist"):
            return None

    class _NoCamEnv:
        low_level = _NoCamLL()

        class cfg:
            default_arm = "left"

    api = CruzrApi(_NoCamEnv())
    monkeypatch.setattr(api, "_ensure_detector", lambda: None)
    out = api.locate_for_grasp("box")
    assert out["ok"] is False
    assert out["reason"] == "no_camera"


def test_locate_for_grasp_no_detection(monkeypatch):
    api = CruzrApi(_Env())
    monkeypatch.setattr(api, "_ensure_detector", lambda: None)
    api._seg_fn = lambda rgb, text_prompt="box": []
    out = api.locate_for_grasp("box")
    assert out["ok"] is False
    assert out["reason"] == "no_detection"


def test_locate_for_grasp_no_depth(monkeypatch):
    class _NoDepthLL:
        def grab_frames(self, camera="waist"):
            h, w = 360, 640
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            K = np.array([[345.0, 0, 320.0], [0, 345.0, 180.0], [0, 0, 1]])
            return rgb, None, K, np.eye(4)

    class _NoDepthEnv:
        low_level = _NoDepthLL()

        class cfg:
            default_arm = "left"

    api = CruzrApi(_NoDepthEnv())
    monkeypatch.setattr(api, "_ensure_detector", lambda: None)
    out = api.locate_for_grasp("box")
    assert out["ok"] is False
    assert out["reason"] == "no_depth"
    assert out["object"] == "box"


def test_locate_for_grasp_no_intrinsics(monkeypatch):
    api = CruzrApi(_Env())
    monkeypatch.setattr(api, "_ensure_detector", lambda: None)
    monkeypatch.setattr(api, "_calib_intrinsics", lambda: None)
    api._seg_fn = lambda rgb, text_prompt="box": []
    # grab_frames returns (rgb, depth, None, tf) to trigger K=None path
    class _NoIntrLL:
        def grab_frames(self, camera="waist"):
            h, w = 360, 640
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            depth = np.zeros((h, w), dtype=np.float32)
            depth[160:200, 300:340] = 0.5
            return rgb, depth, None, np.eye(4)

    class _NoIntrEnv:
        low_level = _NoIntrLL()

        class cfg:
            default_arm = "left"

    api = CruzrApi(_NoIntrEnv())
    monkeypatch.setattr(api, "_ensure_detector", lambda: None)
    monkeypatch.setattr(api, "_calib_intrinsics", lambda: None)
    out = api.locate_for_grasp("box")
    assert out["ok"] is False
    assert out["reason"] == "no_intrinsics"
    assert out["object"] == "box"


def test_locate_for_grasp_requires_live_tf(monkeypatch):
    """Pose-dependent extrinsics: no live TF -> fail loudly, do NOT use static calib.

    Falling back to the static calib after the body moves gives wrong box
    coordinates, so locate_for_grasp must reject a frame with no live TF even when
    a static calib is configured.
    """
    class _NoTfLL:
        def grab_frames(self, camera="waist"):
            h, w = 360, 640
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            depth = np.zeros((h, w), dtype=np.float32)
            depth[160:200, 300:340] = 0.5
            K = np.array([[345.0, 0, 320.0], [0, 345.0, 180.0], [0, 0, 1]])
            return rgb, depth, K, None  # no live TF

    class _NoTfEnv:
        low_level = _NoTfLL()

        class cfg:
            default_arm = "left"

    api = CruzrApi(_NoTfEnv())
    monkeypatch.setattr(api, "_ensure_detector", lambda: None)
    # even with a static calib available, missing live TF must fail loudly
    monkeypatch.setattr(api, "_calib_extrinsics", lambda: np.eye(4))
    out = api.locate_for_grasp("box")
    assert out["ok"] is False
    assert out["reason"] == "no_live_tf"
    assert out["object"] == "box"


def test_locate_for_grasp_tool_tagged_vision():
    from jiuwensymbiosis.tools.builder import list_tool_meta
    api = CruzrApi(_Env())
    meta = {m["name"]: m for m in list_tool_meta(api)}
    assert "locate_for_grasp" in meta
    assert "vision" in meta["locate_for_grasp"].get("tags", [])
