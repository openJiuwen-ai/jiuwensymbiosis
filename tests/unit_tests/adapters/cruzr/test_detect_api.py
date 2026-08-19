# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for CruzrApi.detect (mocked lowlevel + seg_fn)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from jiuwensymbiosis.adapters.cruzr.api import CruzrApi
from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
from jiuwensymbiosis.adapters.cruzr.env import CruzrEnv
from jiuwensymbiosis.tools.builder import list_tool_meta


class _FakeLowLevel:
    def __init__(self, frames):
        self._frames = frames

    def grab_frames(self, camera="waist"):
        return self._frames


def _env_with(frames) -> CruzrEnv:
    env = CruzrEnv(CruzrConfig())
    env._inner = _FakeLowLevel(frames)
    env._connected = True
    return env


_K = np.array([[100.0, 0.0, 4.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]])


def _detection():
    mask = np.zeros((8, 8), dtype=bool)
    mask[3:6, 3:6] = True
    return [{"mask": mask, "box": [3.0, 3.0, 6.0, 6.0], "score": 0.9, "label": "box"}]


def _frames(depth=True, k=_K, tf=None):
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    d = np.ones((8, 8), dtype=np.float32) if depth else None
    return rgb, d, k, tf


def _api(env, *, calib_path=None):
    api = CruzrApi(env, camera_calib_path=calib_path)
    api._seg_fn = lambda image, text_prompt: _detection()  # 绕过真实检测服务
    return api


def test_detect_tool_is_exposed():
    api = _api(_env_with(_frames()))
    names = {m["name"] for m in list_tool_meta(api)}
    assert "detect" in names


def test_detect_success_with_live_k_and_identity_extrinsic(tmp_path):
    calib = tmp_path / "c.json"
    calib.write_text(json.dumps({"tf_base_cam": np.eye(4).tolist()}))
    api = _api(_env_with(_frames()), calib_path=str(calib))
    out = api.detect("box")
    assert out["ok"] is True
    assert out["object"] == "box"
    assert out["camera_name"] == "waist_rgbd"
    assert out["score"] == pytest.approx(0.9)
    assert out["box_2d"] == [3.0, 3.0, 6.0, 6.0]
    assert out["depth_m"] == pytest.approx(1.0)
    # 质心 ~ (4,4) == 主点 → x=y=0, z=1000mm
    np.testing.assert_allclose(out["position"], [0.0, 0.0, 1000.0], atol=1e-6)


def test_detect_no_camera():
    out = _api(_env_with(None)).detect("box")
    assert out == {"ok": False, "reason": "no_camera", "camera_name": "waist_rgbd"}


def test_detect_no_detection():
    api = _api(_env_with(_frames()))
    api._seg_fn = lambda image, text_prompt: []
    out = api.detect("box")
    assert out["ok"] is False
    assert out["reason"] == "no_detection"
    assert out["camera_name"] == "waist_rgbd"


def test_detect_no_depth():
    out = _api(_env_with(_frames(depth=False))).detect("box")
    assert out["ok"] is False
    assert out["reason"] == "no_depth"


def test_detect_no_intrinsics_keeps_2d(tmp_path):
    calib = tmp_path / "c.json"
    calib.write_text(json.dumps({"tf_base_cam": np.eye(4).tolist()}))
    api = _api(_env_with(_frames(k=None)), calib_path=str(calib))  # 无 live K
    out = api.detect("box")
    assert out["ok"] is True
    assert out["position"] is None
    assert out["position_reason"] == "no_intrinsics"


def test_detect_no_extrinsics_keeps_2d():
    api = _api(_env_with(_frames()))  # 无 calib → 无 tf_base_cam
    out = api.detect("box")
    assert out["ok"] is True
    assert out["position"] is None
    assert out["position_reason"] == "no_extrinsics"


def test_unimplemented_vision_stubs_not_exposed():
    api = _api(_env_with(_frames()))
    names = {m["name"] for m in list_tool_meta(api)}
    assert "detect" in names
    assert "get_image" in names
    assert "pixel_to_base_xyz" in names
    assert "get_grasp_info_simple" not in names
    # analyze_scene is now implemented (multi-instance scene scan, P2 phase B) → exposed.
    assert "analyze_scene" in names
