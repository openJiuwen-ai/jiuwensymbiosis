# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Cache hygiene: failed detect/grasp must not leave stale caches that make a
downstream step act on the wrong box (H1/H2 in the fast-mode design)."""

from jiuwensymbiosis.adapters.cruzr.api import CruzrApi


class _NoCamLL:
    def grab_frames(self, camera="waist"):
        return None


class _NoCamEnv:
    low_level = _NoCamLL()

    class cfg:
        default_arm = "left"


def test_locate_for_grasp_clears_stale_detection_on_failure(monkeypatch):
    api = CruzrApi(_NoCamEnv())
    monkeypatch.setattr(api, "_ensure_detector", lambda: None)
    api._last_detection = {"ok": True, "center_mm": [1.0, 2.0, 3.0]}  # stale
    out = api.locate_for_grasp("box")
    assert out["ok"] is False
    assert out["reason"] == "no_camera"
    assert api._last_detection is None


class _UrdfEnv:
    low_level = None

    class cfg:
        urdf_path = "unused-mocked"
        left_arm_leaf = "L_leaf"
        right_arm_leaf = "R_leaf"


def test_dual_arm_grasp_clears_stale_grasp_when_no_detection(monkeypatch):
    # parse_chain is called before the no-detection guard; stub it so we reach
    # the guard without a real URDF.
    monkeypatch.setattr(
        "jiuwensymbiosis.kinematics.urdf_chain.parse_chain",
        lambda *a, **k: object(),
    )
    api = CruzrApi(_UrdfEnv())
    api._last_grasped_box = {"center_mm": [1.0, 2.0, 3.0]}  # stale, from a prior grasp
    api._last_detection = None  # no fresh detection this run
    out = api.dual_arm_grasp()
    assert out["ok"] is False
    assert out["reason"] == "no_detection"
    assert api._last_grasped_box is None
