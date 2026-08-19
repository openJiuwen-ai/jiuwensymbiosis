# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Phase B: CruzrApi.analyze_scene — thin hook over the generic scene scan."""

import numpy as np

from jiuwensymbiosis.adapters.cruzr.api import CruzrApi


class _LL:
    def grab_frames(self, camera="waist"):
        h, w = 360, 640
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        depth = np.zeros((h, w), dtype=np.float32)
        depth[160:200, 100:140] = 0.5
        depth[160:200, 300:340] = 0.5
        K = np.array([[345.0, 0, 320.0], [0, 345.0, 180.0], [0, 0, 1]])
        return rgb, depth, K, np.eye(4)


class _Env:
    low_level = _LL()

    class cfg:
        default_arm = "left"


def _two(rgb, text_prompt="box"):
    m1 = np.zeros((360, 640), dtype=bool)
    m1[160:200, 300:340] = True
    m2 = np.zeros((360, 640), dtype=bool)
    m2[160:200, 100:140] = True
    return [
        {"score": 0.9, "label": text_prompt, "box": [300, 160, 340, 200], "mask": m1},
        {"score": 0.8, "label": text_prompt, "box": [100, 160, 140, 200], "mask": m2},
    ]


def test_analyze_scene_multi(monkeypatch):
    api = CruzrApi(_Env())
    monkeypatch.setattr(api, "_ensure_detector", lambda: None)
    api._seg_fn = _two
    out = api.analyze_scene("box")
    assert out["ok"] and out["count"] == 2
    assert len(out["objects"]) == 2
    assert all("distance_mm" in o and o["distance_mm"] > 0 for o in out["objects"])


def test_analyze_scene_no_camera(monkeypatch):
    class _NoCamEnv:
        class low_level:
            @staticmethod
            def grab_frames(camera="waist"):
                return None

        class cfg:
            default_arm = "left"

    api = CruzrApi(_NoCamEnv())
    monkeypatch.setattr(api, "_ensure_detector", lambda: None)
    out = api.analyze_scene("box")
    assert out["ok"] is False and out["reason"] == "no_camera"
