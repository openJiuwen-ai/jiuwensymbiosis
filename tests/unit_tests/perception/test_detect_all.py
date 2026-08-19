# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Phase B: generic multi-instance detect+project (robot-agnostic, any object).

Heavy logic lives here in perception/; adapters add a few-line hook that grabs a
frame and calls this (same pattern as default_get_grasp_info_simple).
"""

import numpy as np

from jiuwensymbiosis.perception.vision import detect_all_object_geometry


def _pinhole():
    return np.array([[345.0, 0, 320.0], [0, 345.0, 180.0], [0, 0, 1]])


def _scene_two():
    h, w = 360, 640
    depth = np.zeros((h, w), dtype=np.float32)
    depth[160:200, 100:140] = 0.5
    depth[160:200, 300:340] = 0.5
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    m1 = np.zeros((h, w), dtype=bool)
    m1[160:200, 300:340] = True
    m2 = np.zeros((h, w), dtype=bool)
    m2[160:200, 100:140] = True
    seg = [
        {"score": 0.9, "label": "cup", "box": [300, 160, 340, 200], "mask": m1},
        {"score": 0.8, "label": "cup", "box": [100, 160, 140, 200], "mask": m2},
    ]
    return rgb, depth, seg


def test_detect_all_multi_instances():
    rgb, depth, seg = _scene_two()
    seg_fn = lambda image, text_prompt="": seg  # noqa: E731 - test stub, any object
    objs = detect_all_object_geometry(
        rgb, depth, _pinhole(), np.eye(4), seg_fn=seg_fn, object_name="cup"
    )
    assert len(objs) == 2
    for o in objs:
        assert o["object"] == "cup"
        assert len(o["center_mm"]) == 3
        assert o["distance_mm"] > 0
        assert "score" in o
    # nearest-first (smaller forward-x distance first) — spec §8 Q1 default
    assert objs[0]["distance_mm"] <= objs[1]["distance_mm"]


def test_detect_all_empty():
    rgb, depth, _ = _scene_two()
    seg_fn = lambda image, text_prompt="": []  # noqa: E731
    objs = detect_all_object_geometry(rgb, depth, _pinhole(), np.eye(4), seg_fn=seg_fn, object_name="cup")
    assert objs == []


def test_detect_all_score_filter():
    rgb, depth, seg = _scene_two()
    seg[1]["score"] = 0.01  # below threshold → dropped
    seg_fn = lambda image, text_prompt="": seg  # noqa: E731
    objs = detect_all_object_geometry(
        rgb, depth, _pinhole(), np.eye(4), seg_fn=seg_fn, object_name="cup", score_threshold=0.05
    )
    assert len(objs) == 1
