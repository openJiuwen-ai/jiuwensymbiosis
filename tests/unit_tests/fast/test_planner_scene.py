# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Phase C: fast planner is scene-aware (pre-plan perception injected into prompts)."""

import inspect

from jiuwensymbiosis.agent.fast import planner


def test_compose_actions_accepts_scene():
    assert "scene" in inspect.signature(planner.compose_actions).parameters


def test_compile_sequence_accepts_scene():
    assert "scene" in inspect.signature(planner.compile_sequence).parameters


def test_format_scene_renders():
    scene = {
        "count": 2,
        "objects": [
            {"object": "box", "distance_mm": 520.0, "center_mm": [500, 10, 300]},
            {"object": "box", "distance_mm": 800.0, "center_mm": [790, 0, 300]},
        ],
    }
    out = planner._format_scene(scene)
    assert "box" in out
    assert "2" in out  # count surfaced
    assert planner._format_scene(None) == ""
    assert planner._format_scene({}) == ""
    assert planner._format_scene({"count": 0, "objects": []}) == ""


def test_format_scene_renders_reachable_flag():
    base = {"object": "box", "distance_mm": 500.0, "center_mm": [500, 0, 300]}
    assert "可达=是" in planner._format_scene({"count": 1, "objects": [dict(base, reachable=True)]})
    assert "可达=否" in planner._format_scene({"count": 1, "objects": [dict(base, reachable=False)]})
    # No reachable field (e.g. piper) → no 可达 token, prompt unchanged
    assert "可达" not in planner._format_scene({"count": 1, "objects": [base]})


def test_format_scene_renders_reach_prior_when_no_target():
    # No detections but a body reach envelope → the URDF prior still reaches the planner.
    s = planner._format_scene({"count": 0, "objects": [], "reach_prior": {
        "reachable": True, "forward_m": [0.3, 0.8], "lateral_m": [-0.4, 0.4], "height_m": [0.5, 1.1]}})
    assert "本体可达域" in s and "前向" in s
    s2 = planner._format_scene({"count": 0, "objects": [], "reach_prior": {"reachable": False}})
    assert "本体可达域" in s2 and "无处可达" in s2
    # No reach_prior and no objects → still empty (piper unchanged)
    assert planner._format_scene({"count": 0, "objects": []}) == ""
