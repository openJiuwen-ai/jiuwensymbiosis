# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Phase F: run_fast_task's pre-plan scene perception wiring (_perceive_scene)."""

from types import SimpleNamespace

from jiuwensymbiosis.agent.run import _perceive_scene


def _api(objects, caps=("vision.detection",)):
    return SimpleNamespace(
        capabilities=set(caps),
        analyze_scene=lambda t: {"ok": True, "count": len(objects), "objects": [dict(o, object=t) for o in objects]},
    )


def test_perceive_scene_aggregates_nearest_first():
    objs = [{"distance_mm": 800.0, "center_mm": [790, 0, 300]}, {"distance_mm": 500.0, "center_mm": [500, 0, 300]}]
    scene = _perceive_scene(SimpleNamespace(api=_api(objs)), ["box"])
    assert scene["count"] == 2
    assert scene["objects"][0]["distance_mm"] <= scene["objects"][1]["distance_mm"]


def test_perceive_scene_none_without_vision_capability():
    objs = [{"distance_mm": 500.0, "center_mm": [500, 0, 300]}]
    assert _perceive_scene(SimpleNamespace(api=_api(objs, caps=())), ["box"]) is None


def test_perceive_scene_none_when_no_analyze_scene():
    api = SimpleNamespace(capabilities={"vision.detection"})  # no analyze_scene attr
    assert _perceive_scene(SimpleNamespace(api=api), ["box"]) is None


def test_perceive_scene_reports_missing_when_nothing_detected():
    # A clean scan that finds nothing is a FACT ("looked, not in view"), not silence — the planner
    # can act on it by searching first. Only a scan that never ran stays silent (below).
    api = SimpleNamespace(capabilities={"vision.detection"}, analyze_scene=lambda t: {"ok": True, "objects": []})
    scene = _perceive_scene(SimpleNamespace(api=api), ["box"])
    assert scene is not None and scene["count"] == 0 and scene["missing"] == ["box"]


def test_perceive_scene_survives_detector_error():
    # A crashed detector is NOT evidence of absence: no "missing" claim may be made off it.
    def boom(t):
        raise RuntimeError("detector down")

    api = SimpleNamespace(capabilities={"vision.detection"}, analyze_scene=boom)
    assert _perceive_scene(SimpleNamespace(api=api), ["box"]) is None  # best-effort, no crash


def test_perceive_scene_scans_references_and_marks_them_separate():
    def scan(name):
        if name == "drawer":
            return {"ok": True, "objects": [{"object": "drawer", "distance_mm": 600.0, "center_mm": [600, 0, 700]}]}
        return {"ok": True, "objects": []}

    api = SimpleNamespace(capabilities={"vision.detection"}, analyze_scene=scan)
    scene = _perceive_scene(SimpleNamespace(api=api), ["apple"], ["drawer"])
    assert scene["count"] == 0 and scene["objects"] == []       # the apple is not a detection
    assert [o["object"] for o in scene["references"]] == ["drawer"]
    assert scene["missing"] == ["apple"]


def test_perceive_scene_injects_reachable_when_supported():
    objs = [{"distance_mm": 500.0, "center_mm": [500, 0, 300]}]
    api = _api(objs, caps=("vision.detection", "planning.reachability"))
    api.check_reachable = lambda o: o["distance_mm"] < 1000
    scene = _perceive_scene(SimpleNamespace(api=api), ["box"])
    assert scene["objects"][0]["reachable"] is True


def test_perceive_scene_no_reachable_field_without_capability():
    objs = [{"distance_mm": 500.0, "center_mm": [500, 0, 300]}]
    scene = _perceive_scene(SimpleNamespace(api=_api(objs)), ["box"])  # no planning.reachability
    assert "reachable" not in scene["objects"][0]  # piper/others: byte-for-byte unchanged


def test_perceive_scene_survives_reachable_error():
    objs = [{"distance_mm": 500.0, "center_mm": [500, 0, 300]}]
    api = _api(objs, caps=("vision.detection", "planning.reachability"))

    def boom(o):
        raise RuntimeError("ik down")

    api.check_reachable = boom
    scene = _perceive_scene(SimpleNamespace(api=api), ["box"])  # best-effort: no crash, field omitted
    assert scene is not None and "reachable" not in scene["objects"][0]


def test_perceive_scene_no_target_injects_reach_prior():
    # No detections, but a body with planning.reachability still hands the planner its reach envelope.
    api = SimpleNamespace(
        capabilities={"vision.detection", "planning.reachability"},
        analyze_scene=lambda t: {"ok": True, "objects": []},
        describe_reach=lambda: {"reachable": True, "forward_m": [0.3, 0.8],
                                "lateral_m": [-0.4, 0.4], "height_m": [0.5, 1.1]})
    scene = _perceive_scene(SimpleNamespace(api=api), ["box"])
    assert scene is not None and scene["count"] == 0 and scene["reach_prior"]["reachable"] is True


def test_perceive_scene_no_reach_prior_without_capability():
    # No planning.reachability → no reach envelope in the prompt (piper unchanged on that field).
    api = SimpleNamespace(capabilities={"vision.detection"},
                          analyze_scene=lambda t: {"ok": True, "objects": []})
    scene = _perceive_scene(SimpleNamespace(api=api), ["box"])
    assert scene is not None and "reach_prior" not in scene


def _box(name, cx, cy, cz, w, h):
    return {"object": name, "center_mm": [cx, cy, cz], "distance_mm": (cx * cx + cy * cy + cz * cz) ** 0.5,
            "width_mm": w, "height_mm": h, "front_x_mm": cx - w / 2, "back_x_mm": cx + w / 2, "top_z_mm": cz + h / 2}


def _two_apple_api(apples):
    scene = {"apple": apples, "drawer": [_box("drawer", 600, 0, 700, 400, 300)]}
    return SimpleNamespace(capabilities={"vision.detection"},
                           analyze_scene=lambda n: {"ok": True, "objects": scene.get(n, [])})


_IN_DRAWER = {"apple": {"reference": "drawer", "relation": "in"}}


def test_qualifier_rejects_the_same_label_object_elsewhere():
    # Task says "the apple IN the drawer"; the only apple in view is on the table. It must not be
    # advertised as the target — planning against it is how the wrong apple gets grasped.
    api = _two_apple_api([_box("apple", 350, 450, 780, 70, 70)])
    scene = _perceive_scene(SimpleNamespace(api=api), ["apple"], ["drawer"], _IN_DRAWER)
    assert scene["count"] == 0 and scene["objects"] == []
    assert scene["missing"] == ["apple"]                       # the QUALIFIED target is absent
    assert scene["unqualified"][0]["count"] == 1               # ...but say what was rejected
    assert scene["unqualified"][0]["relation"] == "in"


def test_qualifier_keeps_the_matching_one_and_still_reports_the_other():
    api = _two_apple_api([_box("apple", 350, 450, 780, 70, 70), _box("apple", 650, 0, 650, 70, 70)])
    scene = _perceive_scene(SimpleNamespace(api=api), ["apple"], ["drawer"], _IN_DRAWER)
    assert [o["center_mm"] for o in scene["objects"]] == [[650, 0, 650]]
    assert "missing" not in scene
    assert scene["unqualified"][0]["count"] == 1               # the table apple, reported not erased


def test_qualifier_not_applied_when_reference_unseen():
    # Reference not found → nothing judged. A qualifier we could not evaluate must not
    # disqualify the only candidate we have.
    api = SimpleNamespace(capabilities={"vision.detection"},
                          analyze_scene=lambda n: {"ok": True,
                                                   "objects": [_box("apple", 350, 450, 780, 70, 70)] if n == "apple" else []})
    scene = _perceive_scene(SimpleNamespace(api=api), ["apple"], ["drawer"], _IN_DRAWER)
    assert scene["count"] == 1 and "unqualified" not in scene


def test_qualifier_not_applied_to_records_without_measured_bounds():
    # extent_of defaults absent bounds to 0.0 (= the base origin); judging on that would answer
    # confidently about geometry nobody measured.
    scene_objs = {"apple": [{"object": "apple", "center_mm": [350, 450, 780], "distance_mm": 950.0}],
                  "drawer": [_box("drawer", 600, 0, 700, 400, 300)]}
    api = SimpleNamespace(capabilities={"vision.detection"},
                          analyze_scene=lambda n: {"ok": True, "objects": scene_objs.get(n, [])})
    scene = _perceive_scene(SimpleNamespace(api=api), ["apple"], ["drawer"], _IN_DRAWER)
    assert scene["count"] == 1 and "unqualified" not in scene
