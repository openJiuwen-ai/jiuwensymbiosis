# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Ground-truth visual_pick end-to-end test: real perception pipeline vs mock scene.

Verifies the full perception → projection → motion closed loop against known
ground truth, using ``SceneMockApi`` which runs the real ``detect_and_centroid``
+ ``pixel_and_depth_to_camera_xyz`` pipeline against a ``MockScene``'s rendered
frames. No hardware, no detector subprocess — purely synthetic and deterministic.

The workflow driver (``run_visual_pick_flow``) is imported from
``test_visual_pick_flow`` to avoid duplication.
"""

from __future__ import annotations

import pytest

from jiuwensymbiosis.env.mock import MockArmEnv
from tests.mocks.mock_scene import MockObject, MockScene
from tests.mocks.scene_api import SceneMockApi
from tests.unit_tests.skills.test_visual_pick_flow import run_visual_pick_flow


def _make_scene_api(obj: MockObject) -> tuple[MockArmEnv, SceneMockApi]:
    """Build a MockArmEnv + SceneMockApi with one object in the scene."""
    scene = MockScene(objects=[obj])
    env = MockArmEnv(scene=scene)
    api = SceneMockApi(env)
    return env, api


class TestGroundTruthProjection:
    """The real pipeline recovers the object's base-frame XYZ from rendered pixels."""

    def test_single_object_position_matches_ground_truth(self):
        obj = MockObject(
            name="red box",
            base_xy_mm=(230.0, 0.0),
            size_mm=(40.0, 40.0, 30.0),
            color=(255, 0, 0),
        )
        env, api = _make_scene_api(obj)
        det = api.get_grasp_info_simple("red box")
        assert det["ok"] is True
        # position = [x, y, top_z] must match the object's base_xy + height
        assert det["position"][0] == pytest.approx(230.0, abs=2.0)
        assert det["position"][1] == pytest.approx(0.0, abs=2.0)
        assert det["position"][2] == pytest.approx(30.0, abs=2.0)

    def test_object_off_center_y(self):
        obj = MockObject(
            name="blue block",
            base_xy_mm=(180.0, 80.0),
            size_mm=(40.0, 40.0, 25.0),
            color=(0, 0, 255),
        )
        env, api = _make_scene_api(obj)
        det = api.get_grasp_info_simple("blue block")
        assert det["ok"] is True
        assert det["position"][0] == pytest.approx(180.0, abs=2.0)
        assert det["position"][1] == pytest.approx(80.0, abs=2.0)
        assert det["position"][2] == pytest.approx(25.0, abs=2.0)

    def test_grasp_position_is_position_with_grasp_z(self):
        obj = MockObject(
            name="green cube",
            base_xy_mm=(200.0, -50.0),
            size_mm=(40.0, 40.0, 20.0),
            color=(0, 255, 0),
        )
        env, api = _make_scene_api(obj)
        det = api.get_grasp_info_simple("green cube")
        assert det["ok"] is True
        # grasp_z = top_z + offset(0) = top_z; grasp_position = [x, y, grasp_z]
        assert det["grasp_z"] == pytest.approx(det["position"][2])
        assert det["grasp_position"] == pytest.approx([det["position"][0], det["position"][1], det["grasp_z"]])

    def test_pixel_to_base_xyz_recovers_ground_truth(self):
        obj = MockObject(
            name="red box",
            base_xy_mm=(230.0, 0.0),
            size_mm=(40.0, 40.0, 30.0),
            color=(255, 0, 0),
        )
        env, api = _make_scene_api(obj)
        det = api.get_grasp_info_simple("red box")
        u, v = det["pixel_uv"]
        depth_m = det["depth_m"]
        xyz = api.pixel_to_base_xyz(u, v, depth_m)
        assert xyz["x"] == pytest.approx(230.0, abs=2.0)
        assert xyz["y"] == pytest.approx(0.0, abs=2.0)
        assert xyz["z"] == pytest.approx(30.0, abs=2.0)


class TestGroundTruthVisualPickFlow:
    """End-to-end: visual_pick workflow descends to the object's ground-truth position."""

    def test_descend_step_matches_object_position(self):
        obj = MockObject(
            name="red box",
            base_xy_mm=(230.0, 0.0),
            size_mm=(40.0, 40.0, 30.0),
            color=(255, 0, 0),
        )
        env, api = _make_scene_api(obj)
        det = run_visual_pick_flow(api, "red box")
        assert det["ok"] is True
        log = env._move_log
        assert len(log) == 4  # [home, approach, descend, lift]
        descend = log[-2]
        # descend x,y must match the object's base_xy (within pixel-quantization tolerance)
        assert descend["x"] == pytest.approx(230.0, abs=3.0)
        assert descend["y"] == pytest.approx(0.0, abs=3.0)
        # descend z must match grasp_z = object height + offset(0)
        assert descend["z"] == pytest.approx(30.0, abs=3.0)

    def test_lift_step_above_descend(self):
        obj = MockObject(
            name="red box",
            base_xy_mm=(230.0, 0.0),
            size_mm=(40.0, 40.0, 30.0),
            color=(255, 0, 0),
        )
        env, api = _make_scene_api(obj)
        run_visual_pick_flow(api, "red box")
        log = env._move_log
        assert log[-1]["z"] > log[-2]["z"]


class TestMultiObjectScene:
    """Multiple objects: text_prompt selects the right target."""

    def test_selects_named_object_among_several(self):
        scene = MockScene(
            objects=[
                MockObject("red box", (230.0, 0.0), (40.0, 40.0, 30.0), (255, 0, 0)),
                MockObject("blue box", (180.0, 80.0), (40.0, 40.0, 25.0), (0, 0, 255)),
            ]
        )
        env = MockArmEnv(scene=scene)
        api = SceneMockApi(env)

        det_red = api.get_grasp_info_simple("red box")
        assert det_red["ok"] is True
        assert det_red["position"][0] == pytest.approx(230.0, abs=3.0)
        assert det_red["position"][1] == pytest.approx(0.0, abs=3.0)

        det_blue = api.get_grasp_info_simple("blue box")
        assert det_blue["ok"] is True
        assert det_blue["position"][0] == pytest.approx(180.0, abs=3.0)
        assert det_blue["position"][1] == pytest.approx(80.0, abs=3.0)

    def test_unknown_object_returns_no_detection(self):
        scene = MockScene(objects=[MockObject("red box", (230.0, 0.0), (40.0, 40.0, 30.0), (255, 0, 0))])
        env = MockArmEnv(scene=scene)
        api = SceneMockApi(env)
        det = api.get_grasp_info_simple("yellow sphere")
        assert det["ok"] is False
        assert det.get("reason") == "no_detection"


class TestAnalyzeScene:
    def test_analyze_scene_counts_detections(self):
        scene = MockScene(
            objects=[
                MockObject("red box", (230.0, 0.0), (40.0, 40.0, 30.0), (255, 0, 0)),
                MockObject("blue box", (180.0, 80.0), (40.0, 40.0, 25.0), (0, 0, 255)),
            ]
        )
        env = MockArmEnv(scene=scene)
        api = SceneMockApi(env)
        result = api.analyze_scene("box")
        assert result["ok"] is True
        # "box" is a substring of both "red box" and "blue box"
        assert result["n_detections"] == 2
        assert len(result["top_scores"]) == 2
