# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""End-to-end visual_pick workflow tests against the mock api/env.

Drives the visual_pick SKILL.md 7-step workflow by calling api methods directly
(not via RobotControlTool -- that dispatch layer is covered by
test_robot_control.py, and rails are covered by tests/unit_tests/rails/).

Verifies the perception -> planning -> motion closed loop:
- get_grasp_info_simple returns a grasp_position
- the workflow issues goto_xyzr commands targeting that grasp_position
- env._move_log records the descend step at grasp_z (move_log[-2], since the
  final lift step is move_log[-1])
"""

from __future__ import annotations

import pytest

from jiuwensymbiosis.env.mock import MockArmEnv
from tests.mocks.mock_api import MockApi


def run_visual_pick_flow(
    api: MockApi,
    object_name: str,
    *,
    approach_mm: float = 40.0,
    lift_mm: float = 60.0,
) -> dict:
    """Execute the visual_pick SKILL.md 7-step workflow against ``api``.

    Steps (parallel-gripper body, matching MockApi's grasp.parallel capability):
      1. home            -- stable camera baseline
      2. open_gripper    -- ensure gripper empty (<release>)
      3. get_grasp_info_simple(object_name)
      4. goto_xyzr(x, y, grasp_z + approach)
      5. goto_xyzr(x, y, grasp_z)          -- descend
      6. close_gripper                      -- (<grasp>)
      7. goto_xyzr(x, y, grasp_z + lift)   -- lift to transport height

    On detection failure (ok=False), performs SKILL.md failure handling
    (release + home) and returns the failed detection without continuing.

    Returns the detection dict from step 3.
    """
    api.home()
    api.open_gripper()
    det = api.get_grasp_info_simple(object_name)
    if not det.get("ok"):
        api.open_gripper()
        api.home()
        return det
    gp = det["grasp_position"]
    x, y, grasp_z = float(gp[0]), float(gp[1]), float(gp[2])
    api.goto_xyzr(x, y, grasp_z + approach_mm)
    api.goto_xyzr(x, y, grasp_z)
    api.close_gripper()
    api.goto_xyzr(x, y, grasp_z + lift_mm)
    return det


_SUCCESS_DETECTION = {
    "ok": True,
    "object": "black box",
    "position": [230.0, 0.0, 50.0],
    "grasp_z": 45.0,
    "grasp_position": [230.0, 0.0, 45.0],
    "place_z": 55.0,
    "place_position": [230.0, 0.0, 55.0],
    "score": 0.9,
    "pixel_uv": [320, 240],
    "depth_m": 0.20,
}


class TestVisualPickFlowSuccess:
    """Success path: detection ok -> full 7-step workflow -> grasp_position reached."""

    def test_descend_step_targets_grasp_position(self):
        env = MockArmEnv()
        api = MockApi(env, detection_result=dict(_SUCCESS_DETECTION))
        det = run_visual_pick_flow(api, "black box")
        assert det["ok"] is True
        log = env._move_log
        # move_log = [home, approach, descend, lift] (gripper calls don't append)
        assert len(log) == 4
        # descend step is move_log[-2]; move_log[-1] is the lift step.
        descend = log[-2]
        assert descend["x"] == pytest.approx(230.0)
        assert descend["y"] == pytest.approx(0.0)
        assert descend["z"] == pytest.approx(45.0)

    def test_lift_step_is_above_descend(self):
        env = MockArmEnv()
        api = MockApi(env, detection_result=dict(_SUCCESS_DETECTION))
        run_visual_pick_flow(api, "black box")
        log = env._move_log
        # lift (log[-1]) must be above descend (log[-2])
        assert log[-1]["z"] > log[-2]["z"]
        assert log[-1]["z"] == pytest.approx(105.0)  # grasp_z(45) + lift(60)

    def test_approach_step_is_above_descend(self):
        env = MockArmEnv()
        api = MockApi(env, detection_result=dict(_SUCCESS_DETECTION))
        run_visual_pick_flow(api, "black box")
        log = env._move_log
        # approach (log[1]) above descend (log[2])
        assert log[1]["z"] > log[2]["z"]
        assert log[1]["z"] == pytest.approx(85.0)  # grasp_z(45) + approach(40)

    def test_final_pose_at_transport_height(self):
        env = MockArmEnv()
        api = MockApi(env, detection_result=dict(_SUCCESS_DETECTION))
        run_visual_pick_flow(api, "black box")
        assert env._pose["x"] == pytest.approx(230.0)
        assert env._pose["y"] == pytest.approx(0.0)
        assert env._pose["z"] == pytest.approx(105.0)

    def test_home_is_first_move(self):
        env = MockArmEnv()
        api = MockApi(env, detection_result=dict(_SUCCESS_DETECTION))
        run_visual_pick_flow(api, "black box")
        log = env._move_log
        # first move is home (default home pose x=200)
        assert log[0]["x"] == pytest.approx(env._home["x"])
        assert log[0]["z"] == pytest.approx(env._home["z"])


class TestVisualPickFlowFailure:
    """Failure path: detection ok=False -> release + home, no motion toward target."""

    def test_no_detection_does_release_and_home_only(self):
        env = MockArmEnv()
        api = MockApi(env, detection_result={"ok": False, "reason": "no_detection"})
        det = run_visual_pick_flow(api, "black box")
        assert det["ok"] is False
        log = env._move_log
        # Only home moves (step1 home + failure home); gripper calls don't append.
        assert len(log) == 2
        home_x = env._home["x"]
        for entry in log:
            assert entry["x"] == pytest.approx(home_x)
        # no move targeted the detection's grasp x (230) -- motion never started
        assert all(entry["x"] != pytest.approx(230.0) for entry in log)

    def test_no_detection_returns_failure_reason(self):
        env = MockArmEnv()
        api = MockApi(env, detection_result={"ok": False, "reason": "no_detection"})
        det = run_visual_pick_flow(api, "black box")
        assert det.get("reason") == "no_detection"
