# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Process-level smoke test: drive every emitted SO-101 tool via the smoke harness.

Mirrors the plan §A7 requirement: call ``smoke_test_api`` in-process with a fake
env (driver spy already bound) so every ``@robot_tool`` runs once and its return
is JSON-serializable — surfacing spelling/contract bugs that static validation
cannot. No LeRobot / hardware needed.
"""

from __future__ import annotations

import pytest

from scripts.smoke_test_adapter import smoke_test_api

from .test_env_api import _build_api  # reuse the SpyDriver-backed env/api


@pytest.fixture(scope="module")
def smoke_results():
    """Exercise the tool surface once with detection kept fully in-process."""
    api, env, _driver = _build_api()
    api._seg_fn = lambda _image, *, text_prompt: []
    return smoke_test_api(api, env=env)


class TestSmokeEveryTool:
    def test_all_emitted_tools_run_and_serialize(self, smoke_results):
        failures = [r for r in smoke_results if r["status"] == "fail"]
        assert not failures, "smoke_test_api failures: " + "; ".join(f"{r['name']}: {r.get('error')}" for r in failures)

        # The milestone-A tool set must have been exercised (pass or skip).
        names = {r["name"] for r in smoke_results}
        for expected in ("home", "goto_xyzr", "goto_pose", "open_gripper", "close_gripper"):
            assert expected in names, f"smoke harness did not visit tool {expected}"

    def test_expected_tool_names(self, smoke_results):
        names = {r["name"] for r in smoke_results}
        # Motion + grasp + vision tools all emitted (capabilities intersect).
        for expected in (
            "home",
            "goto_xyzr",
            "goto_pose",
            "open_gripper",
            "close_gripper",
            "get_grasp_info_simple",
            "pixel_to_base_xyz",
            "get_image",
            "analyze_scene",
        ):
            assert expected in names, f"smoke harness did not visit tool {expected}"
