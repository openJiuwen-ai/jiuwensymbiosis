# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Full OpenJiuwen agent path for the scripted GUI mock model."""

import pytest

from jiuwensymbiosis.agent import RobotAgentConfig, run_robot_task
from jiuwensymbiosis.agent.abstractions import AgentRail
from jiuwensymbiosis.gui.mock_sessions import build_mock_robot_session, build_scripted_mock_model

pytestmark = [pytest.mark.integration, pytest.mark.timeout(30)]

_SCRIPT = [
    {"tool": "home", "args": {}, "say": "回位"},
    {"tool": "get_grasp_info_simple", "args": {"object_name": "black box"}, "say": "识别"},
    {"tool": "goto_xyzr", "args": {"x": 230, "y": 0, "z": 80}, "say": "移动"},
    {"tool": "close_gripper", "args": {}, "say": "抓取"},
]


def test_scripted_model_drives_full_tool_sequence_end_to_end(tmp_path):
    """The complete DeepAgent path remains covered outside the unit suite."""

    class _CountRail(AgentRail):
        priority = 1

        def __init__(self):
            self.calls: list[str] = []

        async def after_tool_call(self, ctx):
            self.calls.append(getattr(ctx.inputs, "tool_name", ""))

    counter = _CountRail()
    cfg = RobotAgentConfig(
        mode="tool",
        max_iterations=20,
        enable_visual_feedback=False,
        workspace=str(tmp_path),
    )
    cfg.model = build_scripted_mock_model(_SCRIPT)
    cfg.extra_rails = [counter]

    session = build_mock_robot_session()
    with session:
        result = run_robot_task(session, "把黑盒放到白盒上", cfg, conversation_id="test-mock")

    assert counter.calls == ["home", "get_grasp_info_simple", "goto_xyzr", "close_gripper"]
    assert isinstance(result, dict)
    assert result.get("result_type") == "answer"
