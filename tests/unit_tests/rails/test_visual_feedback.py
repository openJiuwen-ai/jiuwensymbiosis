# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.rails.visual_feedback."""

from __future__ import annotations

import numpy as np
import pytest

from jiuwensymbiosis.rails.visual_feedback import VisualFeedbackRail, _encode_jpeg_b64
from jiuwensymbiosis.env.mock import MockArmEnv
from jiuwensymbiosis.agent.session import RobotSession
from tests.mocks.mock_api import MockApi


@pytest.fixture
def mock_session():
    env = MockArmEnv()
    api = MockApi(env)
    return RobotSession(env=env, api=api, name="test")


class TestShouldTrigger:
    def test_motion_tag_triggers(self, mock_session):
        rail = VisualFeedbackRail(mock_session)
        assert rail._should_trigger("goto_xyzr", tool_args=None) is True

    def test_grasp_tag_triggers(self, mock_session):
        rail = VisualFeedbackRail(mock_session)
        assert rail._should_trigger("close_gripper", tool_args=None) is True

    def test_other_tool_does_not_trigger(self, mock_session):
        rail = VisualFeedbackRail(mock_session)
        assert rail._should_trigger("get_pose", tool_args={}) is False

    def test_watch_tools_triggers(self, mock_session):
        rail = VisualFeedbackRail(mock_session, watch_tools={"custom_tool"})
        assert rail._should_trigger("custom_tool", tool_args=None) is True

    def test_robot_control_unwrap(self, mock_session):
        rail = VisualFeedbackRail(mock_session)
        result = rail._should_trigger("robot_control", tool_args={"action": "goto_xyzr", "params": {}})
        assert result is True


class TestEncodeJpegB64:
    def test_valid_image(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = _encode_jpeg_b64(img, quality=80)
        assert result is not None
        import base64

        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_none_on_bad_input(self):
        assert _encode_jpeg_b64("not_array") is None

    def test_none_on_wrong_ndim(self):
        assert _encode_jpeg_b64(np.zeros((10, 10))) is None


class TestGrabFrameB64:
    def test_grabs_from_env(self, mock_session):
        rail = VisualFeedbackRail(mock_session)
        result = rail._grab_frame_b64()
        assert result is not None


class TestVisualFeedbackTraceSink:
    @pytest.mark.asyncio
    async def test_inject_notifies_sink(self, mock_session):
        from jiuwensymbiosis.rails.visual_feedback import VisualFeedbackRail
        from unittest.mock import MagicMock

        class _Sink:
            def __init__(self):
                self.events = []

            def record_rail_event(self, *, rail_name, kind, detail, success):
                self.events.append((rail_name, kind, detail, success))

        sink = _Sink()
        rail = VisualFeedbackRail(mock_session, trace_sink=sink)
        inputs = MagicMock()
        inputs.tool_name = "goto_xyzr"
        inputs.tool_args = {}
        ctx = MagicMock()
        ctx.inputs = inputs
        ctx.extra = {}
        ctx.context = None
        await rail.after_tool_call(ctx)
        assert sink.events
        assert sink.events[0][0] == "VisualFeedback"

    def test_grab_frame_b64_still_works(self, mock_session):
        # regression: refactor kept the public helper
        from jiuwensymbiosis.rails.visual_feedback import VisualFeedbackRail

        rail = VisualFeedbackRail(mock_session)
        assert rail._grab_frame_b64() is not None
