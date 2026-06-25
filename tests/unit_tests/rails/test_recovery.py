# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.rails.recovery."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jiuwensymbiosis.rails.recovery import RecoveryRail
from jiuwensymbiosis.env.mock import MockArmEnv
from jiuwensymbiosis.agent.session import RobotSession
from tests.mocks.mock_api import MockApi


class _FakeCtx:
    def __init__(self, tool_name="", tool_args=None, api=None):
        self.inputs = MagicMock()
        self.inputs.tool_name = tool_name
        self.inputs.tool_args = tool_args or {}


@pytest.fixture
def mock_session():
    env = MockArmEnv()
    api = MockApi(env)
    return RobotSession(env=env, api=api, name="test")


class TestIsWatched:
    def test_motion_tag_watched(self, mock_session):
        rail = RecoveryRail(mock_session)
        api = mock_session.api
        home_method = getattr(api, "home", None)
        assert (
            rail._is_watched("home", tool_args=None) is True
            or rail._is_watched("close_gripper", tool_args=None) is True
        )

    def test_other_tool_not_watched_without_tags(self, mock_session):
        rail = RecoveryRail(mock_session)
        assert rail._is_watched("get_pose", tool_args=None) is False


class TestRecoveryRail:
    @pytest.mark.asyncio
    async def test_recovery_on_grasp_exception(self, mock_session):
        rail = RecoveryRail(mock_session)
        ctx = _FakeCtx(tool_name="close_gripper", tool_args={})
        await rail.on_tool_exception(ctx)
        assert "home" in mock_session.api._call_log or any("open_gripper" in c for c in mock_session.api._call_log)

    @pytest.mark.asyncio
    async def test_recovery_on_motion_exception(self, mock_session):
        rail = RecoveryRail(mock_session)
        ctx = _FakeCtx(tool_name="goto_xyzr", tool_args={"x": 100, "y": 0, "z": 300, "r": 0})
        await rail.on_tool_exception(ctx)
        assert "home" in mock_session.api._call_log


class TestRecoveryRailTraceSink:
    @pytest.mark.asyncio
    async def test_recover_notifies_sink_with_result(self, mock_session):
        class _Sink:
            def __init__(self):
                self.events = []

            def record_rail_event(self, *, rail_name, kind, detail, success):
                self.events.append((rail_name, kind, detail, success))

        sink = _Sink()
        rail = RecoveryRail(mock_session, trace_sink=sink)
        ctx = _FakeCtx(tool_name="goto_xyzr", tool_args={"x": 1, "y": 2, "z": 3})
        await rail.on_tool_exception(ctx)
        assert sink.events
        detail = sink.events[0][2]
        assert "home_ok" in detail
        assert sink.events[0][3] is True  # mock home succeeds
