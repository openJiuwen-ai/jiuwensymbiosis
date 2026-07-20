# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.rails.recovery."""

from __future__ import annotations

import pytest

from jiuwensymbiosis.rails.recovery import RecoveryRail
from tests.helpers import FakeCtx, RecordingRailSink, make_mock_session


@pytest.fixture
def mock_session():
    return make_mock_session()


class TestIsWatched:
    @pytest.mark.parametrize("tool_name", ["home", "close_gripper"], ids=["motion", "grasp"])
    def test_default_tags_watched(self, mock_session, tool_name):
        rail = RecoveryRail(mock_session)
        assert rail._is_watched(tool_name, tool_args=None) is True

    def test_other_tool_not_watched_without_tags(self, mock_session):
        rail = RecoveryRail(mock_session)
        assert rail._is_watched("get_pose", tool_args=None) is False


class TestRecoveryRail:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool_name", "tool_args", "expected_call"),
        [
            ("close_gripper", {}, "home"),
            ("goto_xyzr", {"x": 100, "y": 0, "z": 300, "r": 0}, "home"),
        ],
        ids=["grasp", "motion"],
    )
    async def test_recovery_on_watched_exception(self, mock_session, tool_name, tool_args, expected_call):
        rail = RecoveryRail(mock_session)
        ctx = FakeCtx(tool_name=tool_name, tool_args=tool_args)
        await rail.on_tool_exception(ctx)
        assert any(expected_call in c for c in mock_session.api._call_log)

    @pytest.mark.asyncio
    async def test_typed_safe_failure_skips_recovery(self, mock_session):
        class SafeFailure(RuntimeError):
            skip_recovery = True

        rail = RecoveryRail(mock_session)
        ctx = FakeCtx(tool_name="goto_xyzr", exception=SafeFailure("not reached"))
        await rail.on_tool_exception(ctx)
        assert mock_session.api._call_log == []


class TestRecoveryRailTraceSink:
    @pytest.mark.asyncio
    async def test_recover_notifies_sink_with_result(self, mock_session):
        sink = RecordingRailSink()
        rail = RecoveryRail(mock_session, trace_sink=sink)
        ctx = FakeCtx(tool_name="goto_xyzr", tool_args={"x": 1, "y": 2, "z": 3})
        await rail.on_tool_exception(ctx)
        assert sink.events
        detail = sink.events[0][2]
        assert "home_ok" in detail
        assert sink.events[0][3] is True  # mock home succeeds
