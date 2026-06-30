# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.rails.safety."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jiuwensymbiosis.agent.session import RobotSession
from jiuwensymbiosis.env.mock import MockArmEnv
from jiuwensymbiosis.rails.safety import SafetyRail
from tests.mocks.mock_api import MockApi


class _FakeCtx:
    def __init__(self, tool_name="", tool_args=None):
        self.inputs = MagicMock()
        self.inputs.tool_name = tool_name
        self.inputs.tool_args = tool_args or {}


@pytest.fixture
def mock_session():
    env = MockArmEnv()
    api = MockApi(env)
    return RobotSession(env=env, api=api, name="test")


class TestSafetyRailZFloor:
    @pytest.mark.asyncio
    async def test_z_above_floor_passes(self, mock_session):
        rail = SafetyRail(mock_session, z_floor_mm=50.0)
        ctx = _FakeCtx(tool_name="goto_xyzr", tool_args={"x": 100, "y": 0, "z": 200, "r": 0})
        await rail.before_tool_call(ctx)

    @pytest.mark.asyncio
    async def test_z_below_floor_raises(self, mock_session):
        rail = SafetyRail(mock_session, z_floor_mm=50.0)
        ctx = _FakeCtx(tool_name="goto_xyzr", tool_args={"x": 100, "y": 0, "z": 30, "r": 0})
        with pytest.raises(ValueError, match="below z_floor"):
            await rail.before_tool_call(ctx)

    @pytest.mark.asyncio
    async def test_non_motion_tool_passes(self, mock_session):
        rail = SafetyRail(mock_session, z_floor_mm=50.0)
        ctx = _FakeCtx(tool_name="get_pose", tool_args={})
        await rail.before_tool_call(ctx)


class TestSafetyRailXYBounds:
    @pytest.mark.asyncio
    async def test_within_bounds_passes(self, mock_session):
        rail = SafetyRail(mock_session, xy_bounds_mm=(0, -300, 500, 300))
        ctx = _FakeCtx(tool_name="goto_xyzr", tool_args={"x": 250, "y": 0, "z": 200, "r": 0})
        await rail.before_tool_call(ctx)

    @pytest.mark.asyncio
    async def test_x_out_of_bounds_raises(self, mock_session):
        rail = SafetyRail(mock_session, xy_bounds_mm=(0, -300, 500, 300))
        ctx = _FakeCtx(tool_name="goto_xyzr", tool_args={"x": 600, "y": 0, "z": 200, "r": 0})
        with pytest.raises(ValueError, match="out of bounds"):
            await rail.before_tool_call(ctx)

    @pytest.mark.asyncio
    async def test_y_out_of_bounds_raises(self, mock_session):
        rail = SafetyRail(mock_session, xy_bounds_mm=(0, -300, 500, 300))
        ctx = _FakeCtx(tool_name="goto_xyzr", tool_args={"x": 250, "y": -400, "z": 200, "r": 0})
        with pytest.raises(ValueError, match="out of bounds"):
            await rail.before_tool_call(ctx)


class TestSafetyRailXYFromEnv:
    """SafetyRail(session) with no xy_bounds_mm enforces XY from env.workspace_bounds."""

    @pytest.fixture
    def bounded_session(self):
        env = MockArmEnv(workspace_bounds=(0.0, -300.0, 500.0, 300.0))
        api = MockApi(env)
        return RobotSession(env=env, api=api, name="bounded")

    @pytest.mark.asyncio
    async def test_within_env_bounds_passes(self, bounded_session):
        rail = SafetyRail(bounded_session)  # no xy_bounds_mm
        ctx = _FakeCtx(tool_name="goto_xyzr", tool_args={"x": 250, "y": 0, "z": 200, "r": 0})
        await rail.before_tool_call(ctx)

    @pytest.mark.asyncio
    async def test_out_of_env_bounds_raises(self, bounded_session):
        rail = SafetyRail(bounded_session)  # no xy_bounds_mm
        ctx = _FakeCtx(tool_name="goto_xyzr", tool_args={"x": 600, "y": 0, "z": 200, "r": 0})
        with pytest.raises(ValueError, match="out of bounds"):
            await rail.before_tool_call(ctx)

    @pytest.mark.asyncio
    async def test_enforce_xy_from_env_false_disables_fallback(self, bounded_session):
        rail = SafetyRail(bounded_session, enforce_xy_from_env=False)
        ctx = _FakeCtx(tool_name="goto_xyzr", tool_args={"x": 600, "y": 0, "z": 200, "r": 0})
        await rail.before_tool_call(ctx)  # no env fallback → no XY check

    @pytest.mark.asyncio
    async def test_explicit_bounds_take_precedence(self, bounded_session):
        # Explicit narrow bounds override the env's wider workspace_bounds.
        rail = SafetyRail(bounded_session, xy_bounds_mm=(0, 0, 100, 100))
        ctx = _FakeCtx(tool_name="goto_xyzr", tool_args={"x": 250, "y": 0, "z": 200, "r": 0})
        with pytest.raises(ValueError, match="out of bounds"):
            await rail.before_tool_call(ctx)

    def test_resolve_xy_bounds_reads_env(self, bounded_session):
        rail = SafetyRail(bounded_session)
        assert rail._resolve_xy_bounds() == (0.0, -300.0, 500.0, 300.0)


class TestSafetyRailRobotControlUnwrap:
    @pytest.mark.asyncio
    async def test_robot_control_unwrap(self, mock_session):
        rail = SafetyRail(mock_session, z_floor_mm=50.0)
        ctx = _FakeCtx(
            tool_name="robot_control",
            tool_args={"action": "goto_xyzr", "params": {"x": 100, "y": 0, "z": 30, "r": 0}},
        )
        with pytest.raises(ValueError, match="below z_floor"):
            await rail.before_tool_call(ctx)

    @pytest.mark.asyncio
    async def test_resolve_z_floor_from_env(self, mock_session):
        rail = SafetyRail(mock_session)
        z = rail._resolve_z_floor()
        assert z == 0.0


class TestSafetyRailStringToolArgs:
    """openjiuwen delivers tool_args as a JSON *string* (ToolCall.arguments is
    typed str) — the dict only materialises inside the tool's invoke, *after*
    rails run. SafetyRail must parse the string itself or its z/XY checks
    silently no-op (the bug these tests pin down)."""

    @pytest.mark.asyncio
    async def test_direct_goto_string_args_z_below_floor_raises(self, mock_session):
        rail = SafetyRail(mock_session, z_floor_mm=50.0)
        ctx = _FakeCtx(
            tool_name="goto_xyzr",
            tool_args='{"x": 100, "y": 0, "z": 30, "r": 0}',
        )
        with pytest.raises(ValueError, match="below z_floor"):
            await rail.before_tool_call(ctx)

    @pytest.mark.asyncio
    async def test_direct_goto_string_args_x_out_of_bounds_raises(self, mock_session):
        rail = SafetyRail(mock_session, xy_bounds_mm=(0, -300, 500, 300))
        ctx = _FakeCtx(
            tool_name="goto_xyzr",
            tool_args='{"x": 600, "y": 0, "z": 200, "r": 0}',
        )
        with pytest.raises(ValueError, match="out of bounds"):
            await rail.before_tool_call(ctx)

    @pytest.mark.asyncio
    async def test_robot_control_string_args_unwrap_and_reject(self, mock_session):
        """The fast path + skill path both dispatch via robot_control with a
        string arguments payload — SafetyRail must unwrap + check it."""
        rail = SafetyRail(mock_session, z_floor_mm=50.0)
        ctx = _FakeCtx(
            tool_name="robot_control",
            tool_args='{"action": "goto_xyzr", "params": {"x": 100, "y": 0, "z": 30, "r": 0}}',
        )
        with pytest.raises(ValueError, match="below z_floor"):
            await rail.before_tool_call(ctx)

    @pytest.mark.asyncio
    async def test_robot_control_string_args_xy_rejected(self, mock_session):
        rail = SafetyRail(mock_session, xy_bounds_mm=(0, -300, 500, 300))
        ctx = _FakeCtx(
            tool_name="robot_control",
            tool_args='{"action": "goto_xyzr", "params": {"x": 600, "y": 0, "z": 200, "r": 0}}',
        )
        with pytest.raises(ValueError, match="out of bounds"):
            await rail.before_tool_call(ctx)

    @pytest.mark.asyncio
    async def test_safe_string_args_passes(self, mock_session):
        rail = SafetyRail(mock_session, z_floor_mm=50.0, xy_bounds_mm=(0, -300, 500, 300))
        ctx = _FakeCtx(
            tool_name="goto_xyzr",
            tool_args='{"x": 250, "y": 0, "z": 200, "r": 0}',
        )
        await rail.before_tool_call(ctx)  # in-bounds, above floor → no rejection

    @pytest.mark.asyncio
    async def test_malformed_string_args_does_not_raise(self, mock_session):
        """A malformed JSON string degrades to {} (no params → no rejection),
        never a false positive and never a crash — same behaviour as a non-dict."""
        rail = SafetyRail(mock_session, z_floor_mm=50.0)
        ctx = _FakeCtx(tool_name="goto_xyzr", tool_args="not-json{")
        await rail.before_tool_call(ctx)


class TestSafetyRailTraceSink:
    @pytest.mark.asyncio
    async def test_reject_notifies_sink(self, mock_session):
        class _Sink:
            def __init__(self):
                self.events = []

            def record_rail_event(self, *, rail_name, kind, detail, success):
                self.events.append((rail_name, kind, detail, success))

        sink = _Sink()
        rail = SafetyRail(mock_session, z_floor_mm=50.0, trace_sink=sink)
        ctx = _FakeCtx(tool_name="goto_xyzr", tool_args={"x": 100, "y": 0, "z": 30})
        with pytest.raises(ValueError):
            await rail.before_tool_call(ctx)
        assert sink.events
        assert sink.events[0][0] == "SafetyRail"
        assert sink.events[0][3] is False

    @pytest.mark.asyncio
    async def test_no_sink_does_not_raise(self, mock_session):
        rail = SafetyRail(mock_session, z_floor_mm=50.0, trace_sink=None)
        ctx = _FakeCtx(tool_name="goto_xyzr", tool_args={"x": 100, "y": 0, "z": 30})
        with pytest.raises(ValueError):
            await rail.before_tool_call(ctx)  # no crash with sink=None
