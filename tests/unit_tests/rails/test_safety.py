# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.rails.safety."""

from __future__ import annotations

import pytest

from jiuwensymbiosis.env.mock import MockArmEnv
from jiuwensymbiosis.rails.safety import SafetyRail
from tests.helpers import FakeCtx, RecordingRailSink, make_mock_session
from tests.mocks.mock_api import MockApi


@pytest.fixture
def mock_session():
    return make_mock_session()


GOTO_ARGS = {"x": 100, "y": 0, "z": 200, "r": 0}


class TestSafetyRailZFloor:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool_name", "tool_args"),
        [
            ("goto_xyzr", GOTO_ARGS),
            ("get_pose", {}),
        ],
        ids=["motion-above-floor", "non-motion"],
    )
    async def test_safe_calls_pass(self, mock_session, tool_name, tool_args):
        rail = SafetyRail(mock_session, z_floor_mm=50.0)
        ctx = FakeCtx(tool_name=tool_name, tool_args=tool_args)
        await rail.before_tool_call(ctx)

    @pytest.mark.asyncio
    async def test_z_below_floor_raises(self, mock_session):
        rail = SafetyRail(mock_session, z_floor_mm=50.0)
        ctx = FakeCtx(tool_name="goto_xyzr", tool_args={"x": 100, "y": 0, "z": 30, "r": 0})
        with pytest.raises(ValueError, match="below z_floor"):
            await rail.before_tool_call(ctx)


class TestSafetyRailXYBounds:
    @pytest.mark.asyncio
    async def test_within_bounds_passes(self, mock_session):
        rail = SafetyRail(mock_session, xy_bounds_mm=(0, -300, 500, 300))
        ctx = FakeCtx(tool_name="goto_xyzr", tool_args={"x": 250, "y": 0, "z": 200, "r": 0})
        await rail.before_tool_call(ctx)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_args",
        [
            {"x": 600, "y": 0, "z": 200, "r": 0},
            {"x": 250, "y": -400, "z": 200, "r": 0},
        ],
        ids=["x-out", "y-out"],
    )
    async def test_out_of_bounds_raises(self, mock_session, tool_args):
        rail = SafetyRail(mock_session, xy_bounds_mm=(0, -300, 500, 300))
        ctx = FakeCtx(tool_name="goto_xyzr", tool_args=tool_args)
        with pytest.raises(ValueError, match="out of bounds"):
            await rail.before_tool_call(ctx)


class TestSafetyRailXYFromEnv:
    """SafetyRail(session) with no xy_bounds_mm enforces XY from env.workspace_bounds."""

    @pytest.fixture
    def bounded_session(self):
        env = MockArmEnv(workspace_bounds=(0.0, -300.0, 500.0, 300.0))
        api = MockApi(env)
        return make_mock_session(name="bounded", env=env, api=api)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("rail_kwargs", "tool_args", "should_raise"),
        [
            ({}, {"x": 250, "y": 0, "z": 200, "r": 0}, False),
            ({}, {"x": 600, "y": 0, "z": 200, "r": 0}, True),
            ({"enforce_xy_from_env": False}, {"x": 600, "y": 0, "z": 200, "r": 0}, False),
            ({"xy_bounds_mm": (0, 0, 100, 100)}, {"x": 250, "y": 0, "z": 200, "r": 0}, True),
        ],
        ids=["within-env-bounds", "outside-env-bounds", "env-fallback-disabled", "explicit-bounds-precedence"],
    )
    async def test_env_bounds_policy(self, bounded_session, rail_kwargs, tool_args, should_raise):
        rail = SafetyRail(bounded_session, **rail_kwargs)
        ctx = FakeCtx(tool_name="goto_xyzr", tool_args=tool_args)
        if should_raise:
            with pytest.raises(ValueError, match="out of bounds"):
                await rail.before_tool_call(ctx)
        else:
            await rail.before_tool_call(ctx)

    def test_resolve_xy_bounds_reads_env(self, bounded_session):
        rail = SafetyRail(bounded_session)
        assert rail._resolve_xy_bounds() == (0.0, -300.0, 500.0, 300.0)


class TestSafetyRailRobotControlUnwrap:
    @pytest.mark.asyncio
    async def test_robot_control_unwrap(self, mock_session):
        rail = SafetyRail(mock_session, z_floor_mm=50.0)
        ctx = FakeCtx(
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
    @pytest.mark.parametrize(
        ("rail_kwargs", "tool_name", "tool_args", "error_match"),
        [
            ({"z_floor_mm": 50.0}, "goto_xyzr", '{"x": 100, "y": 0, "z": 30, "r": 0}', "below z_floor"),
            (
                {"xy_bounds_mm": (0, -300, 500, 300)},
                "goto_xyzr",
                '{"x": 600, "y": 0, "z": 200, "r": 0}',
                "out of bounds",
            ),
            (
                {"z_floor_mm": 50.0},
                "robot_control",
                '{"action": "goto_xyzr", "params": {"x": 100, "y": 0, "z": 30, "r": 0}}',
                "below z_floor",
            ),
            (
                {"xy_bounds_mm": (0, -300, 500, 300)},
                "robot_control",
                '{"action": "goto_xyzr", "params": {"x": 600, "y": 0, "z": 200, "r": 0}}',
                "out of bounds",
            ),
            (
                {"z_floor_mm": 50.0, "xy_bounds_mm": (0, -300, 500, 300)},
                "goto_xyzr",
                '{"x": 250, "y": 0, "z": 200, "r": 0}',
                None,
            ),
            ({"z_floor_mm": 50.0}, "goto_xyzr", "not-json{", None),
        ],
        ids=["direct-z-low", "direct-x-out", "robot-control-z-low", "robot-control-x-out", "safe", "malformed"],
    )
    async def test_string_args_policy(self, mock_session, rail_kwargs, tool_name, tool_args, error_match):
        rail = SafetyRail(mock_session, **rail_kwargs)
        ctx = FakeCtx(tool_name=tool_name, tool_args=tool_args)
        if error_match:
            with pytest.raises(ValueError, match=error_match):
                await rail.before_tool_call(ctx)
        else:
            await rail.before_tool_call(ctx)


class TestSafetyRailTraceSink:
    @pytest.mark.asyncio
    async def test_reject_notifies_sink(self, mock_session):
        sink = RecordingRailSink()
        rail = SafetyRail(mock_session, z_floor_mm=50.0, trace_sink=sink)
        ctx = FakeCtx(tool_name="goto_xyzr", tool_args={"x": 100, "y": 0, "z": 30})
        with pytest.raises(ValueError):
            await rail.before_tool_call(ctx)
        assert sink.events
        assert sink.events[0][0] == "SafetyRail"
        assert sink.events[0][3] is False

    @pytest.mark.asyncio
    async def test_no_sink_does_not_raise(self, mock_session):
        rail = SafetyRail(mock_session, z_floor_mm=50.0, trace_sink=None)
        ctx = FakeCtx(tool_name="goto_xyzr", tool_args={"x": 100, "y": 0, "z": 30})
        with pytest.raises(ValueError):
            await rail.before_tool_call(ctx)  # no crash with sink=None
