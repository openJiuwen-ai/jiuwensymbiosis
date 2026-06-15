# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.rails.safety."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jiuwensymbiosis.rails.safety import SafetyRail
from jiuwensymbiosis.env.mock import MockArmEnv
from jiuwensymbiosis.agent.session import RobotSession
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
