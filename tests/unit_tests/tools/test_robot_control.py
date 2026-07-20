# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.tools.robot_control_tool."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jiuwensymbiosis.agent.abstractions import ToolOutput
from jiuwensymbiosis.tools.robot_control_tool import RobotControlTool, _build_action_index


class TestBuildActionIndex:
    def test_contains_expected_actions(self, mock_api):
        index = _build_action_index(mock_api)
        assert "home" in index
        assert "goto_xyzr" in index
        assert "close_gripper" in index

    def test_values_are_bound_methods(self, mock_api):
        index = _build_action_index(mock_api)
        for method in index.values():
            assert callable(method)

    def test_env_capabilities_gate_actions(self, mock_api):
        env = SimpleNamespace(capabilities=frozenset({"motion.cartesian"}))
        index = _build_action_index(mock_api, env=env)
        assert "goto_xyzr" in index
        assert "close_gripper" not in index
        assert "get_image" not in index


class TestRobotControlTool:
    def test_construction(self, mock_api):
        tool = RobotControlTool(mock_api)
        assert tool.card.name == "robot_control"

    def test_available_actions(self, mock_api):
        tool = RobotControlTool(mock_api)
        assert "home" in tool.available_actions

    @pytest.mark.asyncio
    async def test_invoke_valid_action(self, mock_api):
        tool = RobotControlTool(mock_api)
        result = await tool.invoke({"action": "home"})
        assert isinstance(result, ToolOutput)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_invoke_with_params(self, mock_api):
        tool = RobotControlTool(mock_api)
        result = await tool.invoke({"action": "goto_xyzr", "params": {"x": 100, "y": 50, "z": 300}})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_invoke_invalid_action(self, mock_api):
        tool = RobotControlTool(mock_api)
        result = await tool.invoke({"action": "nonexistent"})
        assert result.success is False
        assert "unknown action" in result.error

    @pytest.mark.asyncio
    async def test_invoke_missing_action(self, mock_api):
        tool = RobotControlTool(mock_api)
        result = await tool.invoke({})
        assert result.success is False
        assert "required" in result.error

    @pytest.mark.asyncio
    async def test_invoke_bad_params(self, mock_api):
        tool = RobotControlTool(mock_api)
        result = await tool.invoke({"action": "goto_xyzr", "params": "not_a_dict"})
        assert result.success is False
