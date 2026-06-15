# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.tools.builder."""

from __future__ import annotations

from jiuwensymbiosis.tools.builder import build_robot_tools, list_tool_meta
from jiuwensymbiosis.agent.abstractions import LocalFunction

from tests.mocks.mock_api import MockApi
from jiuwensymbiosis.env.mock import MockArmEnv


class TestListToolMeta:
    def test_returns_all_tools(self, mock_api):
        metas = list_tool_meta(mock_api)
        assert len(metas) > 0
        for m in metas:
            assert "name" in m
            assert "description" in m

    def test_has_expected_tools(self, mock_api):
        metas = list_tool_meta(mock_api)
        names = {m["name"] for m in metas}
        assert "home" in names
        assert "goto_xyzr" in names
        assert "close_gripper" in names
        assert "get_grasp_info_simple" in names

    def test_capability_gating(self):
        env = MockArmEnv()
        api = MockApi(env)
        metas = list_tool_meta(api)
        for m in metas:
            if m.get("capability"):
                assert m["capability"] in api.capabilities


class TestBuildRobotTools:
    def test_returns_local_function_instances(self, mock_api):
        tools = build_robot_tools(mock_api)
        assert len(tools) > 0
        for t in tools:
            assert isinstance(t, LocalFunction)

    def test_count_matches_list_tool_meta(self, mock_api):
        tools = build_robot_tools(mock_api)
        metas = list_tool_meta(mock_api)
        assert len(tools) == len(metas)

    def test_capability_filtering(self):
        env = MockArmEnv()
        api = MockApi(env)
        metas = list_tool_meta(api)
        names = {m["name"] for m in metas}
        tools = build_robot_tools(api)
        tool_names = {t.card.name for t in tools}
        assert tool_names == names
