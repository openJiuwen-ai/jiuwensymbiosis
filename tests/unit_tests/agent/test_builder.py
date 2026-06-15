# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.agent.builder."""

from __future__ import annotations

import pytest

from jiuwensymbiosis.agent.builder import (
    _resolve_workspace,
    _format_tool_list,
    _build_system_prompt,
    _RailRegistry,
)
from jiuwensymbiosis.env.mock import MockArmEnv
from jiuwensymbiosis.agent.session import RobotSession

from tests.mocks.mock_api import MockApi


@pytest.fixture
def mock_session():
    env = MockArmEnv()
    api = MockApi(env)
    return RobotSession(env=env, api=api, name="test_mock")


class TestResolveWorkspace:
    def test_explicit_arg(self, mock_session, tmp_path):
        ws = _resolve_workspace(mock_session, str(tmp_path / "my_ws"))
        assert "my_ws" in ws

    def test_env_var(self, mock_session, tmp_path, monkeypatch):
        env_path = str(tmp_path / "env_ws")
        monkeypatch.setenv("JIUWENSYMBIOSIS_WORKSPACE", env_path)
        ws = _resolve_workspace(mock_session, None)
        assert "env_ws" in ws

    def test_default(self, mock_session, monkeypatch):
        monkeypatch.delenv("JIUWENSYMBIOSIS_WORKSPACE", raising=False)
        monkeypatch.delenv("JIUWENSYMBIOSIS_SETTINGS", raising=False)
        ws = _resolve_workspace(mock_session, None)
        assert "test_mock_workspace" in ws


class TestFormatToolList:
    def test_format(self, mock_api):
        result = _format_tool_list(mock_api)
        assert "home" in result
        assert "goto_xyzr" in result


class TestBuildSystemPrompt:
    def test_default_prompt(self, mock_session):
        prompt = _build_system_prompt(mock_session, None)
        assert "test_mock" in prompt or "robot" in prompt

    def test_custom_prompt(self, mock_session):
        prompt = _build_system_prompt(mock_session, "Custom prompt")
        assert prompt == "Custom prompt"


class TestRailRegistry:
    def test_should_enable_visual_feedback(self):
        cfg = _RailRegistry._rails[0]
        assert cfg.required_flags == ["enable_visual_feedback"]
        assert cfg.required_capabilities == ["vision.camera"]

    def test_should_enable_safety(self):
        cfg = _RailRegistry._rails[1]
        assert cfg.required_flags == ["enable_safety"]
        assert cfg.required_capabilities == ["motion.cartesian"]

    def test_should_enable_recovery(self):
        cfg = _RailRegistry._rails[2]
        assert cfg.required_flags == ["enable_recovery"]
        assert cfg.any_capabilities == ["motion.cartesian", "grasp.suction", "grasp.parallel"]

    def test_conditions_met_visual_feedback(self):
        cfg = _RailRegistry._rails[0]
        flags = {"enable_visual_feedback": True, "enable_safety": True, "enable_recovery": True}
        caps = {"vision.camera", "motion.cartesian"}
        assert _RailRegistry._should_enable(flags, caps, cfg) is True

    def test_conditions_not_met_visual_feedback(self):
        cfg = _RailRegistry._rails[0]
        flags = {"enable_visual_feedback": True, "enable_safety": True}
        caps = {"motion.cartesian"}
        assert _RailRegistry._should_enable(flags, caps, cfg) is False

    def test_conditions_any_caps_recovery(self):
        cfg = _RailRegistry._rails[2]
        flags = {"enable_recovery": True, "enable_safety": True}
        caps = {"grasp.parallel"}
        assert _RailRegistry._should_enable(flags, caps, cfg) is True
