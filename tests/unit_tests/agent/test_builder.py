# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.agent.builder."""

from __future__ import annotations

import pytest

from jiuwensymbiosis.agent.builder import (
    _build_system_prompt,
    _format_tool_list,
    _resolve_workspace,
)
from jiuwensymbiosis.agent.session import RobotSession
from jiuwensymbiosis.env.mock import MockArmEnv
from tests.helpers import make_mock_session
from tests.mocks.mock_api import MockApi


@pytest.fixture
def mock_session():
    return make_mock_session(name="test_mock")


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

    def test_custom_prompt_skips_globals(self, mock_session):
        # A fully custom prompt is used verbatim — globals are not appended.
        mock_session.extra_globals = {"my_helper": object()}
        prompt = _build_system_prompt(mock_session, "Custom", mode="code")
        assert prompt == "Custom"

    def test_code_mode_lists_globals(self):
        env = MockArmEnv()
        api = MockApi(env)
        s = RobotSession(env=env, api=api, name="t", extra_globals={"my_helper": object()})
        prompt = _build_system_prompt(s, None, mode="code")
        # Built-in globals always present.
        for name in ("env", "api", "np"):
            assert name in prompt
        # extra_globals surfaced automatically.
        assert "my_helper" in prompt

    def test_tool_mode_omits_globals(self):
        env = MockArmEnv()
        api = MockApi(env)
        s = RobotSession(env=env, api=api, name="t", extra_globals={"my_helper": object()})
        prompt = _build_system_prompt(s, None, mode="tool")
        assert "my_helper" not in prompt

    def test_default_mode_is_hybrid_lists_globals(self):
        env = MockArmEnv()
        api = MockApi(env)
        s = RobotSession(env=env, api=api, name="t", extra_globals={"my_helper": object()})
        # Default mode (hybrid) includes the inproc code tool → globals listed.
        prompt = _build_system_prompt(s, None)
        assert "my_helper" in prompt
