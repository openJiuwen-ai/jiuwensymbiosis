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


@pytest.mark.parametrize("subagent", [False, True])
@pytest.mark.parametrize("allowed", [frozenset(), frozenset({"home"}), frozenset({"extra"})])
def test_tool_allowlist_excludes_other_robot_tools(mock_session, tmp_path, monkeypatch, subagent, allowed):
    from types import SimpleNamespace

    from jiuwensymbiosis.agent import builder
    from jiuwensymbiosis.agent.config import RobotAgentConfig

    config = RobotAgentConfig(
        model=object(),
        workspace=str(tmp_path),
        execution={"mode": "stepagent", "stepagent": {"mode": "tool"}},
        modules={"skills": {"enabled": False}},
        tool_allowlist=allowed,
        extra_tools=[SimpleNamespace(card=SimpleNamespace(name="extra"))],
    )
    monkeypatch.setattr(builder, "create_deep_agent", lambda **kwargs: SimpleNamespace(**kwargs))
    result = (
        builder.build_robot_agent_config(mock_session, config=config)
        if subagent
        else builder.build_robot_agent(mock_session, config=config)
    )
    assert {tool.card.name for tool in result.tools} == allowed


@pytest.mark.parametrize("supported", [True, False, None])
def test_mobile_constraints_reach_default_prompt(supported):
    from jiuwensymbiosis.adapters.cruzr.api import CruzrApi
    from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
    from jiuwensymbiosis.adapters.cruzr.env import CruzrEnv

    env = CruzrEnv(CruzrConfig())
    env.supports_lateral = supported
    session = RobotSession(env=env, api=CruzrApi(env), name="mobile")
    assert session.describe()["base_motion"] == {"supports_lateral": supported}
    import json

    prompt = _build_system_prompt(session, None, mode="tool")
    assert "Body motion constraints: " + json.dumps({"supports_lateral": supported}) in prompt
