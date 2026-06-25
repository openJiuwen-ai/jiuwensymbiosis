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


class TestTracingBuild:
    """build_robot_agent wiring of TraceRail + sinks (Issue #9)."""

    def _build(self, mock_session, **cfg_kwargs):
        from jiuwensymbiosis.agent.builder import _resolve_rails
        from jiuwensymbiosis.agent.config import RobotAgentConfig

        cfg = RobotAgentConfig(enable_tracing=True, **cfg_kwargs)
        rails = _resolve_rails(
            mock_session, cfg.enable_visual_feedback, cfg.enable_safety,
            cfg.enable_recovery, cfg.extra_rails,
        )
        # replicate builder sink injection when tracing is on
        from jiuwensymbiosis.agent.trace import TraceRail

        trace_rail = TraceRail(mock_session, workspace="/tmp/trace_test")
        from jiuwensymbiosis.agent.builder import _inject_trace_sinks

        _inject_trace_sinks(rails, trace_rail)
        return trace_rail, rails

    def test_trace_rail_prepended_when_enabled(self, mock_session, tmp_path):
        from jiuwensymbiosis.agent.config import RobotAgentConfig
        from jiuwensymbiosis.agent.builder import build_robot_agent
        from jiuwensymbiosis.agent.trace import TraceRail

        cfg = RobotAgentConfig(enable_tracing=True, workspace=str(tmp_path))
        agent = build_robot_agent(mock_session, cfg)
        # The agent's rails include a TraceRail; inspect via the session ref.
        assert isinstance(mock_session._trace_rail, TraceRail)
        mock_session.disconnect()

    def test_no_trace_rail_when_disabled(self, mock_session, tmp_path):
        from jiuwensymbiosis.agent.config import RobotAgentConfig
        from jiuwensymbiosis.agent.builder import build_robot_agent

        cfg = RobotAgentConfig(enable_tracing=False, workspace=str(tmp_path))
        build_robot_agent(mock_session, cfg)
        assert mock_session._trace_rail is None

    def test_safety_rail_gets_trace_sink(self, mock_session):
        from jiuwensymbiosis.rails.safety import SafetyRail

        trace_rail, rails = self._build(mock_session, enable_safety=True)
        safety = next(r for r in rails if isinstance(r, SafetyRail))
        assert safety.trace_sink is trace_rail

    def test_recovery_rail_gets_trace_sink(self, mock_session):
        from jiuwensymbiosis.rails.recovery import RecoveryRail

        trace_rail, rails = self._build(mock_session, enable_recovery=True)
        recovery = next(r for r in rails if isinstance(r, RecoveryRail))
        assert recovery.trace_sink is trace_rail

    def test_visual_feedback_rail_gets_frame_sink(self, mock_session):
        from jiuwensymbiosis.rails.visual_feedback import VisualFeedbackRail

        # mock env has vision.camera → visual feedback enabled
        trace_rail, rails = self._build(mock_session, enable_visual_feedback=True)
        vf = next(r for r in rails if isinstance(r, VisualFeedbackRail))
        assert vf.trace_sink is trace_rail
        assert vf.frame_sink is not None

