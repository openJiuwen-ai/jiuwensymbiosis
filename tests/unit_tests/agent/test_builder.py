# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.agent.builder."""

from __future__ import annotations

import pytest

from jiuwensymbiosis.agent.builder import (
    _build_system_prompt,
    _format_tool_list,
    _RailRegistry,
    _resolve_workspace,
)
from jiuwensymbiosis.agent.session import RobotSession
from jiuwensymbiosis.env.mock import MockArmEnv
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
    """build_robot_agent wiring of TraceRail + sinks."""

    def _build(self, mock_session, *, save_frames=False, **cfg_kwargs):
        from jiuwensymbiosis.agent.builder import _resolve_rails
        from jiuwensymbiosis.agent.config import RobotAgentConfig

        cfg = RobotAgentConfig(enable_tracing=True, **cfg_kwargs)
        rails = _resolve_rails(
            mock_session,
            cfg.enable_visual_feedback,
            cfg.enable_safety,
            cfg.enable_recovery,
            cfg.extra_rails,
        )
        # replicate builder sink injection when tracing is on
        from jiuwensymbiosis.agent.trace import TraceRail

        trace_rail = TraceRail(mock_session, workspace="/tmp/trace_test", save_frames=save_frames)
        from jiuwensymbiosis.agent.builder import _inject_trace_sinks

        _inject_trace_sinks(rails, trace_rail)
        return trace_rail, rails

    def test_trace_rail_prepended_when_enabled(self, mock_session, tmp_path):
        from jiuwensymbiosis.agent.builder import build_robot_agent
        from jiuwensymbiosis.agent.config import RobotAgentConfig
        from jiuwensymbiosis.agent.trace import TraceRail

        cfg = RobotAgentConfig(enable_tracing=True, workspace=str(tmp_path))
        build_robot_agent(mock_session, cfg)
        # The agent's rails include a TraceRail; inspect via the session ref.
        assert isinstance(mock_session._trace_rail, TraceRail)
        mock_session.disconnect()

    def test_no_trace_rail_when_disabled(self, mock_session, tmp_path):
        from jiuwensymbiosis.agent.builder import build_robot_agent
        from jiuwensymbiosis.agent.config import RobotAgentConfig

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

    def test_visual_feedback_rail_frame_sink_gated_by_save_frames(self, mock_session):
        """``frame_sink`` is installed only when ``trace_save_frames=True``;
        default (False) leaves it None so disabled frame saving isn't silently
        bypassed."""
        from jiuwensymbiosis.rails.visual_feedback import VisualFeedbackRail

        # default: save_frames=False → no frame_sink
        trace_rail, rails = self._build(mock_session, enable_visual_feedback=True)
        vf = next(r for r in rails if isinstance(r, VisualFeedbackRail))
        assert vf.trace_sink is trace_rail
        assert vf.frame_sink is None

        # save_frames=True → frame_sink installed
        trace_rail, rails = self._build(mock_session, enable_visual_feedback=True, save_frames=True)
        vf = next(r for r in rails if isinstance(r, VisualFeedbackRail))
        assert vf.trace_sink is trace_rail
        assert vf.frame_sink is not None

    def test_public_builder_clears_stale_sinks_when_tracing_disabled(
        self, mock_session, tmp_path, monkeypatch
    ):
        """The public build path reconciles reused rails on tracing-on → off."""
        from jiuwensymbiosis.agent import builder as builder_mod
        from jiuwensymbiosis.agent.config import RobotAgentConfig
        from jiuwensymbiosis.rails.visual_feedback import VisualFeedbackRail

        vf = VisualFeedbackRail(mock_session)
        monkeypatch.setattr(builder_mod, "create_deep_agent", lambda **kwargs: kwargs)
        common = {
            "model": object(),
            "workspace": str(tmp_path),
            "log_dir": None,
            "enable_visual_feedback": False,
            "extra_rails": [vf],
        }

        builder_mod.build_robot_agent(
            mock_session,
            RobotAgentConfig(enable_tracing=True, trace_save_frames=True, **common),
        )
        assert vf.trace_sink is not None
        assert vf.frame_sink is not None
        assert mock_session._trace_rail is vf.trace_sink

        builder_mod.build_robot_agent(
            mock_session,
            RobotAgentConfig(enable_tracing=False, **common),
        )
        assert vf.trace_sink is None
        assert vf.frame_sink is None
        assert mock_session._trace_rail is None

        custom_trace_sink = object()

        def custom_frame_sink(*_args):
            return None

        vf.trace_sink = custom_trace_sink
        vf.frame_sink = custom_frame_sink
        builder_mod.build_robot_agent(
            mock_session,
            RobotAgentConfig(enable_tracing=False, **common),
        )
        assert vf.trace_sink is custom_trace_sink
        assert vf.frame_sink is custom_frame_sink


class TestParallelToolCalls:
    """Robot motion is sequential; parallel_tool_calls defaults OFF and is
    propagated to both ``create_deep_agent`` (single-robot) and
    ``SubAgentConfig`` (multi-robot). ``True`` with motion/grasp caps raises —
    physical devices don't get a foot-gun."""

    def test_default_is_false(self):
        from jiuwensymbiosis.agent.config import RobotAgentConfig

        assert RobotAgentConfig().parallel_tool_calls is False

    def test_builder_defaults_false_to_create_deep_agent(self, mock_session, tmp_path, monkeypatch):
        from jiuwensymbiosis.agent import builder as builder_mod
        from jiuwensymbiosis.agent.config import RobotAgentConfig

        captured: dict = {}

        def _fake_create_deep_agent(*args, **kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(builder_mod, "create_deep_agent", _fake_create_deep_agent)
        cfg = RobotAgentConfig(workspace=str(tmp_path))
        builder_mod.build_robot_agent(mock_session, cfg)
        assert captured.get("parallel_tool_calls") is False

    def test_builder_passes_true_when_vision_only(self, mock_session, tmp_path, monkeypatch):
        """``True`` is allowed for non-motion caps (e.g. vision-only agent)."""
        from jiuwensymbiosis.agent import builder as builder_mod
        from jiuwensymbiosis.agent.config import RobotAgentConfig
        from jiuwensymbiosis.env.mock import MockArmEnv

        captured: dict = {}

        def _fake_create_deep_agent(*args, **kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(builder_mod, "create_deep_agent", _fake_create_deep_agent)
        monkeypatch.setattr(MockArmEnv, "capabilities", frozenset({"vision.camera", "vision.detection"}))
        cfg = RobotAgentConfig(workspace=str(tmp_path), parallel_tool_calls=True)
        builder_mod.build_robot_agent(mock_session, cfg)
        assert captured.get("parallel_tool_calls") is True

    def test_true_with_motion_caps_raises(self, mock_session, tmp_path):
        from jiuwensymbiosis.agent.builder import build_robot_agent
        from jiuwensymbiosis.agent.config import RobotAgentConfig

        cfg = RobotAgentConfig(workspace=str(tmp_path), parallel_tool_calls=True)
        with pytest.raises(ValueError, match="parallel_tool_calls=True is not allowed"):
            build_robot_agent(mock_session, cfg)

    def test_true_with_grasp_caps_raises(self, mock_session, tmp_path, monkeypatch):
        from jiuwensymbiosis.agent.builder import build_robot_agent
        from jiuwensymbiosis.agent.config import RobotAgentConfig
        from jiuwensymbiosis.env.mock import MockArmEnv

        monkeypatch.setattr(MockArmEnv, "capabilities", frozenset({"grasp.parallel"}))
        cfg = RobotAgentConfig(workspace=str(tmp_path), parallel_tool_calls=True)
        with pytest.raises(ValueError, match="grasp"):
            build_robot_agent(mock_session, cfg)

    def test_subagent_config_default_false(self, mock_session):
        from jiuwensymbiosis.agent.builder import build_robot_agent_config

        sac = build_robot_agent_config(mock_session)
        assert sac.parallel_tool_calls is False

    def test_subagent_config_true_passes_through_when_vision_only(self, mock_session, monkeypatch):
        from jiuwensymbiosis.agent.builder import build_robot_agent_config
        from jiuwensymbiosis.agent.config import RobotAgentConfig
        from jiuwensymbiosis.env.mock import MockArmEnv

        monkeypatch.setattr(MockArmEnv, "capabilities", frozenset({"vision.camera"}))
        cfg = RobotAgentConfig(parallel_tool_calls=True)
        sac = build_robot_agent_config(mock_session, config=cfg)
        assert sac.parallel_tool_calls is True

    def test_subagent_config_true_with_motion_raises(self, mock_session):
        from jiuwensymbiosis.agent.builder import build_robot_agent_config
        from jiuwensymbiosis.agent.config import RobotAgentConfig

        cfg = RobotAgentConfig(parallel_tool_calls=True)
        with pytest.raises(ValueError):
            build_robot_agent_config(mock_session, config=cfg)

    def test_both_paths_same_default(self, mock_session, tmp_path, monkeypatch):
        """Single-robot and SubAgent paths share the same default."""
        from jiuwensymbiosis.agent import builder as builder_mod
        from jiuwensymbiosis.agent.builder import build_robot_agent_config
        from jiuwensymbiosis.agent.config import RobotAgentConfig

        captured: dict = {}

        def _fake_create_deep_agent(*args, **kwargs):
            captured["single"] = kwargs.get("parallel_tool_calls")
            return object()

        monkeypatch.setattr(builder_mod, "create_deep_agent", _fake_create_deep_agent)
        builder_mod.build_robot_agent(mock_session, RobotAgentConfig(workspace=str(tmp_path)))
        sac = build_robot_agent_config(mock_session)
        assert captured["single"] == sac.parallel_tool_calls is False

    def test_parallel_with_tracing_raises_vision_only(self, mock_session, tmp_path, monkeypatch):
        """TraceRail keys current step via shared ctx.extra / entries[-1],
        which races under parallel dispatch — so tracing + parallel is rejected
        even for vision-only agents."""
        from jiuwensymbiosis.agent.builder import build_robot_agent
        from jiuwensymbiosis.agent.config import RobotAgentConfig
        from jiuwensymbiosis.env.mock import MockArmEnv

        monkeypatch.setattr(MockArmEnv, "capabilities", frozenset({"vision.camera"}))
        cfg = RobotAgentConfig(workspace=str(tmp_path), parallel_tool_calls=True, enable_tracing=True)
        with pytest.raises(ValueError, match="enable_tracing=True is not supported"):
            build_robot_agent(mock_session, cfg)

    def test_parallel_without_tracing_vision_only_ok(self, mock_session, tmp_path, monkeypatch):
        from jiuwensymbiosis.agent import builder as builder_mod
        from jiuwensymbiosis.agent.config import RobotAgentConfig
        from jiuwensymbiosis.env.mock import MockArmEnv

        captured: dict = {}

        def _fake_create_deep_agent(*args, **kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(builder_mod, "create_deep_agent", _fake_create_deep_agent)
        monkeypatch.setattr(MockArmEnv, "capabilities", frozenset({"vision.camera"}))
        cfg = RobotAgentConfig(workspace=str(tmp_path), parallel_tool_calls=True, enable_tracing=False)
        builder_mod.build_robot_agent(mock_session, cfg)
        assert captured.get("parallel_tool_calls") is True

    def test_parallel_with_tracing_raises_subagent(self, mock_session, monkeypatch):
        from jiuwensymbiosis.agent.builder import build_robot_agent_config
        from jiuwensymbiosis.agent.config import RobotAgentConfig
        from jiuwensymbiosis.env.mock import MockArmEnv

        monkeypatch.setattr(MockArmEnv, "capabilities", frozenset({"vision.camera"}))
        cfg = RobotAgentConfig(parallel_tool_calls=True, enable_tracing=True)
        with pytest.raises(ValueError, match="enable_tracing=True is not supported"):
            build_robot_agent_config(mock_session, config=cfg)
