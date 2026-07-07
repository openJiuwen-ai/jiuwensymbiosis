# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for parallel_tool_calls safeguards in builder paths."""

from __future__ import annotations

import pytest

from tests.helpers import make_mock_session


@pytest.fixture
def mock_session():
    return make_mock_session(name="test_mock")


def _patch_create_deep_agent(monkeypatch):
    from jiuwensymbiosis.agent import builder as builder_mod

    captured: dict = {}

    def _fake_create_deep_agent(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(builder_mod, "create_deep_agent", _fake_create_deep_agent)
    return builder_mod, captured


def _set_env_caps(monkeypatch, caps):
    from jiuwensymbiosis.env.mock import MockArmEnv

    monkeypatch.setattr(MockArmEnv, "capabilities", frozenset(caps))


class TestParallelToolCalls:
    """Parallel tool calls are blocked for robot motion/grasp and tracing."""

    def test_default_is_false(self):
        from jiuwensymbiosis.agent.config import RobotAgentConfig

        assert RobotAgentConfig().parallel_tool_calls is False

    def test_builder_defaults_false_to_create_deep_agent(self, mock_session, tmp_path, monkeypatch):
        from jiuwensymbiosis.agent.config import RobotAgentConfig

        builder_mod, captured = _patch_create_deep_agent(monkeypatch)
        builder_mod.build_robot_agent(mock_session, RobotAgentConfig(workspace=str(tmp_path)))
        assert captured.get("parallel_tool_calls") is False

    @pytest.mark.parametrize(
        ("caps", "match"),
        [
            ({"motion.cartesian"}, "parallel_tool_calls=True is not allowed"),
            ({"grasp.parallel"}, "grasp"),
        ],
        ids=["motion", "grasp"],
    )
    def test_true_with_robot_caps_raises(self, mock_session, tmp_path, monkeypatch, caps, match):
        from jiuwensymbiosis.agent.builder import build_robot_agent
        from jiuwensymbiosis.agent.config import RobotAgentConfig

        _set_env_caps(monkeypatch, caps)
        cfg = RobotAgentConfig(workspace=str(tmp_path), parallel_tool_calls=True)
        with pytest.raises(ValueError, match=match):
            build_robot_agent(mock_session, cfg)

    def test_builder_passes_true_when_vision_only_without_tracing(self, mock_session, tmp_path, monkeypatch):
        from jiuwensymbiosis.agent.config import RobotAgentConfig

        builder_mod, captured = _patch_create_deep_agent(monkeypatch)
        _set_env_caps(monkeypatch, {"vision.camera", "vision.detection"})
        cfg = RobotAgentConfig(workspace=str(tmp_path), parallel_tool_calls=True, enable_tracing=False)
        builder_mod.build_robot_agent(mock_session, cfg)
        assert captured.get("parallel_tool_calls") is True

    def test_subagent_config_default_false(self, mock_session):
        from jiuwensymbiosis.agent.builder import build_robot_agent_config

        sac = build_robot_agent_config(mock_session)
        assert sac.parallel_tool_calls is False

    def test_subagent_config_true_passes_through_when_vision_only(self, mock_session, monkeypatch):
        from jiuwensymbiosis.agent.builder import build_robot_agent_config
        from jiuwensymbiosis.agent.config import RobotAgentConfig

        _set_env_caps(monkeypatch, {"vision.camera"})
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
        from jiuwensymbiosis.agent.builder import build_robot_agent_config
        from jiuwensymbiosis.agent.config import RobotAgentConfig

        builder_mod, captured = _patch_create_deep_agent(monkeypatch)
        builder_mod.build_robot_agent(mock_session, RobotAgentConfig(workspace=str(tmp_path)))
        sac = build_robot_agent_config(mock_session)
        assert captured["parallel_tool_calls"] is False
        assert sac.parallel_tool_calls is False

    @pytest.mark.parametrize("path", ["single", "subagent"])
    def test_parallel_with_tracing_raises_vision_only(self, mock_session, tmp_path, monkeypatch, path):
        """TraceRail keys current step via shared ctx.extra / entries[-1], which races under parallel dispatch."""
        from jiuwensymbiosis.agent.builder import build_robot_agent, build_robot_agent_config
        from jiuwensymbiosis.agent.config import RobotAgentConfig

        _set_env_caps(monkeypatch, {"vision.camera"})
        cfg = RobotAgentConfig(workspace=str(tmp_path), parallel_tool_calls=True, enable_tracing=True)
        with pytest.raises(ValueError, match="enable_tracing=True is not supported"):
            if path == "single":
                build_robot_agent(mock_session, cfg)
            else:
                build_robot_agent_config(mock_session, config=cfg)
