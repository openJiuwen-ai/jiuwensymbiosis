# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for build_robot_agent rail resolution and trace wiring."""

from __future__ import annotations

import pytest

from jiuwensymbiosis.agent.builder import _RailRegistry
from tests.helpers import make_mock_session


@pytest.fixture
def mock_session():
    return make_mock_session(name="test_mock")


class TestRailRegistry:
    def test_declared_conditions(self):
        visual_feedback, safety, recovery = _RailRegistry._rails[:3]
        assert visual_feedback.required_flags == ["enable_visual_feedback"]
        assert visual_feedback.required_capabilities == ["vision.camera"]
        assert safety.required_flags == ["enable_safety"]
        assert safety.required_capabilities == ["motion.cartesian"]
        assert recovery.required_flags == ["enable_recovery"]
        assert recovery.any_capabilities == ["motion.cartesian", "grasp.suction", "grasp.parallel"]

    @pytest.mark.parametrize(
        ("rail_index", "flags", "caps", "expected"),
        [
            (0, {"enable_visual_feedback": True, "enable_safety": True}, {"vision.camera", "motion.cartesian"}, True),
            (0, {"enable_visual_feedback": True, "enable_safety": True}, {"motion.cartesian"}, False),
            (2, {"enable_recovery": True, "enable_safety": True}, {"grasp.parallel"}, True),
        ],
        ids=["visual-feedback-enabled", "visual-feedback-missing-camera", "recovery-any-cap"],
    )
    def test_should_enable_conditions(self, rail_index, flags, caps, expected):
        cfg = _RailRegistry._rails[rail_index]
        assert _RailRegistry._should_enable(flags, caps, cfg) is expected


class TestTracingBuild:
    """build_robot_agent wiring of TraceRail + sinks."""

    def _build(self, mock_session, *, save_frames=False, **cfg_kwargs):
        from jiuwensymbiosis.agent.builder import _inject_trace_sinks, _resolve_rails
        from jiuwensymbiosis.agent.config import RobotAgentConfig
        from jiuwensymbiosis.agent.trace import TraceRail

        cfg = RobotAgentConfig(enable_tracing=True, **cfg_kwargs)
        rails = _resolve_rails(
            mock_session,
            cfg.enable_visual_feedback,
            cfg.enable_safety,
            cfg.enable_recovery,
            cfg.extra_rails,
        )
        trace_rail = TraceRail(mock_session, workspace="/tmp/trace_test", save_frames=save_frames)
        _inject_trace_sinks(rails, trace_rail)
        return trace_rail, rails

    def test_trace_rail_prepended_when_enabled(self, mock_session, tmp_path):
        from jiuwensymbiosis.agent.builder import build_robot_agent
        from jiuwensymbiosis.agent.config import RobotAgentConfig
        from jiuwensymbiosis.agent.trace import TraceRail

        cfg = RobotAgentConfig(enable_tracing=True, workspace=str(tmp_path))
        build_robot_agent(mock_session, cfg)
        assert isinstance(mock_session._trace_rail, TraceRail)
        mock_session.disconnect()

    def test_no_trace_rail_when_disabled(self, mock_session, tmp_path):
        from jiuwensymbiosis.agent.builder import build_robot_agent
        from jiuwensymbiosis.agent.config import RobotAgentConfig

        cfg = RobotAgentConfig(enable_tracing=False, workspace=str(tmp_path))
        build_robot_agent(mock_session, cfg)
        assert mock_session._trace_rail is None

    @pytest.mark.parametrize(
        ("rail_cls_path", "flag"),
        [
            ("jiuwensymbiosis.rails.safety.SafetyRail", "enable_safety"),
            ("jiuwensymbiosis.rails.recovery.RecoveryRail", "enable_recovery"),
        ],
        ids=["safety", "recovery"],
    )
    def test_motion_rails_get_trace_sink(self, mock_session, rail_cls_path, flag):
        import importlib

        module_name, cls_name = rail_cls_path.rsplit(".", 1)
        rail_cls = getattr(importlib.import_module(module_name), cls_name)
        trace_rail, rails = self._build(mock_session, **{flag: True})
        rail = next(r for r in rails if isinstance(r, rail_cls))
        assert rail.trace_sink is trace_rail

    def test_visual_feedback_rail_frame_sink_gated_by_save_frames(self, mock_session):
        """``frame_sink`` is installed only when ``trace_save_frames=True``."""
        from jiuwensymbiosis.rails.visual_feedback import VisualFeedbackRail

        trace_rail, rails = self._build(mock_session, enable_visual_feedback=True)
        vf = next(r for r in rails if isinstance(r, VisualFeedbackRail))
        assert vf.trace_sink is trace_rail
        assert vf.frame_sink is None

        trace_rail, rails = self._build(mock_session, enable_visual_feedback=True, save_frames=True)
        vf = next(r for r in rails if isinstance(r, VisualFeedbackRail))
        assert vf.trace_sink is trace_rail
        assert vf.frame_sink is not None

    def test_public_builder_clears_stale_sinks_when_tracing_disabled(self, mock_session, tmp_path, monkeypatch):
        """The public build path reconciles reused rails on tracing-on -> off."""
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
