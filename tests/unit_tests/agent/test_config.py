# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.agent.config."""

from __future__ import annotations

from jiuwensymbiosis.agent.config import (
    ROBOT_PROMPT_TEMPLATE,
    ModelSpec,
    RailConfig,
    RobotAgentConfig,
    build_model,
)


class TestModelSpec:
    def test_defaults(self):
        spec = ModelSpec()
        assert spec.provider == "OpenAI"
        assert spec.model_name != ""
        assert spec.temperature == 0.3
        assert spec.max_tokens == 2048

    def test_custom(self):
        spec = ModelSpec(provider="TestProvider", api_base="http://localhost:1234")
        assert spec.provider == "TestProvider"
        assert spec.api_base == "http://localhost:1234"


class TestBuildModel:
    def test_returns_model(self):
        from jiuwensymbiosis.agent.abstractions import Model

        spec = ModelSpec(provider="OpenAI", api_base="http://127.0.0.1:8110/v1")
        m = build_model(spec)
        assert isinstance(m, Model)

    def test_default_spec(self):
        from jiuwensymbiosis.agent.abstractions import Model

        m = build_model()
        assert isinstance(m, Model)


class TestRailConfig:
    def test_basic(self):
        rc = RailConfig(
            rail_class_path="jiuwensymbiosis.rails.safety.SafetyRail",
            required_flags=["enable_safety"],
        )
        assert rc.rail_class_path == "jiuwensymbiosis.rails.safety.SafetyRail"
        assert rc.required_flags == ["enable_safety"]

    def test_empty_capabilities_normalized(self):
        rc = RailConfig(
            rail_class_path="x.Y",
            required_flags=[],
            required_capabilities=[],
            any_capabilities=[],
        )
        assert rc.required_capabilities is None
        assert rc.any_capabilities is None


class TestRobotAgentConfig:
    def test_defaults(self):
        cfg = RobotAgentConfig()
        assert cfg.mode == "hybrid"
        assert cfg.enable_visual_feedback is True
        assert cfg.enable_safety is True
        assert cfg.enable_recovery is True
        assert cfg.enable_skill is False
        assert cfg.max_iterations == 15

    def test_mode_literal(self):
        for mode in ("tool", "code", "hybrid"):
            cfg = RobotAgentConfig(mode=mode)
            assert cfg.mode == mode

    def test_strict_capabilities_default_false(self):
        cfg = RobotAgentConfig()
        assert cfg.strict_capabilities is False

    def test_strict_capabilities_settable(self):
        cfg = RobotAgentConfig(strict_capabilities=True)
        assert cfg.strict_capabilities is True


class TestPromptTemplate:
    def test_contains_robot_name_placeholder(self):
        assert "{robot_name}" in ROBOT_PROMPT_TEMPLATE


class TestTracingAndLoggingConfig:
    def test_tracing_defaults_off(self):
        cfg = RobotAgentConfig()
        assert cfg.enable_tracing is False
        assert cfg.trace_max_entries == 200
        assert cfg.trace_max_frames == 50
        assert cfg.trace_save_frames is False
        assert cfg.trace_console is False
        assert cfg.trace_dir is None

    def test_trace_capture_loggers_default(self):
        cfg = RobotAgentConfig()
        assert cfg.trace_capture_loggers == ["jiuwensymbiosis"]

    def test_logging_defaults(self):
        cfg = RobotAgentConfig()
        assert cfg.log_level == "INFO"
        assert cfg.log_dir == "./logs"

    def test_tracing_fields_settable(self):
        cfg = RobotAgentConfig(
            enable_tracing=True,
            trace_max_entries=10,
            trace_save_frames=True,
            trace_console=True,
            trace_dir="/tmp/x",
            log_level="DEBUG",
            log_dir="/tmp/logs",
        )
        assert cfg.enable_tracing is True
        assert cfg.trace_max_entries == 10
        assert cfg.trace_save_frames is True
        assert cfg.log_level == "DEBUG"
        assert cfg.log_dir == "/tmp/logs"

    def test_trace_capture_loggers_independent_default(self):
        # mutable default must not be shared across instances
        a = RobotAgentConfig()
        b = RobotAgentConfig()
        a.trace_capture_loggers.append("custom")
        assert b.trace_capture_loggers == ["jiuwensymbiosis"]


class TestRobotAgentConfigFromDict:
    """YAML ``agent:`` block → RobotAgentConfig.from_dict (mirrors ModelSpec/PiperConfig)."""

    def test_empty_or_none_returns_defaults(self):
        assert RobotAgentConfig.from_dict(None).enable_tracing is False
        assert RobotAgentConfig.from_dict({}).enable_tracing is False

    def test_applies_trace_and_logging_keys(self):
        cfg = RobotAgentConfig.from_dict(
            {
                "enable_tracing": True,
                "trace_save_frames": True,
                "trace_console": True,
                "trace_max_entries": 42,
                "trace_max_frames": 7,
                "log_level": "DEBUG",
                "log_dir": "/tmp/logs",
            }
        )
        assert cfg.enable_tracing is True
        assert cfg.trace_save_frames is True
        assert cfg.trace_console is True
        assert cfg.trace_max_entries == 42
        assert cfg.trace_max_frames == 7
        assert cfg.log_level == "DEBUG"
        assert cfg.log_dir == "/tmp/logs"

    def test_pops_model_keys(self):
        # model / model_spec are owned by the separate ``model:`` YAML block;
        # from_dict must drop them so a stray YAML entry doesn't reach __init__
        # (where ``model`` expects a built instance, not a dict).
        cfg = RobotAgentConfig.from_dict({"model": {"model_name": "x"}, "model_spec": {"model_name": "x"}})
        assert cfg.model is None
        assert cfg.model_spec is None

    def test_unknown_key_raises_typeerror(self):
        # Catches YAML typos (e.g. ``enable_trace`` vs ``enable_tracing``) at
        # load time instead of silently ignoring them.
        import pytest

        with pytest.raises(TypeError):
            RobotAgentConfig.from_dict({"enable_trace": True})
