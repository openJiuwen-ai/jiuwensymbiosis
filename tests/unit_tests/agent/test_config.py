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
            required_flags=["safety"],
        )
        assert rc.rail_class_path == "jiuwensymbiosis.rails.safety.SafetyRail"
        assert rc.required_flags == ["safety"]

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
        assert cfg.execution.stepagent.mode == "hybrid"
        assert cfg.modules.visual_feedback.enabled is False
        assert cfg.modules.safety.enabled is True
        assert cfg.modules.recovery.enabled is True
        assert cfg.modules.skills.enabled is True
        assert cfg.execution.stepagent.max_iterations == 15

    def test_mode_literal(self):
        for mode in ("tool", "code", "hybrid"):
            cfg = RobotAgentConfig(execution={"stepagent": {"mode": mode}})
            assert cfg.execution.stepagent.mode == mode

    def test_strict_capabilities_default_false(self):
        cfg = RobotAgentConfig()
        assert cfg.strict_capabilities is False

    def test_strict_capabilities_settable(self):
        cfg = RobotAgentConfig(strict_capabilities=True)
        assert cfg.strict_capabilities is True

    def test_fast_special_ops_default_on(self):
        assert RobotAgentConfig().execution.fastagent.tracking.enabled is True

    def test_fast_special_ops_settable(self):
        assert (
            RobotAgentConfig(
                execution={"fastagent": {"tracking": {"enabled": False}}}
            ).execution.fastagent.tracking.enabled
            is False
        )

    def test_fast_special_ops_from_dict(self):
        assert (
            RobotAgentConfig.from_dict(
                {"execution": {"fastagent": {"tracking": {"enabled": False}}}}
            ).execution.fastagent.tracking.enabled
            is False
        )


class TestPromptTemplate:
    def test_contains_robot_name_placeholder(self):
        assert "{robot_name}" in ROBOT_PROMPT_TEMPLATE


class TestTracingAndLoggingConfig:
    def test_tracing_defaults_off(self):
        cfg = RobotAgentConfig()
        assert cfg.modules.tracing.enabled is False
        assert cfg.modules.tracing.max_entries == 200
        assert cfg.modules.tracing.max_frames == 50
        assert cfg.modules.tracing.save_frames is False
        assert cfg.modules.tracing.console is False
        assert cfg.modules.tracing.dir is None

    def test_trace_capture_loggers_default(self):
        cfg = RobotAgentConfig()
        assert cfg.modules.tracing.capture_loggers == ["jiuwensymbiosis"]

    def test_logging_defaults(self):
        cfg = RobotAgentConfig()
        assert cfg.logging.level == "INFO"
        assert cfg.logging.dir == "./logs"

    def test_tracing_fields_settable(self):
        cfg = RobotAgentConfig(
            modules={
                "tracing": {"enabled": True, "max_entries": 10, "save_frames": True, "console": True, "dir": "/tmp/x"}
            },
            logging={"level": "DEBUG", "dir": "/tmp/logs"},
        )
        assert cfg.modules.tracing.enabled is True
        assert cfg.modules.tracing.max_entries == 10
        assert cfg.modules.tracing.save_frames is True
        assert cfg.logging.level == "DEBUG"
        assert cfg.logging.dir == "/tmp/logs"

    def test_trace_capture_loggers_independent_default(self):
        # mutable default must not be shared across instances
        a = RobotAgentConfig()
        b = RobotAgentConfig()
        a.modules.tracing.capture_loggers.append("custom")
        assert b.modules.tracing.capture_loggers == ["jiuwensymbiosis"]


class TestRobotAgentConfigFromDict:
    """YAML ``agent:`` block → RobotAgentConfig.from_dict (mirrors ModelSpec/PiperConfig)."""

    def test_empty_or_none_returns_defaults(self):
        assert RobotAgentConfig.from_dict(None).modules.tracing.enabled is False
        assert RobotAgentConfig.from_dict({}).modules.tracing.enabled is False

    def test_applies_trace_and_logging_keys(self):
        cfg = RobotAgentConfig.from_dict(
            {
                "modules": {
                    "tracing": {
                        "enabled": True,
                        "save_frames": True,
                        "console": True,
                        "max_entries": 42,
                        "max_frames": 7,
                    }
                },
                "logging": {"level": "DEBUG", "dir": "/tmp/logs"},
            }
        )
        assert cfg.modules.tracing.enabled is True
        assert cfg.modules.tracing.save_frames is True
        assert cfg.modules.tracing.console is True
        assert cfg.modules.tracing.max_entries == 42
        assert cfg.modules.tracing.max_frames == 7
        assert cfg.logging.level == "DEBUG"
        assert cfg.logging.dir == "/tmp/logs"

    def test_python_model_dependencies_are_not_yaml_settings(self):
        import pytest

        for key in ("model", "model_spec"):
            with pytest.raises(TypeError, match="Python-only"):
                RobotAgentConfig.from_dict({key: {"model_name": "x"}})

    def test_unknown_key_raises_typeerror(self):
        # Catches unknown YAML fields at
        # load time instead of silently ignoring them.
        import pytest

        with pytest.raises(TypeError):
            RobotAgentConfig.from_dict({"enable_trace": True})
