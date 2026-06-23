# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.agent.config."""

from __future__ import annotations

from jiuwensymbiosis.agent.config import (
    ModelSpec,
    RailConfig,
    RobotAgentConfig,
    ROBOT_PROMPT_TEMPLATE,
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
