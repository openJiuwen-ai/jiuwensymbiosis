# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""One agent schema across Python, YAML and runtime entry points."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from jiuwensymbiosis.agent.config import ModulesConfig, RobotAgentConfig
from jiuwensymbiosis.agent.execution_config import ServoConfig, TrackingConfig


def test_grouped_settings_round_trip_without_sharing_mutable_state():
    cfg = RobotAgentConfig(modules={"skills": {"enabled": False}, "tracing": {"max_frames": 7}})
    cfg.modules.tracing.max_frames = 9
    clone = RobotAgentConfig.from_dict(yaml.safe_load(yaml.safe_dump(cfg.to_dict())))
    assert clone.to_dict() == cfg.to_dict()
    clone.modules.skills.enabled = True
    assert not cfg.modules.skills.enabled
    assert copy.deepcopy(cfg).to_dict() == cfg.to_dict()


def test_tracking_has_one_type_and_tuning_does_not_change_enabled():
    cfg = RobotAgentConfig.from_dict(
        {"execution": {"fastagent": {"tracking": {"enabled": False, "servo": {"control_hz": 8}}}}}
    )
    tracking = cfg.execution.fastagent.tracking
    assert type(tracking) is TrackingConfig
    assert isinstance(tracking.servo, ServoConfig)
    tracking.servo.control_hz = 9
    cfg.validate()
    assert cfg.execution.fastagent.tracking.servo.control_hz == 9
    assert not cfg.execution.fastagent.tracking.enabled


def test_defaults_depend_on_selected_execution_not_entry_point():
    fast = RobotAgentConfig.from_dict({})
    assert RobotAgentConfig(modules=ModulesConfig()).to_dict() == fast.to_dict()
    step = RobotAgentConfig.from_dict({}, overrides={"execution": {"mode": "stepagent"}})
    assert fast.modules.skills.enabled and not fast.modules.visual_feedback.enabled
    assert not step.modules.skills.enabled and step.modules.visual_feedback.enabled
    fast.validate(for_execution=True)
    step.validate(for_execution=True)


def test_partial_overrides_preserve_other_settings():
    base = {
        "execution": {"stepagent": {"mode": "tool", "max_iterations": 8}},
        "modules": {"tracing": {"enabled": True, "max_frames": 12}},
    }
    original = copy.deepcopy(base)
    cfg = RobotAgentConfig.from_dict(
        base,
        overrides={"execution": {"stepagent": {"max_iterations": 3}}, "modules": {"tracing": {"console": True}}},
    )
    assert cfg.execution.stepagent.mode == "tool" and cfg.execution.stepagent.max_iterations == 3
    assert cfg.modules.tracing.max_frames == 12 and cfg.modules.tracing.enabled and cfg.modules.tracing.console
    assert base == original


@pytest.mark.parametrize("old", ["exec_config", "exec_mode", "enable_skill", "trace_console", "log_dir", "mode"])
def test_removed_fields_rejected_in_all_config_inputs(old):
    with pytest.raises(TypeError, match=old):
        RobotAgentConfig.from_dict({old: None})
    with pytest.raises(TypeError, match=old):
        RobotAgentConfig.from_dict({}, overrides={old: None})
    with pytest.raises(TypeError, match=old):
        RobotAgentConfig(**{old: None})
    cfg = RobotAgentConfig()
    with pytest.raises(AttributeError):
        setattr(cfg, old, None)
    with pytest.raises(AttributeError):
        getattr(cfg, old)


@pytest.mark.parametrize(
    "data",
    [
        {"execution": {"mode": "typo"}},
        {"modules": {"safety": {"enabled": "false"}}},
        {"execution": {"stepagent": {"max_iterations": -1}}},
        {"modules": {"skills": {"enabld": True}}},
        {"modules": {"spatial_memory": {"enabled": True}}},
        {"execution": {"fastagent": {"max_replans": True}}},
        {"execution": {"fastagent": {"tracking": {"detect_hz": float("nan")}}}},
        {"extra_rails": []},
        {"model": {}},
        {"model_spec": {}},
        {"modules": []},
    ],
)
def test_invalid_configuration_fails_before_execution(data):
    with pytest.raises((TypeError, ValueError)):
        RobotAgentConfig.from_dict(data)


def test_module_dependencies_and_execution_support():
    with pytest.raises(ValueError, match="requires"):
        RobotAgentConfig(modules={"diagnosis": {"enabled": True}})
    for module in ("visual_feedback", "diagnosis"):
        cfg = RobotAgentConfig.from_dict({"modules": {module: {"enabled": True}, "tracing": {"enabled": True}}})
        with pytest.raises(ValueError, match="only supported by stepagent"):
            cfg.validate(for_execution=True)
        cfg.execution.mode = "stepagent"
        cfg.validate(for_execution=True)


@pytest.mark.parametrize("skills", [False, True])
def test_fast_builder_keeps_dispatcher_without_loading_skill_documents(monkeypatch, tmp_path, skills):
    from jiuwensymbiosis.agent import builder
    from tests.helpers import make_mock_session

    monkeypatch.setattr(builder, "create_deep_agent", lambda **kw: kw)
    cfg = RobotAgentConfig(
        modules={"skills": {"enabled": skills}}, model=object(), workspace=str(tmp_path), logging={"dir": None}
    )
    fast = builder._build_fast_agent(make_mock_session(), cfg)
    assert "robot_control" in [tool.card.name for tool in fast["tools"]]
    assert not any(type(rail).__name__ == "SkillUseRail" for rail in fast["rails"])
    assert cfg.execution.mode == "fastagent"


def test_runtime_uses_same_defaults_and_deep_override():
    from jiuwensymbiosis.runtime.worker import build_agent_config

    binding = SimpleNamespace(
        workspace=Path("/tmp/config-contract"),
        config_data=lambda: {"agent": {"modules": {"tracing": {"max_frames": 4}}}},
    )
    cfg = build_agent_config(binding, {"modules": {"tracing": {"enabled": True}}})
    assert cfg.modules.tracing.max_frames == 4 and cfg.modules.tracing.enabled
    assert cfg.execution.fastagent.tracking.servo == RobotAgentConfig().execution.fastagent.tracking.servo


def test_prepare_session_applies_connection_settings():
    cfg = RobotAgentConfig(strict_capabilities=True, logging={"motion_dir": "/tmp/motion-test"})
    session = SimpleNamespace()
    cfg.prepare_session(session)
    assert session.strict_capabilities is True
    assert session.motion_log_dir == "/tmp/motion-test"
