# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Public step-agent builders require a matching execution profile."""

import pytest

from jiuwensymbiosis.agent import builder
from jiuwensymbiosis.agent.config import RobotAgentConfig
from tests.helpers import make_mock_session


@pytest.fixture(params=["build_robot_agent", "build_robot_agent_config"])
def public_builder(request, monkeypatch, tmp_path):
    monkeypatch.setattr(builder, "create_deep_agent", lambda **kwargs: kwargs)
    monkeypatch.setattr(builder, "SubAgentConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(builder, "build_model", lambda _spec: object())
    monkeypatch.setattr(builder, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(builder, "_resolve_workspace", lambda *_args: str(tmp_path))
    return getattr(builder, request.param)


@pytest.mark.parametrize("explicit", [False, True], ids=["omitted-config", "explicit-stepagent"])
def test_stepagent_defaults_are_the_same_with_or_without_explicit_config(public_builder, explicit):
    session = make_mock_session()
    if explicit:
        cfg = RobotAgentConfig(execution={"mode": "stepagent"})
        before = cfg.to_dict()
        built = public_builder(session, config=cfg)
        assert cfg.to_dict() == before
    else:
        built = public_builder(session)

    assert "robot_control" not in [tool.card.name for tool in built["tools"]]
    rail_names = {type(rail).__name__ for rail in built["rails"]}
    assert "SkillUseRail" not in rail_names
    assert "VisualFeedbackRail" in rail_names


@pytest.mark.parametrize("skills", [False, True])
def test_stepagent_preserves_explicit_module_switches(public_builder, skills):
    cfg = RobotAgentConfig(
        execution={"mode": "stepagent"},
        modules={"skills": {"enabled": skills}, "visual_feedback": {"enabled": False}},
    )
    before = cfg.to_dict()
    built = public_builder(make_mock_session(), config=cfg)
    assert cfg.to_dict() == before
    assert ("robot_control" in [tool.card.name for tool in built["tools"]]) is skills
    rail_names = {type(rail).__name__ for rail in built["rails"]}
    assert "VisualFeedbackRail" not in rail_names
    assert ("SkillUseRail" in rail_names) is (skills and public_builder is builder.build_robot_agent)


@pytest.mark.parametrize("explicit_mode", [False, True], ids=["default-config", "explicit-fastagent"])
def test_mode_mismatch_fails_before_build_side_effects(public_builder, monkeypatch, explicit_mode):
    cfg = RobotAgentConfig(execution={"mode": "fastagent"}) if explicit_mode else RobotAgentConfig()
    before = cfg.to_dict()
    session = make_mock_session(strict_capabilities=True)

    def unexpected(*_args, **_kwargs):
        pytest.fail("A mismatched profile must fail before constructing the agent or its dependencies")

    for name in (
        "configure_logging",
        "build_model",
        "_build_tools",
        "_resolve_rails",
        "create_deep_agent",
        "SubAgentConfig",
    ):
        monkeypatch.setattr(builder, name, unexpected)

    with pytest.raises(ValueError, match="agent.execution.mode='stepagent'; got 'fastagent'") as error:
        public_builder(session, config=cfg)
    assert "run_robot_task" in str(error.value)
    assert cfg.to_dict() == before
    assert session.strict_capabilities is True
