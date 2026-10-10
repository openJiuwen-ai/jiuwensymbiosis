# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Servo settings come from agent configuration, independent of body profiles."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from jiuwensymbiosis.agent.config import RobotAgentConfig
from jiuwensymbiosis.runtime.worker import build_agent_config


@pytest.mark.parametrize("config_root", ["configs", "jiuwensymbiosis_gui/workbench/data/configs"])
@pytest.mark.parametrize("body, expected", [("so101", (20.0, 3.0)), ("piper", (10.0, 5.0))])
def test_shipped_body_servo_tuning_is_explicit_and_preserved_by_runtime(config_root, body, expected, tmp_path):
    root = Path(__file__).resolve().parents[3]
    data = yaml.safe_load((root / config_root / body / f"{body}.yaml").read_text())
    explicit = data["agent"]["execution"]["fastagent"]["tracking"]["servo"]
    assert (explicit["control_hz"], explicit["max_lin_step_mm"]) == expected
    binding = SimpleNamespace(workspace=tmp_path, config_data=lambda: data)
    runtime = build_agent_config(binding, {})
    parsed = RobotAgentConfig.from_dict(data["agent"])
    assert runtime.execution.fastagent.tracking.servo == parsed.execution.fastagent.tracking.servo
    overridden = build_agent_config(binding, {"execution": {"fastagent": {"tracking": {"servo": {"control_hz": 12}}}}})
    assert overridden.execution.fastagent.tracking.servo.control_hz == 12
    assert overridden.execution.fastagent.tracking.servo.max_lin_step_mm == expected[1]


@pytest.mark.parametrize("trajectory_hz", [None, 10.0, 100.0])
def test_body_profile_does_not_override_agent_servo_defaults(trajectory_hz):
    data = {
        "env": {
            "cfg": {"low_level": {"trajectory_hz": trajectory_hz, "motion_runtime": {"max_cartesian_vel_mm_s": 30}}}
        }
    }
    binding = SimpleNamespace(workspace=Path("/tmp/servo-config"), config_data=lambda: data)
    cfg = build_agent_config(binding, {})
    assert cfg.execution.fastagent.tracking.servo == RobotAgentConfig().execution.fastagent.tracking.servo


def test_partial_servo_override_preserves_yaml_tuning():
    data = {"agent": {"execution": {"fastagent": {"tracking": {"servo": {"control_hz": 10, "max_lin_step_mm": 3}}}}}}
    binding = SimpleNamespace(workspace=Path("/tmp/servo-config"), config_data=lambda: data)
    cfg = build_agent_config(binding, {"execution": {"fastagent": {"tracking": {"servo": {"control_hz": 12}}}}})
    assert cfg.execution.fastagent.tracking.servo.control_hz == 12
    assert cfg.execution.fastagent.tracking.servo.max_lin_step_mm == 3
