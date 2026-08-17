# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""P2: fast planner is capability-aware (api_capabilities injected into prompts)."""

import inspect

from jiuwensymbiosis.agent.fast import planner


def test_compose_actions_accepts_api_capabilities():
    assert "api_capabilities" in inspect.signature(planner.compose_actions).parameters


def test_compile_sequence_accepts_api_capabilities():
    assert "api_capabilities" in inspect.signature(planner.compile_sequence).parameters


def test_format_capabilities_renders_when_given():
    out = planner._format_capabilities(["motion.base", "grasp.dual_arm"])
    assert "motion.base" in out and "grasp.dual_arm" in out
    assert planner._format_capabilities(None) == ""
    assert planner._format_capabilities([]) == ""
