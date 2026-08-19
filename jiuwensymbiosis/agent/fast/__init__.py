# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Fast path: plan a task into an action sequence, then run it with no per-step LLM.

    plan_task(query, skills_md=..., action_index=..., world_tokens=...)
        → parse_sequence(raw, allowed_ops, special_ops=...) → run_sequence(session, steps)

``plan_task`` composes registered skills when the library covers the task and
derives a sequence from the action contracts when it does not — one LLM call for
the first, a second only when it declines; see ``planner`` for the three
conditions that route between them.

Add a skill = add a ``SKILL.md`` directory (auto-discovered by the registry) or
``register_skill_dir(path)``. No Python executor per skill.
"""

from jiuwensymbiosis.agent.fast.planner import (
    PlanResult,
    compile_sequence,
    compose_actions,
    plan_task,
)
from jiuwensymbiosis.agent.fast.realtime.mask_tracking import MaskTrackingConfig
from jiuwensymbiosis.agent.fast.registry import (
    DEFAULT_REGISTRY,
    SkillRegistry,
    SkillSpec,
    register_skill,
    register_skill_dir,
)
from jiuwensymbiosis.agent.fast.realtime import ServoConfig, servo_config_from_session
from jiuwensymbiosis.agent.fast.runner import SkillExecConfig, run_sequence
from jiuwensymbiosis.agent.fast.sequence import (
    KNOWN_SPECIAL_OPS,
    TRACK_DETECT,
    TRACK_GRASP,
    ActionStep,
    SequenceError,
    parse_sequence,
)

__all__ = [
    # config
    "SkillExecConfig",
    "ServoConfig",
    "servo_config_from_session",
    "MaskTrackingConfig",
    # pipeline
    "plan_task",
    "PlanResult",
    "parse_sequence",
    "run_sequence",
    "ActionStep",
    "SequenceError",
    "TRACK_DETECT",
    "TRACK_GRASP",
    "KNOWN_SPECIAL_OPS",
    # planning tiers (plan_task drives both; exposed for tests and distillation)
    "compile_sequence",
    "compose_actions",
    # registry
    "SkillSpec",
    "SkillRegistry",
    "DEFAULT_REGISTRY",
    "register_skill",
    "register_skill_dir",
]
