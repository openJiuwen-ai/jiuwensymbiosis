# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""C1 fast path: one LLM call compiles SKILL.md → an action sequence; a generic
runner executes it with no per-step LLM + real-time tracking.

Single source of truth = each skill's ``SKILL.md`` (the same file the agent
reads). Pipeline:

    compile_sequence(query, skills_md, action_vocab, special_ops=...)  # 1 LLM call
        → parse_sequence(raw, allowed_ops, special_ops=...) → run_sequence(session, steps)

Add a skill = add a ``SKILL.md`` directory (auto-discovered by the registry) or
``register_skill_dir(path)``. No Python executor per skill.
"""

from jiuwensymbiosis.agent.fast.planner import compile_sequence, plan_skills
from jiuwensymbiosis.agent.fast.registry import (
    DEFAULT_REGISTRY,
    SkillRegistry,
    SkillSpec,
    register_skill,
    register_skill_dir,
)
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
    # pipeline
    "compile_sequence",
    "parse_sequence",
    "run_sequence",
    "ActionStep",
    "SequenceError",
    "TRACK_DETECT",
    "TRACK_GRASP",
    "KNOWN_SPECIAL_OPS",
    # legacy skill planner (skill-name selection only)
    "plan_skills",
    # registry
    "SkillSpec",
    "SkillRegistry",
    "DEFAULT_REGISTRY",
    "register_skill",
    "register_skill_dir",
]
