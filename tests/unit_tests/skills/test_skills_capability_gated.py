# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""visual_pick / visual_place are capability-branched (gripper 支 A / dual-arm 支 B).

The deterministic capability filter (``_filter_skills_md_by_capability``) shows a
robot only the branch/steps its capabilities support, and drops a whole skill whose
frontmatter ``capabilities`` (requires-any) the robot lacks — so a fixed-arm gripper
robot never sees dual-arm actions, a dual-arm robot never sees gripper actions, and a
base-less robot never sees ``transport`` or the mobile-base steps.
"""

from pathlib import Path

import jiuwensymbiosis.skills as sp
from jiuwensymbiosis.agent.fast import DEFAULT_REGISTRY
from jiuwensymbiosis.agent.fast.planner import _filter_skills_md_by_capability

_GRIPPER = ["motion.cartesian", "grasp.parallel", "vision.detection"]
_DUAL_MOBILE = ["grasp.dual_arm", "motion.base", "motion.waist", "motion.lift", "vision.detection"]
_DUAL_FIXED = ["grasp.dual_arm", "vision.detection", "vision.depth"]

_GRIPPER_ACTIONS = ("goto_xyzr", "open_gripper", "close_gripper", "get_grasp_info_simple")
_DUAL_ACTIONS = ("dual_arm_grasp", "dual_arm_place")
# The actionable base steps (table rows / gated bullets). ``face_surface`` is excluded:
# it can survive in a cross-cutting guidance sentence, but it isn't in a base-less
# robot's vocab, so it can never be emitted — only the executable steps matter.
_BASE_ACTIONS = ("approach_for_place", "approach_for_grasp", "rotate_base", "navigate_relative")


def _filtered_text(caps: list[str]) -> str:
    smd = DEFAULT_REGISTRY.skills_markdown()
    return "\n".join(s["markdown"] for s in _filter_skills_md_by_capability(smd, caps))


def _kept_skills(caps: list[str]) -> set[str]:
    smd = DEFAULT_REGISTRY.skills_markdown()
    return {s["name"] for s in _filter_skills_md_by_capability(smd, caps)}


def test_dual_arm_robot_sees_no_gripper_actions():
    text = _filtered_text(_DUAL_MOBILE)
    assert not any(a in text for a in _GRIPPER_ACTIONS)
    assert all(a in text for a in _DUAL_ACTIONS)


def test_gripper_robot_sees_no_dual_arm_actions():
    text = _filtered_text(_GRIPPER)
    assert not any(a in text for a in _DUAL_ACTIONS)
    assert "goto_xyzr" in text and "open_gripper" in text  # its own branch survives


def test_transport_gated_on_a_movable_base_or_waist():
    assert "transport" in _kept_skills(_DUAL_MOBILE)  # has base + waist
    assert "transport" not in _kept_skills(_GRIPPER)  # fixed arm
    assert "transport" not in _kept_skills(_DUAL_FIXED)  # dual-arm bolted down


def test_fixed_dual_arm_has_no_mobile_base_steps():
    text = _filtered_text(_DUAL_FIXED)
    assert not any(a in text for a in _BASE_ACTIONS)  # no base → base steps gone
    assert all(a in text for a in _DUAL_ACTIONS)  # dual-arm grasp/place still there


def test_skills_declare_both_branches():
    for name in ("visual_pick", "visual_place"):
        md = (Path(sp.__file__).parent / name / "SKILL.md").read_text(encoding="utf-8")
        assert "grasp.dual_arm" in md and "motion.cartesian" in md  # both end-effector branches
        assert "能力" in md
