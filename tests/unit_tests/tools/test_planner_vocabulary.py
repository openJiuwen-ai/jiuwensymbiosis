# coding: utf-8
"""规划器词表的跨本体不变量。

守两件事：
① 规划器看到的动作**全部**来自共享 `ActionSpec` 词表——本体专有工具不进词表，
   否则 coding agent 学到的编排一换本体就失效；
② 内置 SKILL.md 引用的动作名**全部**在词表内——否则 `parse_sequence` 会以
   `unknown op` 整条拒绝，而这只会在真机跑起来时才暴露。
"""

from __future__ import annotations

import re

import pytest

from jiuwensymbiosis.api.actions import ACTIONS
from jiuwensymbiosis.introspect import build_session
from jiuwensymbiosis.tools.robot_control_tool import _build_action_index

CONFIGS = ["configs/piper/piper.yaml", "configs/so101/so101.yaml", "configs/cruzr/cruzr.yaml"]


def _ids(path: str) -> str:
    return path.split("/")[1]


@pytest.mark.parametrize("config", CONFIGS, ids=_ids)
def test_planner_only_actions_all_come_from_the_shared_vocabulary(config):
    session = build_session(config)
    visible = _build_action_index(session.api, env=session.env, planner_only=True)
    stray = sorted(set(visible) - set(ACTIONS))
    assert not stray, (
        f"{_ids(config)} exposes {stray} to the planner but they are not in the shared "
        f"vocabulary. A body-specific tool must stay planner-invisible, or be promoted "
        f"to an ActionSpec so every body means the same thing by it."
    )


@pytest.mark.parametrize("config", CONFIGS, ids=_ids)
def test_dispatch_index_still_carries_the_body_specific_tools(config):
    """Invisible ≠ unavailable: robot_control and scripts must still reach them."""
    session = build_session(config)
    dispatchable = _build_action_index(session.api, env=session.env)
    visible = _build_action_index(session.api, env=session.env, planner_only=True)
    assert set(visible) < set(dispatchable)


def test_builtin_skills_only_name_actions_the_vocabulary_defines():
    """Every ``action`` a SKILL.md prescribes must exist as a shared action.

    A name that has drifted (renamed, merged away, demoted to a private method)
    makes ``parse_sequence`` reject the whole expansion with ``unknown op`` — at
    run time, on the robot. Catch it here instead.
    """
    from jiuwensymbiosis.agent.fast import DEFAULT_REGISTRY
    from jiuwensymbiosis.agent.fast.sequence import KNOWN_SPECIAL_OPS

    known = set(ACTIONS) | set(KNOWN_SPECIAL_OPS)
    offenders: dict[str, set[str]] = {}
    for skill in DEFAULT_REGISTRY.skills_markdown():
        # Action names appear as `backticked_identifier` in the workflow tables.
        named = set(re.findall(r"`([a-z_][a-z0-9_]*)`", skill["markdown"]))
        # Only judge names that look like actions we might have moved: anything the
        # vocabulary once had. Free prose (`robot_control`, `motion.base`, …) is not
        # an action and must not be flagged.
        suspects = {n for n in named if n in known or n in _RETIRED_ACTION_NAMES}
        gone = {n for n in suspects if n not in known}
        if gone:
            offenders[skill["name"]] = gone
    assert not offenders, f"SKILL.md names actions that no longer exist: {offenders}"


# Names that used to be actions and were deliberately retired. Listing them keeps the
# check above honest: a SKILL.md still mentioning one is a real bug, not free prose.
_RETIRED_ACTION_NAMES = {"face_object", "face_surface"}
