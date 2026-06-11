# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Body-agnostic slot-pick skill (multi-chip → slot placement loop).

Layering:
  * ``detect``   — detection / candidate filtering / tray-swap polling (api duck-type)
  * ``strategy`` — per-body motion guards + grasp/release (``SlotPickStrategy`` / ``GripperStrategy``)
  * ``skill``    — ``SlotPickConfig`` + ``run_slot_pick`` loop + ``build_slot_pick_tool``

An adapter (e.g. piper) supplies a ``GripperStrategy(api, ...)`` and a
``SlotPickConfig`` to ``build_slot_pick_tool`` to get a ready openjiuwen Tool.
"""

from jiuwensymbiosis.tools.slot_pick.skill import (
    SlotPickConfig,
    SlotPickSkillTool,
    build_slot_pick_tool,
    geometric_completion_judge,
    run_slot_pick,
    run_watch_pick_place,
)
from jiuwensymbiosis.tools.slot_pick.strategy import (
    GripperStrategy,
    SlotPickStrategy,
)
from jiuwensymbiosis.tools.slot_pick.vlm_judge import make_vlm_completion_judge

__all__ = [
    "GripperStrategy",
    "SlotPickConfig",
    "SlotPickSkillTool",
    "SlotPickStrategy",
    "build_slot_pick_tool",
    "geometric_completion_judge",
    "make_vlm_completion_judge",
    "run_slot_pick",
    "run_watch_pick_place",
]
