# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Piper binding for the body-agnostic slot-pick skill.

The slot-pick framework (``jiuwensymbiosis.tools.slot_pick``) is body-agnostic;
piper only has to supply a ``SlotPickStrategy``. Piper is 6-DoF + parallel
gripper and its tilted-tool geometry already lives inside ``PiperApi.goto_xyzr``,
so the default ``GripperStrategy`` is exactly what it needs — this module is the
thin glue that builds it from a ``SlotPickConfig`` and loads that config from a
YAML ``slot_pick:`` block.

Usage::

    from jiuwensymbiosis.adapters.piper import build_piper_session
    from jiuwensymbiosis.adapters.piper.slot_pick import (
        load_piper_slot_pick_config,
        build_piper_slot_pick_tool,
    )

    session = build_piper_session.from_yaml("configs/piper/slot_pick.yaml")
    cfg = load_piper_slot_pick_config("configs/piper/slot_pick.yaml")
    with session:
        tool = build_piper_slot_pick_tool(session.api, cfg)
        agent = build_robot_agent(session, enable_skill=True, extra_tools=[tool])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from jiuwensymbiosis.tools.slot_pick import (
    GripperStrategy,
    SlotPickConfig,
    build_slot_pick_tool,
)


def build_piper_slot_pick_strategy(api: Any, cfg: SlotPickConfig) -> GripperStrategy:
    """Build piper's slot-pick motion/grasp strategy.

    Reach clamping + safe-travel floor are sourced from the slot-pick config so a
    single YAML drives both the loop and the body guards. Piper's tilted-tool
    handling is already inside ``PiperApi.goto_xyzr`` — nothing extra here.
    """
    return GripperStrategy(
        api,
        max_reach_radius_mm=cfg.max_reach_radius_mm,
        safe_travel_z_min_mm=cfg.safe_travel_z_min_mm,
    )


def build_piper_slot_pick_tool(
    api: Any,
    cfg: SlotPickConfig,
    *,
    name: str = "slot_pick",
    agent_id: str | None = None,
) -> Any:
    """Build the openjiuwen ``slot_pick`` Tool bound to a piper api."""
    strategy = build_piper_slot_pick_strategy(api, cfg)
    return build_slot_pick_tool(api, strategy, cfg, name=name, agent_id=agent_id)


def load_piper_slot_pick_config(path: str | Path) -> SlotPickConfig:
    """Load a ``SlotPickConfig`` from a YAML's ``slot_pick:`` block.

    The same YAML can also carry the standard piper ``env.cfg.low_level`` block
    consumed by ``build_piper_session.from_yaml`` — the two live side by side.
    """
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    block = data.get("slot_pick")
    if not isinstance(block, dict):
        raise ValueError(f"{path} has no 'slot_pick:' block; cannot build a SlotPickConfig.")
    return SlotPickConfig.from_mapping(block)
