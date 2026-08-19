# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Phase E: built-in transport skill — a GENERAL move (not "must be holding")."""

from pathlib import Path

import jiuwensymbiosis.skills as skills_pkg

MD = Path(skills_pkg.__file__).parent / "transport" / "SKILL.md"


def test_transport_frontmatter_and_capability_gated():
    assert MD.exists()
    text = MD.read_text(encoding="utf-8")
    assert "name: transport" in text
    assert "description:" in text
    # capability-gated reorientation/move (base yaw / relative / waist)
    assert "motion.base" in text and "motion.waist" in text


def test_transport_is_general_move_not_holding_gated():
    text = MD.read_text(encoding="utf-8")
    # A general move: explicitly usable standalone / before pick / while holding —
    # must NOT declare "holding an object" as a precondition.
    assert "先移动" in text or "单独" in text or "无需持物" in text


def test_transport_registered_in_default_registry():
    from jiuwensymbiosis.agent.fast.registry import DEFAULT_REGISTRY

    assert "transport" in DEFAULT_REGISTRY.names()
