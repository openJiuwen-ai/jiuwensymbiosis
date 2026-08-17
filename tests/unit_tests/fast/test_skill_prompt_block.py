# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The skill block a planner reads states each fact once.

``_format_skills_md`` renders the frontmatter into a header and then pastes the
SKILL.md. Pasting the frontmatter as well said everything twice — and a restated
fact is the one that goes stale, because nothing checks prose against the fields it
restates. Dropping it is only safe while the header really does render every field,
so that is what these tests pin.
"""

from __future__ import annotations

import pytest
import yaml

from jiuwensymbiosis.agent.fast import DEFAULT_REGISTRY
from jiuwensymbiosis.agent.fast.planner import (
    _filter_skills_md_by_capability,
    _format_skills_md,
    _strip_frontmatter,
)

_FRONTMATTER_KEYS = {"name", "description", "capabilities", "requires", "provides",
                     "invalidates", "invalidates_locations"}


@pytest.fixture
def skills():
    return DEFAULT_REGISTRY.skills_markdown()


def test_every_frontmatter_field_survives_in_the_header(skills):
    """Strip only what the header already says — anything else would be a real loss."""
    for s in skills:
        declared = set(yaml.safe_load(s["markdown"].split("---")[1]) or {})
        assert declared <= _FRONTMATTER_KEYS, f"{s['name']}: unrendered frontmatter key(s)"
        block = _format_skills_md([s])
        head = block.split("\n#", 1)[0]
        assert s["name"] in head and s["description"][:20] in head
        for cap in s.get("capabilities") or ():
            assert cap in head, f"{s['name']}: capability gate {cap} lost"
        for token in (*s.get("requires", ()), *s.get("provides", ()), *s.get("invalidates", ())):
            assert token in head, f"{s['name']}: contract token {token} lost"


def test_the_frontmatter_is_not_pasted_a_second_time(skills):
    for s in skills:
        body = _format_skills_md([s]).split("\n", 1)[1]
        assert not body.lstrip().startswith("---")
        assert "\nrequires:" not in body and "\nprovides:" not in body


def test_the_workflow_itself_is_untouched(skills):
    for s in skills:
        block = _format_skills_md([s])
        assert f"# {s['name']}" in block, "the SKILL.md body must still be there in full"
        assert block.rstrip().endswith(s["markdown"].rstrip()[-40:])


def test_a_body_without_frontmatter_is_passed_through():
    assert _strip_frontmatter("# skill\nbody") == "# skill\nbody"
    assert _strip_frontmatter("---\nname: x\nno closing fence") == "---\nname: x\nno closing fence"


def test_capability_filtering_is_unaffected(skills):
    """It runs before rendering and reads the field, not the prose — but pin it anyway."""
    kept = _filter_skills_md_by_capability(skills, ["grasp.parallel", "vision.detection"])
    assert {s["name"] for s in kept} == {"visual_pick", "visual_place"}  # transport needs a base
    kept_mobile = _filter_skills_md_by_capability(skills, ["motion.base", "grasp.dual_arm"])
    assert "transport" in {s["name"] for s in kept_mobile}
