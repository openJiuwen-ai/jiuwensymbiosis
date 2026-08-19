# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Skill catalogue for the C1 fast path — skills are SKILL.md, not Python.

The fast path's single source of truth is each skill's ``SKILL.md`` (the same
file the agent reads). This registry just *enumerates* skills and hands their
full markdown to the sequence compiler (``planner.compile_sequence``); it holds
NO per-skill Python executor — the workflow lives only in the markdown.

By default it auto-discovers the built-in skills under ``skills/`` (every
subdirectory with a ``SKILL.md``). Register an external skill directory with
``register_skill_dir`` to extend the catalogue without touching this file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from jiuwensymbiosis.api.state import validate_tokens
from jiuwensymbiosis.skills import SKILLS_DIR

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillSpec:
    """One skill: name, one-line description, gates, planning contract, and the SKILL.md.

    ``capabilities`` is the skill's **requires-any** gate (from frontmatter): a robot
    lacking *all* of them can't use the skill, so it's dropped before the compiler ever
    sees it. Empty = applies to any robot (its internal per-capability branches decide).

    ``requires`` / ``provides`` / ``invalidates`` / ``invalidates_locations`` are the
    same planning contract an action carries (``api.state``) — a skill is a compound
    action, so skill-level and action-level planning share one reasoning machinery.
    They are derivable from the skill's own action sequence: ``requires`` =
    pre-conditions no earlier step in the sequence satisfies, ``provides`` = tokens the
    end state gained, ``invalidates`` = tokens it lost, ``invalidates_locations`` = any
    step moves the body. The built-in skills declare them by hand; distilled skills get
    them computed.
    """

    name: str
    description: str
    md_path: Path
    capabilities: frozenset[str] = frozenset()
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    invalidates: tuple[str, ...] = ()
    invalidates_locations: bool = False

    def markdown(self) -> str:
        """Full SKILL.md text (frontmatter + workflow body)."""
        return self.md_path.read_text(encoding="utf-8")


def _parse_frontmatter(md_text: str) -> dict[str, Any]:
    """Best-effort parse of a SKILL.md YAML frontmatter into a dict ({} if none/bad)."""
    if not md_text.startswith("---"):
        return {}
    end = md_text.find("\n---", 3)
    if end == -1:
        return {}
    try:
        meta = yaml.safe_load(md_text[3:end]) or {}
    except Exception:  # noqa: BLE001 - best-effort frontmatter parse
        return {}
    return meta if isinstance(meta, dict) else {}


def _frontmatter_tokens(meta: dict[str, Any], key: str, *, owner: str) -> tuple[str, ...]:
    """Read one state-token list from frontmatter, validated against the closed vocabulary."""
    raw = meta.get(key) or []
    if not isinstance(raw, list):
        return ()
    return validate_tokens((str(t) for t in raw), field=key, owner=f"skill {owner}")


def _load_skill(skill_dir: Path) -> SkillSpec | None:
    """Build a ``SkillSpec`` from a skill directory, or ``None`` if no SKILL.md."""
    md = skill_dir / "SKILL.md"
    if not md.is_file():
        return None
    meta = _parse_frontmatter(md.read_text(encoding="utf-8"))
    caps = meta.get("capabilities") or []
    return SkillSpec(
        name=skill_dir.name,
        description=str(meta.get("description", "")),
        md_path=md,
        capabilities=frozenset(str(c) for c in caps) if isinstance(caps, list) else frozenset(),
        requires=_frontmatter_tokens(meta, "requires", owner=skill_dir.name),
        provides=_frontmatter_tokens(meta, "provides", owner=skill_dir.name),
        invalidates=_frontmatter_tokens(meta, "invalidates", owner=skill_dir.name),
        invalidates_locations=bool(meta.get("invalidates_locations", False)),
    )


class SkillRegistry:
    """Name → ``SkillSpec`` map; the compiler reads markdown from it."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillSpec] = {}

    def register(self, spec: SkillSpec, *, overwrite: bool = False) -> None:
        """Add a skill. Raises on a duplicate name unless ``overwrite=True``."""
        if spec.name in self._skills and not overwrite:
            raise ValueError(f"skill {spec.name!r} already registered (pass overwrite=True)")
        self._skills[spec.name] = spec

    def register_dir(self, skills_dir: Path, *, overwrite: bool = False) -> int:
        """Discover and register every ``SKILL.md`` subdirectory. Returns the count."""
        n = 0
        for child in sorted(Path(skills_dir).iterdir()):
            if not child.is_dir():
                continue
            spec = _load_skill(child)
            if spec is not None:
                self.register(spec, overwrite=overwrite)
                n += 1
        return n

    def get(self, name: str) -> SkillSpec | None:
        """Look up a skill by name, or ``None``."""
        return self._skills.get(name)

    def names(self) -> list[str]:
        """All registered skill names."""
        return list(self._skills)

    def catalogue(self) -> list[dict[str, Any]]:
        """``[{name, description, capabilities, requires, provides, invalidates}]``.

        The compact planning view (no markdown): enough for skill-level planning to
        reason about which skills apply and in what order, without reading any prose.
        """
        return [
            {
                "name": s.name,
                "description": s.description,
                "capabilities": sorted(s.capabilities),
                "requires": list(s.requires),
                "provides": list(s.provides),
                "invalidates": list(s.invalidates),
                "invalidates_locations": s.invalidates_locations,
            }
            for s in self._skills.values()
        ]

    def skills_markdown(self) -> list[dict[str, Any]]:
        """The catalogue view plus each skill's full ``SKILL.md`` text."""
        return [{**entry, "markdown": self._skills[entry["name"]].markdown()} for entry in self.catalogue()]


# The process-wide default registry — auto-loaded with the built-in skills.
DEFAULT_REGISTRY = SkillRegistry()


def register_skill(spec: SkillSpec, *, overwrite: bool = False) -> None:
    """Register a skill in the default registry."""
    DEFAULT_REGISTRY.register(spec, overwrite=overwrite)


def register_skill_dir(skills_dir: Path, *, overwrite: bool = False) -> int:
    """Discover and register a skills directory in the default registry."""
    return DEFAULT_REGISTRY.register_dir(skills_dir, overwrite=overwrite)


_n = DEFAULT_REGISTRY.register_dir(SKILLS_DIR)
logger.debug("[registry] loaded %d built-in skills from %s", _n, SKILLS_DIR)
