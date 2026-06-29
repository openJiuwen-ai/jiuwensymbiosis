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

from jiuwensymbiosis.skills import SKILLS_DIR

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillSpec:
    """One skill: its name, one-line description, and the SKILL.md on disk."""

    name: str
    description: str
    md_path: Path

    def markdown(self) -> str:
        """Full SKILL.md text (frontmatter + workflow body)."""
        return self.md_path.read_text(encoding="utf-8")


def _parse_frontmatter_description(md_text: str) -> str:
    """Best-effort pull of ``description`` from a SKILL.md YAML frontmatter."""
    if not md_text.startswith("---"):
        return ""
    end = md_text.find("\n---", 3)
    if end == -1:
        return ""
    try:
        meta = yaml.safe_load(md_text[3:end]) or {}
    except Exception:  # noqa: BLE001 - best-effort frontmatter parse
        return ""
    return str(meta.get("description", "")) if isinstance(meta, dict) else ""


def _load_skill(skill_dir: Path) -> SkillSpec | None:
    """Build a ``SkillSpec`` from a skill directory, or ``None`` if no SKILL.md."""
    md = skill_dir / "SKILL.md"
    if not md.is_file():
        return None
    text = md.read_text(encoding="utf-8")
    name = skill_dir.name
    return SkillSpec(name=name, description=_parse_frontmatter_description(text), md_path=md)


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
        """``[{name, description}]`` — a compact catalogue (no markdown)."""
        return [{"name": s.name, "description": s.description} for s in self._skills.values()]

    def skills_markdown(self) -> list[dict[str, str]]:
        """``[{name, markdown}]`` — full SKILL.md text for the sequence compiler."""
        return [{"name": s.name, "markdown": s.markdown()} for s in self._skills.values()]


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
