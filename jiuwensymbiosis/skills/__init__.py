# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Built-in jiuwensymbiosis skills package.

Each subdirectory is an openjiuwen ``SkillManager``-compatible skill,
containing a ``SKILL.md`` file with YAML frontmatter (at minimum a
``description`` field) followed by a markdown workflow body.

``SKILLS_DIR`` is passed directly as ``SkillUseRail.skills_dir`` so
``SkillUseRail`` loads all built-in skills using openjiuwen's standard
discovery mechanism.
"""

from pathlib import Path

SKILLS_DIR: Path = Path(__file__).parent

__all__ = ["SKILLS_DIR"]
