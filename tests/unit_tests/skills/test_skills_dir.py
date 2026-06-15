# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.skills — SKILLS_DIR and SKILL.md integrity."""

from __future__ import annotations

from jiuwensymbiosis.skills import SKILLS_DIR


class TestSkillsDir:
    def test_skills_dir_exists(self):
        assert SKILLS_DIR.is_dir()

    def test_skill_md_files_present(self):
        expected = {"visual_pick", "visual_place", "slot_pick"}
        actual = {d.name for d in SKILLS_DIR.iterdir() if d.is_dir()}
        assert expected.issubset(actual), f"Missing: {expected - actual}"

    def test_skill_md_has_frontmatter(self):
        import yaml

        for skill_dir in SKILLS_DIR.iterdir():
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            content = skill_md.read_text(encoding="utf-8")
            assert content.strip().startswith("---"), f"{skill_md} missing YAML frontmatter"
            parts = content.split("---", 2)
            if len(parts) >= 3:
                fm_text = parts[1]
                fm = yaml.safe_load(fm_text)
                assert "description" in fm, f"{skill_md} frontmatter missing 'description'"

    def test_skill_md_has_workflow_body(self):
        for skill_dir in SKILLS_DIR.iterdir():
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            content = skill_md.read_text(encoding="utf-8")
            parts = content.split("---")
            if len(parts) >= 3:
                body = "---".join(parts[2:]).strip()
                assert len(body) > 0, f"{skill_md} has empty workflow body"
