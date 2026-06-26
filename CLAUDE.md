@AGENTS.md

# Claude Code Notes

Shared project rules live in `AGENTS.md` (cross-tool — Cursor / Copilot read
it too). This file only adds Claude-specific pointers.

## Rules & Skills Index

Topic-scoped **rules** (short, hard, path-gated via frontmatter `paths`)
live in `.claude/rules/`. They are injected only when you touch matching
files. **Skills** (longer, on-demand reference manuals) live in
`.claude/skills/`.

### Rules (`.claude/rules/`)

| File | Scope | When it loads |
|---|---|---|
| `karpathy-principles.md` | Coding behavior (think / simplify / surgical / goal-driven) | Always (`alwaysApply: true`) |
| `code-style.md` | Python style, formatting, naming, imports, async safety | `jiuwensymbiosis/**/*.py` |
| `security.md` | Credentials, **physical safety**, proxy hygiene, dependency review | `jiuwensymbiosis/**/*.py`, `configs/**/*.yaml` |
| `testing.md` | Test location, mock-hardware pattern, async tests, running | `tests/**/*.py` |
| `python/coding-style.md` | Immutability, modern type hints, toolchain, anti-patterns | `jiuwensymbiosis/**/*.py` |
| `python/security.md` | Secret management, subprocess safety, dependency review | `jiuwensymbiosis/**/*.py` |
| `python/testing.md` | Pytest markers, fixtures, mocking, async tests | `tests/**/*.py` |

### Skills (`.claude/skills/`)

On-demand deep references — invoke when the task needs the full pattern
catalog, not on every edit.

| Skill | Use for |
|---|---|
| `python-patterns` | Python idioms: frozen dataclasses, Protocol, exception hierarchy, async, decorators, package layout |
| `python-testing` | Deep pytest guide: TDD, fixtures, factory fixtures, mocking, async, adapter smoke tests |
| `security-review` | Pre-PR checklist: secrets, physical safety, subprocess, dependencies, log/trace hygiene |

## Permissions & Env

Permissions and env vars: see `.claude/settings.local.json`.

## Claude Workflow

- Run `/memory` to manage auto memory.
- Run `/context` to see which files are loaded in the current session.
