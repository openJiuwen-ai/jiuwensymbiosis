@AGENTS.md

# Claude Code Notes

Shared project rules live in `AGENTS.md` (cross-tool — Cursor / Copilot read
it too). This file only adds Claude-specific pointers.

## Rules & Skills Index

**Rules** in `.claude/rules/` define recurring constraints; scoped rules
declare their file patterns in `paths`. **Skills** in `.claude/skills/`
provide task-specific workflows. Read linked references only when needed.

Keep facts in their owning source: tooling in `pyproject.toml`, shared
project constraints in `AGENTS.md`, and behavior in implementation and tests.
When documentation and code disagree, report the discrepancy instead of
silently treating one as proof of the other. Avoid copying schemas, tool
configuration, or runnable examples into multiple guidance files.

### Rules (`.claude/rules/`)

| File | Scope | When it loads |
|---|---|---|
| [development-principles.md](.claude/rules/development-principles.md) | Scope, design effort, evidence, focused changes | Session-wide; no `paths` |
| [code-style.md](.claude/rules/code-style.md) | Public types, state, async, exceptions, logging, suppressions | Python under package, scripts, templates, examples, tests |
| [security.md](.claude/rules/security.md) | Motion/recovery, execution trust, subprocesses, credentials and persistence | Package/runtime skills, scripts, examples, templates, configs, dependencies; exact patterns in file |
| [testing.md](.claude/rules/testing.md) | Isolation, doubles, async callbacks, assertions, test selection | `tests/**/*.py` |

### Skills (`.claude/skills/`)

Use the workflow matching the task; a small local change does not require
a design document or a full review report.

| Skill | Use for |
|---|---|
| [module-design](.claude/skills/module-design/SKILL.md) | Design public-contract, ownership, lifecycle or cross-module changes before implementation |
| [review-architecture-change](.claude/skills/review-architecture-change/SKILL.md) | Compare base/head architecture and explain findings; Markdown by default, HTML on request |
| [python-testing](.claude/skills/python-testing/SKILL.md) | Select and implement project-specific tests using existing doubles and architecture checks |
| [security-review](.claude/skills/security-review/SKILL.md) | Explicitly invoked safety/security review of actual execution paths; retains manual invocation |

The shared [change-validation map](.claude/references/change-validation.md)
links change types to existing checks. It is an on-demand selection aid,
not a second architecture specification or evidence that tests were run.

## Permissions & Env

Permissions and env vars: see `.claude/settings.local.json`.

## 语言约定 (Language Convention)

- **面向用户的输出一律用中文**：计划书、提问、方案说明、交互解释、总结等
  写给用户看的内容,统一使用中文。
- 代码、标识符、英文技术术语、日志、docstring、注释等尽量使用英文——本约定只约束"对用户说话"的部分,不改变代码本身的语言。

## Claude Workflow

- Run `/memory` to manage auto memory.
- Run `/context` to see which files are loaded in the current session.
