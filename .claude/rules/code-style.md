---
description: Python code style, formatting, naming, imports, and async safety rules for jiuwensymbiosis.
language: chinese
paths:
  - "jiuwensymbiosis/**/*.py"
  - "scripts/**/*.py"
  - "templates/**/*.py"
  - "tests/**/*.py"
alwaysApply: false
---

# Code Style Rules

## Language and Formatting

- Python 3.11+ required (see `.python-version`).
- **Line length: 120 characters** (matches `[tool.ruff]` in `pyproject.toml`).
- `ruff` is the **single** tool for both linting and formatting (a
  Black-compatible drop-in — no separate `black` dependency). Run
  `ruff check .` to lint, `ruff format .` to format; `ruff check --fix .`
  auto-fixes. See `[tool.ruff]` / `[tool.ruff.lint]` / `[tool.ruff.format]`
  in `pyproject.toml`.
- Match surrounding module style before introducing new patterns.
- Add type hints for new public APIs; keep docstrings aligned with the
  surrounding module.

## Comments and Docstrings

- Prefer self-documenting names and small functions over explanatory comments.
- Do not add comments that restate the code, repeat the test name/assertion, or
  quote issue numbers/PR context. Put historical context in commits, issues, or
  docs instead.
- Add a comment only when it explains a non-obvious invariant, hardware safety
  contract, external API quirk, unit convention, ordering requirement, or
  compatibility decision that a maintainer could otherwise break.
- Keep required comments short: one sentence or at most two wrapped lines. If a
  longer explanation is needed, move it to `docs/` and link or name that doc.
- Docstrings are for public APIs, generated-user-facing skeletons, and complex
  helpers. Avoid docstrings on tests or private helpers when the function name
  and assertions already describe the behavior.
- Generated templates (`scripts/new_adapter/render.py`, `templates/`) must be
  especially terse because every comment is copied into user code. Include only
  comments that adapter authors must act on.
- Before finishing a change, scan your diff for added `#` comments and
  docstrings. Delete any that fail the "would this prevent a future bug?"
  test.

## Async Safety

- Keep library code async-safe. Avoid blocking calls in async paths unless
  the module already does so deliberately.
- For async file I/O, prefer `aiofiles` or `asyncio.to_thread()` over
  synchronous `open()`.
- jiuwensymbiosis is mostly synchronous (hardware I/O is blocking by
  nature); do not sprinkle `async` into adapter/driver code without reason.

## Logging

- Do not use `print()` in library code. Use `get_logger(name)` from
  `jiuwensymbiosis.utils.logging` — it routes through `configure_logging`
  so `TraceLogHandler` and file handlers attach correctly.
- Legacy `logging.getLogger(__name__)` calls remain valid but prefer
  `get_logger` for new code.
- Full rules: see `.claude/rules/logging.md` (project-specific, not migrated
  from agent-core).

## Naming Conventions

- Follow PEP 8; `ruff format` enforces default style.
- Type aliases and schemas go in `schema/` or `types/` subdirectories.
- Capability strings: dotted `"<domain>.<verb>"` (e.g. `motion.cartesian`,
  `grasp.suction`, `vision.detection`) — see `env/base.py:KNOWN_CAPABILITIES`.
- Config/dataclass types: `<Feature>Config`; env subclasses: `BaseRobotEnv`;
  api subclasses: `BaseRobotApi`; driver subclasses: `RobotDriver`.

## Imports

- Use absolute imports within the `jiuwensymbiosis` package.
- Do not use wildcard imports (`from module import *`) in library code.
- Group imports: stdlib, third-party, local/relative (isort handles this if
  installed; otherwise match surrounding files).
- `clear_proxy_env()` (from `jiuwensymbiosis.utils`) must be called before
  `import openjiuwen` in any entry point — see root `CLAUDE.md` "Proxy Hygiene".

## File Organization

- One public class per module preferred; small related utilities may share
  a module.
- Private implementation details start with `_` or `__`.
- `__init__.py` exports the public surface only; keep it minimal.
- Adapter code lives under `jiuwensymbiosis/adapters/<name>/` following the
  6-file pattern (config/lowlevel/env/api/session/config_template.yaml).
  See `.claude/rules/adapters.md` (project-specific) for the full pattern.
