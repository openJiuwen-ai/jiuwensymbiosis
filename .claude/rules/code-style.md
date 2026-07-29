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
- `# noqa` suppressions are discouraged. `pyproject.toml` already ignores
  `BLE001` (the rule that forbids `except Exception`), so `# noqa: BLE001`
  must NOT be sprinkled on `except Exception as exc:` blocks — the project
  permits that catch form. Reserve `# noqa` for genuinely unfixable cases,
  always with the rule code AND a reason (`# noqa: SLF001 - standalone hardware calibration`),
  never bare. Fix the lint issue first when the fix is cheaper than the
  suppression.
- Before finishing a change, scan your diff for added `#` comments and
  docstrings. Delete any that fail the "would this prevent a future bug?"
  test.

## Exception Handling

- `except Exception as exc:` is **allowed** (BLE001 is globally ignored in
  `pyproject.toml`). The project does not forbid catching the broad
  `Exception` base — hardware/transport/adapter code routinely deals with
  vendor-raised exceptions of many types, and narrowing every catch site is
  not required.
- **Every caught exception must be logged.** Use `get_logger(name)`:
  `logger.warning` for recoverable paths, `logger.error` for degraded/unsafe
  states; pass `exc` through (or `logger.exception` for the traceback).
- What is **forbidden** is *suppressing or silently dropping* an exception:
  a body that swallows it with a bare `pass` / empty `return` / `break` /
  `continue` and **no log and no recovery**. Log at minimum; recover or
  re-raise where safety demands it.
- Catch `ValueError` specifically in safety/motion code so the LLM can
  self-correct; broad `Exception` catches are for best-effort teardown or
  fallback paths (e.g. torque restore, frame save) where the recovery
  contract is "try, log, proceed", not "ignore".
- Never swallow `KeyboardInterrupt` / `SystemExit` under a bare
  `except Exception` — `Exception` does not catch `BaseException` subclasses,
  but a hand-written `except BaseException` that absorbs them is always a bug.

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
