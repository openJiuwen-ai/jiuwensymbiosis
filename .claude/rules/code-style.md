---
paths:
  - "jiuwensymbiosis/**/*.py"
  - "scripts/**/*.py"
  - "templates/**/*.py"
  - "examples/**/*.py"
  - "tests/**/*.py"
---

# Python Code Style

## Sources and public surface

- Read Python versions, dependencies, Ruff and mypy settings from
  [pyproject.toml](../../pyproject.toml); do not maintain copies in guidance.
  Ruff owns formatting, linting, and import sorting.
- Match nearby code and add type annotations to new public interfaces. Use
  absolute package imports and explicit public exports in `__init__.py`.
- Keep types with their owning module. Reuse shared result contracts from
  `jiuwensymbiosis/contracts.py`; preserve its dependency-free role rather than
  introducing competing schema packages.
- Use the capability-sliced Protocols in `jiuwensymbiosis/env/protocol.py` for
  driver interfaces. Do not introduce a common driver base requiring unrelated
  hardware capabilities.
- Follow the action-contract model in `AGENTS.md`. Read existing
  `ActionSpec` / `ToolMeta` definitions before changing them; do not copy
  contract fields into another carrier or recreate the removed mixin layer.

## State and concurrency

- Prefer immutable value records when their lifecycle permits it. Existing
  configuration dataclasses support runtime overrides; check callers before
  freezing them. Do not change mutability as a style-only refactor.
- Hardware I/O is generally synchronous. Do not add async wrappers without
  checking driver thread affinity, serialization, and cancellation behavior.
  Offloading work to a thread does not stop physical motion when the awaiting
  coroutine is cancelled.
- Keep resource ownership explicit. Use the existing session and subsystem
  lifecycle instead of adding a second owner for a driver or sidecar.

## Exceptions and logging

- Reuse `jiuwensymbiosis/errors.py` and relevant subsystem errors. Preserve
  `SafetyViolationError`'s `ValueError` compatibility and machine-readable
  failure codes when wrapping or forwarding failures.
- `except Exception as exc:` is allowed for vendor errors and best-effort
  cleanup. Log a suppressed failure and perform the required recovery; never
  turn an unsafe or incomplete operation into success. Re-raising alone does
  not require duplicate logging at every layer.
- Avoid bare `except:`. Catching `BaseException` requires an explicit
  execution-boundary contract; preserve cancellation, interrupt, and teardown
  semantics rather than absorbing them accidentally.
- Use `get_logger(__name__)` from `jiuwensymbiosis.utils.logging` in new
  library code. Existing `logging.getLogger` calls remain valid. Use the
  central logging configuration; keep `print()` for user-facing CLI output.

## Comments, suppressions, and helpers

- Comments should explain a non-obvious invariant, unit, ordering requirement,
  hardware constraint, or external API behavior. Avoid narrating code or tests.
  Keep generated adapter-template comments limited to actionable guidance.
- Docstrings describe public interfaces and complex behavior. Put long design
  rationale in the relevant documentation and refer to it.
- Fix lint/type errors at their source. If a suppression is unavoidable, use
  a specific rule/error code and a reason; never bare `# noqa` or
  `# type: ignore`. Do not suppress a rule already disabled in the project.
- Use explicit runtime validation for conditions required in production;
  `assert` can disappear under `python -O`.
- Prefer a module function or `staticmethod` for helpers that need no instance
  state. Preserve required instance signatures for protocol methods and
  overrides. Group static helpers consistently with the surrounding class.
