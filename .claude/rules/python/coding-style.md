---
description: Python-specific coding conventions — immutability, type annotations, toolchain, and anti-patterns for jiuwensymbiosis.
language: chinese
paths:
  - "jiuwensymbiosis/**/*.py"
alwaysApply: false
---

# Python Coding Style (Extended)

Extends `rules/code-style.md` with Python-specific conventions.
See `skills/python-patterns` for the deep pattern reference.

## Immutability

Prefer immutable data structures. Use `@dataclass(frozen=True)` for
data-only types (configs, cards, event records). Use `NamedTuple` for
simple fixed-length records. See `skills/python-patterns` for complete
examples.

jiuwensymbiosis configs (`<Feature>Config`) are mutable dataclasses today
because `from_yaml()` / runtime overrides need to set fields; do not force
`frozen=True` onto existing config classes without checking call sites.

## Modern Type Annotations

Use Python 3.9+ built-in generics instead of `typing` module equivalents:

```python
# Preferred (Python 3.9+)
def process(items: list[int], mapping: dict[str, str]) -> set[str]: ...

# Avoid (legacy form)
from typing import List, Dict, Set
def process(items: List[int], mapping: Dict[str, str]) -> Set[str]: ...
```

Use `typing.Protocol` for structural subtyping (duck typing with type hints).
The `RobotDriver` Protocol in `jiuwensymbiosis/adapters/_common/protocol.py`
is the canonical example — new drivers satisfy it structurally, no
inheritance required.

## Toolchain

- **Linter & formatter**: `ruff` (single tool, Black-compatible). Configured
  in `[tool.ruff]` / `[tool.ruff.lint]` / `[tool.ruff.format]` in
  `pyproject.toml`. Run `ruff check .` to lint, `ruff format .` to format,
  `ruff check --fix .` to auto-fix.
- **Type checker**: `mypy` (optional, configured in `[tool.mypy]`). Run
  `mypy jiuwensymbiosis/`.
- **Import sorter**: `ruff` (`I` rule) — no standalone `isort` needed.

> These tools are **not** installed by default. `pip install -e ".[dev]"`
> pulls `pytest` only; install `ruff mypy` explicitly when you want to run
> them (no `black` — `ruff format` replaces it). The configs in
> `pyproject.toml` are pre-wired so the tools work as soon as they are
> installed.

## Memory Optimization

Use `__slots__` for lightweight classes instantiated frequently. Only when
the class has a fixed set of attributes and memory efficiency matters; do
not use `__slots__` when the class needs arbitrary attributes or is
subclassed with additional fields. See `skills/python-patterns`.

## Anti-Patterns

- **Mutable default arguments** — use `None` and initialize inside function:
  `def f(x: list[str] | None = None)` instead of `def f(x=[])`.
- **`type()` checking** — use `isinstance()` instead: `isinstance(x, str)`.
- **Bare `except`** — always catch specific exceptions, never bare `except:`.
  In rail / motion code, catch `ValueError` for safety rejections so the LLM
  can self-correct; re-raise everything else.
- **`print()` in library code** — use `get_logger(name)`.

See `skills/python-patterns` for correct patterns and detailed examples.
