---
description: Test location, style, async patterns, mocking, and running conventions for jiuwensymbiosis.
language: chinese
paths:
  - "tests/**/*.py"
alwaysApply: false
---

# Testing Rules

## Test Location

- Prefer targeted unit tests that mirror the source path.
  Example: `jiuwensymbiosis/tools/build_robot_tools.py`
  -> `tests/unit_tests/tools/test_build_robot_tools.py`.
- `tests/unit_tests/`: fast deterministic coverage, no hardware/GPU, run in
  CI. Subdirectories already mirror `jiuwensymbiosis/` subsystems
  (`agent/`, `api/`, `tools/`, `adapters/`, `rails/`, `skills/`, `env/`,
  `serving/`, `utils/`).
- `tests/integration/`: requires real hardware, GPU, or external services;
  commonly skipped in CI.
- `tests/mocks/`: shared `MockApi`, `MockEnv`, `MockDriver`, `MockScene`
  fixtures — use these to keep unit tests hardware-free.

## Choosing Test Patterns

| Pattern | When to Use | Characteristics |
|---------|-------------|----------------|
| **Unit test** (`@pytest.mark.unit`) | Isolated logic, no I/O, fast feedback | Mock all hardware via `tests/mocks/` |
| **Integration test** (`@pytest.mark.integration`) | Real arm / camera / detector | Skipped in CI; run on the bench |

**Decision rules**:
- If it touches serial/CAN/socket, a real camera, or the detection
  subprocess -> integration test.
- If it tests a single function/mixin/rail in isolation -> unit test, using
  `MockEnv` / `MockApi`.
- If you change capability gating or tool emission -> add unit tests under
  `tests/unit_tests/api/` and `tests/unit_tests/tools/`.

## Test Style

- This repo uses `pytest` with `asyncio_mode = "auto"` (see `pyproject.toml`).
- `pytest-mock` is available; prefer the `mocker` fixture for patches.
- Test class naming: `Test<Feature>` or `Test<FeatureName>`.

## Credentials and Mocks

- jiuwensymbiosis has no real-credential surface in library code (no API
  keys / tokens in source). Keep it that way.
- For LLM calls in tests, use `MockModel` (the `--mock` path) instead of
  real model credentials. `build_robot_agent(..., model=MockModel())`.
- Never hard-code real hardware endpoints or device paths in test files;
  use `MockDriver` / `MockArmEnv`.

## Async Tests

- `pytest-asyncio` is configured with `asyncio_mode = "auto"` and function
  loop scope — no `@pytest.mark.asyncio` boilerplate needed.
- Example:

```python
async def test_tool_emits_for_capability():
    api = build_test_api()
    tools = build_robot_tools(api, env=MockEnv())
    assert "goto_xyzr" in {t.name for t in tools}
```

## Assertions and Coverage

- Use descriptive assertion messages for non-obvious conditions.
- New public API changes (new `@robot_tool`, new mixin method, new env
  property) require corresponding test updates.
- When behavior changes are user-visible, update `examples/` and `docs/`
  alongside tests.

## Running Tests

- Run all unit tests: `pytest tests/unit_tests/`
- Run a single file: `pytest tests/unit_tests/tools/test_build_robot_tools.py`
- Filter by name: `pytest -k "test_capabilities"`
- Run the full suite (incl. integration, usually skipped): `pytest`
- Adapter smoke test (runtime, every `@robot_tool`):
  `python scripts/smoke_test_adapter.py`
- Adapter static check: `python scripts/validate_adapter.py --module jiuwensymbiosis.adapters.my_robot`
