---
description: Pytest markers, fixtures, mocking, and coverage conventions for jiuwensymbiosis Python code.
language: chinese
paths:
  - "tests/**/*.py"
alwaysApply: false
---

# Python Testing (Extended)

Extends `rules/testing.md` with pytest-specific conventions.
See `skills/python-testing` for the deep reference.

## Pytest Markers

jiuwensymbiosis defines two markers in `pyproject.toml`:

```python
import pytest

@pytest.mark.unit          # No hardware or GPU required — runs in CI
class TestBuildRobotTools:
    ...

@pytest.mark.integration   # Requires real hardware, GPU, or external services
class TestPiperPick:
    ...
```

Run only fast tests in CI:

```bash
pytest -m unit
```

Run integration tests on the bench (skipped otherwise):

```bash
pytest -m integration
```

## Fixtures

Shared fixtures live in `tests/conftest.py` and `tests/unit_tests/conftest.py`.
For mock hardware, prefer the ready-made fakes in `tests/mocks/`
(`MockApi`, `MockEnv`, `MockDriver`, `MockScene`) over hand-rolled mocks.

```python
# tests/conftest.py
import pytest
from tests.mocks import MockEnv, MockApi

@pytest.fixture
def mock_env():
    return MockEnv()

@pytest.fixture
def mock_api(mock_env):
    return MockApi(env=mock_env)
```

## Mocking

Use `pytest-mock`'s `mocker` fixture (already a dev dependency):

```python
def test_rail_rejects_below_z_floor(mocker):
    env = MockEnv()
    env.z_min_safe = 50.0
    mocker.patch.object(env, "goto_xyzr")
    rail = SafetyRail(env=env)
    with pytest.raises(ValueError):
        rail.before_tool_call("goto_xyzr", {"x": 0, "y": 0, "z": 10})
```

For patching module-level symbols (e.g. the detector sidecar client):

```python
def test_uses_detector_sidecar(mocker):
    mock_detect = mocker.patch("jiuwensymbiosis.adapters._common.detector_client.init_detector")
    # ...
    mock_detect.assert_called_once()
```

## Async Tests

`pytest-asyncio` is configured with `asyncio_mode = "auto"` in
`pyproject.toml` — async test functions need no decorator:

```python
async def test_async_tool_runs():
    api = build_test_api()
    result = await api.some_async_method()
    assert result.ok
```

## Test Organization

Mirror the source path in test paths:

| Source | Test |
|--------|------|
| `jiuwensymbiosis/tools/build_robot_tools.py` | `tests/unit_tests/tools/test_build_robot_tools.py` |
| `jiuwensymbiosis/rails/safety_rail.py` | `tests/unit_tests/rails/test_safety_rail.py` |
| `jiuwensymbiosis/adapters/piper/api.py` | `tests/unit_tests/adapters/test_piper_api.py` |

## Credentials in Tests

Never hardcode real model API keys or hardware endpoints. Use `MockModel`
for LLM paths and `MockDriver`/`MockArmEnv` for hardware paths. The
`--mock` demo mode (`examples/piper_pick_demo.py --mock`) shows the
end-to-end no-credentials pattern.
