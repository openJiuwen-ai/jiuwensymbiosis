---
name: python-testing
description: Deep pytest guide for jiuwensymbiosis — fixtures, mocking, async tests, and the mock-hardware pattern.
---

# Python Testing

Comprehensive pytest patterns for jiuwensymbiosis. This skill extends
`.claude/rules/python/testing.md` and `.claude/rules/testing.md`.

## The Mock-Hardware Pattern (core convention)

jiuwensymbiosis's defining test convention: **unit tests never touch real
hardware**. The `tests/mocks/` package provides `MockApi`, `MockEnv`,
`MockDriver`, `MockScene` for this. Use them instead of hand-rolled fakes.

```python
from tests.mocks import MockEnv, MockApi, MockDriver
from jiuwensymbiosis.tools import build_robot_tools

def test_motion_tools_emit_when_capable():
    env = MockEnv(capabilities={"motion.cartesian", "grasp.parallel"})
    api = MockApi(env=env)
    tools = {t.name for t in build_robot_tools(api, env=env)}
    assert "goto_xyzr" in tools
    assert "close_gripper" in tools
```

`MockModel` (in `jiuwensymbiosis.agent`) replaces the LLM in `--mock` runs
and in agent-level tests.

## TDD Workflow

Write tests before implementation. Follow the red-green-refactor cycle:

1. **RED** — Write a failing test that describes the desired behavior
2. **GREEN** — Write the minimal implementation to make the test pass
3. **REFACTOR** — Improve code quality while keeping tests green

```python
# RED: Write the test first
class TestSafetyRail:
    def test_rejects_z_below_floor(self):
        env = MockEnv()
        env.z_min_safe = 50.0
        rail = SafetyRail(env=env)
        with pytest.raises(ValueError):
            rail.before_tool_call("goto_xyzr", {"x": 0, "y": 0, "z": 10})

    # GREEN: minimal SafetyRail implementation
    # ...

    # REFACTOR: add edge cases
    def test_accepts_z_above_floor(self):
        env = MockEnv()
        env.z_min_safe = 50.0
        rail = SafetyRail(env=env)
        # Should not raise
        rail.before_tool_call("goto_xyzr", {"x": 0, "y": 0, "z": 100})
```

## Fixtures

### conftest.py Organization

Define fixtures in `tests/conftest.py` for project-wide fixtures, or in
`tests/unit_tests/<subsystem>/conftest.py` for subsystem-specific fixtures.

```python
# tests/conftest.py
import pytest
from tests.mocks import MockEnv, MockApi, MockDriver

@pytest.fixture
def mock_env():
    return MockEnv()

@pytest.fixture
def mock_driver():
    return MockDriver()

@pytest.fixture
def mock_api(mock_env):
    return MockApi(env=mock_env)
```

### Factory Fixtures

Useful when tests need slightly different configurations:

```python
@pytest.fixture
def make_env():
    """Factory: build a MockEnv with custom capabilities."""
    def _make(capabilities=None, z_min_safe=50.0):
        env = MockEnv()
        if capabilities is not None:
            env._capabilities = frozenset(capabilities)
        env.z_min_safe = z_min_safe
        return env
    return _make

def test_suction_tool_only_when_capable(make_env):
    env = make_env(capabilities={"motion.cartesian", "grasp.suction"})
    api = MockApi(env=env)
    tools = {t.name for t in build_robot_tools(api, env=env)}
    assert "suction_on" in tools
    assert "close_gripper" not in tools  # no grasp.parallel
```

### autouse Fixtures

Use sparingly — only for global setup that must happen for every test:

```python
@pytest.fixture(autouse=True)
def reset_trace_state():
    """Ensure no trace state leaks between tests."""
    import jiuwensymbiosis.agent.trace as trace_mod
    saved = trace_mod._active_trace
    trace_mod._active_trace = None
    yield
    trace_mod._active_trace = saved
```

## Pytest Marks

### Selective Execution

```bash
# Run only fast unit tests (CI default)
pytest -m unit

# Run integration tests on the bench
pytest -m integration

# Run everything
pytest

# Filter by name
pytest -k "test_capabilities"
```

## Mocking

### pytest-mock `mocker` fixture (preferred)

```python
def test_rail_logs_rejection(mocker, mock_env):
    mock_env.z_min_safe = 50.0
    log_spy = mocker.patch("jiuwensymbiosis.rails.safety_rail.get_logger")
    rail = SafetyRail(env=mock_env)
    with pytest.raises(ValueError):
        rail.before_tool_call("goto_xyzr", {"x": 0, "y": 0, "z": 10})
    log_spy.return_value.warning.assert_called_once()
```

### Patching module-level symbols

```python
def test_uses_detector_sidecar(mocker):
    mock_init = mocker.patch(
        "jiuwensymbiosis.adapters._common.detector_client.init_detector"
    )
    # ... exercise the path that calls init_detector
    mock_init.assert_called_once()
```

### AsyncMock for async methods

```python
def test_session_starts_sidecar(mocker):
    from unittest import mock
    with mock.patch(
        "jiuwensymbiosis.agent.RobotSession._start_sidecars",
        new_callable=mock.AsyncMock,
    ) as mock_start:
        # ... trigger session enter
        mock_start.assert_awaited_once()
```

## Async Testing

`pytest-asyncio` is configured with `asyncio_mode = "auto"` in
`pyproject.toml` — async test functions need no `@pytest.mark.asyncio`
decorator:

```python
async def test_agent_invoke_with_mock_model(mock_env):
    from jiuwensymbiosis.agent import build_robot_agent, MockModel
    agent = build_robot_agent(env=mock_env, model=MockModel(), tools=[])
    result = await agent.invoke("pick up the box")
    assert result is not None
```

## Test Organization

Mirror the source path in test paths:

| Source | Test |
|--------|------|
| `jiuwensymbiosis/tools/build_robot_tools.py` | `tests/unit_tests/tools/test_build_robot_tools.py` |
| `jiuwensymbiosis/rails/safety_rail.py` | `tests/unit_tests/rails/test_safety_rail.py` |
| `jiuwensymbiosis/api/motion_mixin.py` | `tests/unit_tests/api/test_motion_mixin.py` |
| `jiuwensymbiosis/adapters/piper/api.py` | `tests/unit_tests/adapters/test_piper_api.py` |
| `jiuwensymbiosis/agent/trace.py` | `tests/unit_tests/agent/test_trace.py` |

## Adapter Smoke Tests

Two scripts complement unit tests when working on adapters:

```bash
# Static: every adapter exposes the expected 6 files + symbols
python scripts/validate_adapter.py --module jiuwensymbiosis.adapters.piper

# Runtime: every @robot_tool is callable + JSON-serializable, using MockEnv
python scripts/smoke_test_adapter.py
```

Run both before claiming an adapter change is done.

## pyproject.toml Configuration

Already in place:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "unit: no hardware or GPU required",
    "integration: requires real hardware, GPU, or external services",
]
asyncio_mode = "auto"
filterwarnings = [
    "ignore::DeprecationWarning:pymilvus",
    "ignore::DeprecationWarning:openjiuwen",
    "ignore::pydantic.warnings.PydanticDeprecatedSince20",
]
```

> No coverage gate is configured yet. If you add one, target 80% for
> `jiuwensymbiosis/` core (skip `adapters/` vendor-specific code from the
> gate — it's hard to cover without hardware).
