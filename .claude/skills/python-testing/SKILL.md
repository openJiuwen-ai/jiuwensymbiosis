---
name: python-testing
description: Select and implement hardware-free tests for JiuwenSymbiosis action contracts, capability gates, rails, adapters, planning, and lifecycle changes. Use for project-specific test work and verification planning; basic Python syntax and generic pytest tutorials are outside scope.
---

# Project Testing Workflow

Use [testing rules](../../rules/testing.md) for conventions and
[the change-validation map](../../references/change-validation.md) to choose
existing checks. Test configuration and dependencies live in
`pyproject.toml`; do not copy that configuration into this skill.

## Start from the changed behavior

1. Identify the public behavior, invariant, or failure mode being changed.
   Read its callers and nearby tests before choosing fixtures or patches.
2. Select relevant rows from the validation map. Extend existing assertions
   where possible; a new test should distinguish correct behavior from a
   plausible regression, not restate the implementation.
3. Reproduce a reported bug before fixing it where practical. For a refactor,
   preserve behavior at the public interface and add checks only for gaps.
4. Run focused tests, inspect failures and collection counts, then broaden
   verification when shared behavior or unresolved concerns warrant it.
   Do not expand a documentation-only change into hardware or integration runs.

## Choose a test double at the correct interface

| Need | Existing source |
|---|---|
| In-memory arm and common fixtures | `jiuwensymbiosis/env/mock.py`, `tests/conftest.py` |
| API, Piper driver, dual-arm and detector doubles | `tests/mocks/__init__.py` and the implementation modules it exports |
| Session, callback context, trace sink | `tests/helpers.py`: `make_mock_session`, `FakeCtx`, `RecordingRailSink` |
| LLM replacement | `jiuwensymbiosis/agent/mock_model.py`: `build_mock_model`, configured through `RobotAgentConfig(model=build_mock_model())` |
| Calibration workflow dependencies | `tests/unit_tests/calibration/conftest.py` and nearby workflow tests |

Inspect exports and signatures rather than guessing names such as
`MockEnv` or `MockDriver`. `MockArmEnv` has fixed capabilities and does not
accept a `capabilities` constructor argument. For capability subsets, follow
`TestEnvIntersectionGating` in `tests/unit_tests/tools/test_builder.py`:
use a small `BaseRobotEnv` subclass that expresses the needed contract.
Model both admitted and excluded actions.

Use real deterministic logic behind the interface under test; replace hardware,
LLM, network, and time dependencies at their existing seams. A focused local
fake is appropriate when shared doubles would hide the condition being tested.

## Test safety and execution separately

`SafetyRail` takes a session; its async callback takes a context. This complete
example checks the same rejection for a direct tool name and wrapped dispatch:

```python
import pytest

from jiuwensymbiosis.rails.safety import SafetyRail
from tests.helpers import FakeCtx, make_mock_session


@pytest.mark.parametrize("wrapped", [False, True])
async def test_policy_rejects_both_tool_shapes(wrapped):
    session = make_mock_session()
    rail = SafetyRail(session, z_floor_mm=50.0)
    motion = {"x": 100, "y": 0, "z": 30, "r": 0}
    ctx = FakeCtx(
        tool_name="robot_control" if wrapped else "goto_xyzr",
        tool_args={"action": "goto_xyzr", "params": motion} if wrapped else motion,
    )
    with pytest.raises(ValueError, match="below z_floor"):
        await rail.before_tool_call(ctx)
```

A callback test establishes policy behavior, not that every caller invokes it.
When changing dispatch, also test the affected wiring: rejection occurs before
the motion side effect, inner business failures remain failures, and recovery
is performed by the intended owner. Do not infer callback coverage from a
successful direct API call or from a mocked ability manager.

For logging behavior, use `caplog` and exercise a path that actually logs.
For async dependencies, preserve the awaited interface when replacing them.

## Validate adapters and architecture

Follow the validation map for adapter static/smoke scripts and the affected
body tests. Stub-driver success does not validate real-device timing, physical
reachability, or collision safety. Keep geometry, protocol conversion, unit
conventions, and failure behavior under hardware-free tests where possible.

Check architecture tests alongside the changed behavior when modifying action
contracts, capability gates, state freshness, or dependency direction. Review
known forks and xfails explicitly; passing the remaining cases does not retire
an accepted debt.

## Report evidence

Give the exact commands and their outcomes, including skips and missing
dependencies. Separate static inspection, executed unit tests, and any
authorized integration evidence. Name residual gaps without inventing a
coverage percentage or declaring unexecuted checks passed.
