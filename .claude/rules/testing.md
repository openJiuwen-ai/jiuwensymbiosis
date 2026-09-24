---
paths:
  - "tests/**/*.py"
---

# Testing Rules

- Put deterministic tests for the core and domain packages in
  `tests/unit_tests/`, normally mirroring the source subsystem. Put GUI
  launcher, plugin, and workbench tests in `tests/gui/`; these may need the
  relevant GUI extra but should use fake runtime/hardware. Real serial/CAN/device
  access, cameras, GPU inference, and external services belong in explicitly
  marked integration tests. Do not assume selecting all tests automatically
  skips those paths.
- Read pytest settings and dependencies from
  [pyproject.toml](../../pyproject.toml). Select core tests with
  `python -m pytest tests/unit_tests/` or `make test-core`; select the complete
  GUI suite with `python -m pytest tests/gui/` or `make test-gui` in an
  environment with GUI dependencies installed. `make test` defaults to the
  core suite and requires only `.[dev]`; `make test test-gui` runs both
  no-hardware suites with `.[dev,gui]`. `-m unit` is not equivalent because
  existing tests are not uniformly marked.
- Keep `tests/conftest.py` free of eager core/GUI imports so launcher tests can
  run without importing the core. Reuse its lazy fixtures and the test doubles exported by
  `tests/mocks/__init__.py`, and lifecycle helpers in `tests/helpers.py`.
  Inspect their interfaces before adding a new fake. A focused local fake is
  appropriate when the existing doubles do not express the needed contract.
- Core tests and GUI workbench tests initialize proxy cleanup from their
  suite-level `conftest.py` before importing `openjiuwen`; launcher tests keep
  that dependency boundary lightweight.
- For LLM tests, use `build_mock_model()` via `RobotAgentConfig(model=...)`.
  Do not require real credentials or hardware configuration.
- Async tests run under the configured asyncio auto mode. Await async hooks;
  use `FakeCtx` for rail callbacks and `make_mock_session()` when a rail
  requires a session.
- Patch dependencies where the code under test looks them up. Assert
  observable results, failure behavior, and side effects; use `caplog` for
  logging behavior rather than patching a hypothetical logger factory.
- Isolate files with `tmp_path`. For code that writes `os.environ` directly,
  arrange fixture teardown to restore each variable's original presence and
  value, even when an assertion fails. A monkeypatch operation after mutation
  records the modified state; deleting an already absent key before the call
  does not register an undo either. Snapshot/restore explicitly when needed.
- Cover changed public contracts, capability gates, failure propagation, and
  resource cleanup. Prefer boundary assertions over tests coupled to private
  implementation details. Adapter logic that can use stub drivers remains
  eligible for unit tests; do not exclude all adapters from coverage.
- Use [python-testing](../skills/python-testing/SKILL.md) for the workflow and
  the [change-validation map](../references/change-validation.md) to select
  existing architecture checks. Extend the relevant check when a new
  invariant can be tested; do not add a parallel assertion framework.
