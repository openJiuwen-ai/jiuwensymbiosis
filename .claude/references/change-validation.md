# Change-to-Validation Map

Use this map when designing, implementing, or reviewing a change. Paths below
are relative to the repository root. Select rows affected by the actual
behavior or dependency changes; this is not a requirement to run every suite.

Architecture requirements come from `AGENTS.md` and the relevant design
documents; implementation and tests establish what is currently enforced.
When they disagree, record the discrepancy rather than silently rewriting
the requirement or treating a passing test as proof of full coverage.

Suite ownership is `tests/unit_tests/` for core/domain behavior and `tests/gui/`
for launcher and workbench behavior. The runtime suite verifies tasks and local
resource admission using fake hardware; it does not establish real-device timing
or vendor SDK cleanup guarantees.

| Change or invariant | Existing verification entry points | What to examine |
|---|---|---|
| Shared action contracts and carrier | `tests/unit_tests/api/test_actions.py`, `tests/unit_tests/api/test_decorators.py` | Contract ownership, accepted parameters, result schema, capability/state tokens; keep metadata tied to its spec |
| Tool exposure and capability intersection | `tests/unit_tests/tools/test_builder.py`, `tests/unit_tests/tools/test_planner_vocabulary.py` | Actions absent from the body's capabilities stay unavailable to planners |
| Cross-body action vocabulary | `tests/unit_tests/test_vocabulary_forks.py` | Body differences do not silently create competing names; inspect the registered bodies and documented accepted forks/xfails |
| Independent action adoption | `tests/unit_tests/api/test_no_bundling.py` | Shared implementations do not implicitly advertise neighboring actions |
| Core algorithm layering | `tests/unit_tests/test_layering.py` | Perception/motion do not import the API layer; inspect new shared types for ownership and dependency direction |
| Planning, observed state, and sensing freshness | `tests/unit_tests/fast/test_sequence_contract.py`, `tests/unit_tests/api/test_world_state.py`, `tests/unit_tests/tools/test_sensing_cache_invalidation.py` | Preconditions, unknown state, bindings, and stale locations remain consistent across planning and execution |
| Skill selection and planning tiers | `tests/unit_tests/fast/test_two_tier_planning.py`, `tests/unit_tests/fast/test_planner_capabilities.py`, `tests/unit_tests/skills/test_skills_capability_gated.py` | Capability gating and the conditions for falling back to action composition |
| Rail policy and attachment | `tests/unit_tests/rails/test_safety.py`, `tests/unit_tests/agent/test_builder_rails.py` | Applicable bounds, invalid inputs, wrapped actions, and capability/config-dependent attachment |
| Fast execution and servo dispatch | `tests/unit_tests/fast/test_ability_exec.py`, `tests/unit_tests/fast/test_run_fast_task.py`, `tests/unit_tests/fast/test_tracking.py` | Failure propagation, executor wiring, per-target policy and cancellation; inspect the real dispatch path as well |
| Recovery and held payload | `tests/unit_tests/rails/test_recovery.py`, `tests/unit_tests/adapters/cruzr/test_recovery_home.py` | Payload preservation, safe retreat, failed home, and duplicate recovery |
| Driver protocol or adapter binding | `tests/unit_tests/env/test_protocol.py`, the affected tests under `tests/unit_tests/adapters/` | Capability slices, units, geometry, adapter-local failure behavior; also use the adapter scripts below |
| Session and subprocess lifecycle | `tests/unit_tests/agent/test_session.py`, `tests/unit_tests/perception/test_detector_sidecar.py` | Idempotent connect/disconnect, partial startup failure, and cleanup ownership |
| Code execution trust boundary | `tests/unit_tests/tools/test_inproc_code.py` | Executor behavior only: these tests do not establish sandboxing or inner-motion policy enforcement |
| Calibration boundaries and publication | `tests/unit_tests/calibration/test_dependency_direction.py`, `tests/unit_tests/calibration/test_workflows.py`, `tests/unit_tests/calibration/test_artifacts.py`, `tests/unit_tests/calibration/test_integration.py` | Layer/import boundaries, acceptance gates, candidate rejection, and adapter reload validation |
| Calibration adapter ports and hand guiding | `tests/unit_tests/adapters/piper/test_calibration_device.py`, `tests/unit_tests/adapters/so101/test_calibration_device.py`, `tests/unit_tests/adapters/so101/test_lowlevel.py` | Body tests invoke the helper in `tests/unit_tests/calibration/test_adapter_conformance.py`; also check torque restoration and end-effector behavior where applicable |
| Persisted diagnostics and traces | `tests/unit_tests/rails/test_trace.py`, `tests/unit_tests/utils/test_logging.py` | Capture configuration, persisted fields, frames, and failure visibility; inspect new payloads for sensitive data |
| GUI plugin discovery and startup | `tests/gui/launcher/test_plugins.py`, `tests/gui/launcher/test_gui_entry.py` | Listing/help should stay light; selected plugin entry points receive normalized options; the old module entry remains a temporary forwarding shim |
| Installed wheel and GUI resources | `tests/gui/packaging/test_wheel.py` | Offline wheel build/install, installed task and GUI entry points outside the checkout, and bundled YAML/icons/default configs; requires the dev build tools |
| Workbench behavior and import boundary | `tests/gui/workbench/unit/`, `tests/gui/workbench/components/` | Workbench state, engines, and views stay within the GUI suite; page imports must not eagerly load calibration |
| Runtime persistence store | `tests/unit_tests/runtime/test_store.py` | Snapshot/event transactions, request ID idempotency/conflict, ordering, and close behavior; this does not establish execution or resource-lifecycle guarantees |
| Runtime job execution and resource admission | `tests/unit_tests/runtime/`, `tests/unit_tests/agent/test_cancel_tracking.py`, `tests/unit_tests/agent/test_session_cleanup.py` | Source/config isolation, cross-process contention, request retries, state/event transactions, bounded artifact reads, pending helpers and failed cleanup; official CLI paths share admission |

## Commands and evidence

Use the project environment from `AGENTS.md`. Run the selected files first,
for example:

```bash
python -m pytest tests/unit_tests/api/test_actions.py tests/unit_tests/tools/test_builder.py
```

After behavior changes to an adapter's public binding or driver integration,
run both checks for that adapter. Piper is the concrete example:

```bash
python scripts/validate_adapter.py --module jiuwensymbiosis.adapters.piper
python scripts/smoke_test_adapter.py --module jiuwensymbiosis.adapters.piper
```

The smoke script uses a stub driver. It checks dispatch and result shape;
it does not establish physical reachability, collision safety, or real-device
timing. Add focused behavior tests for the changed adapter logic.

For broader core verification use `python -m pytest tests/unit_tests/` or
`make test-core`. The GUI suite requires its declared extras and is selected with
`python -m pytest tests/gui/` or `make test-gui`; `make test` runs both
no-hardware suites. Integration and live-state commands require their own hardware/service
conditions; do not run them as automatic follow-ups to a documentation review.

For Python edits, run the configured Ruff checks on changed files and the
relevant type checks. `make check` selects staged files by default (or
`COMMITS=N`); inspect its output because its current recipes tolerate tool
failures. A successful Make exit alone does not establish passing checks.

Record what ran, what passed/failed/skipped, and what was only inspected.
Check test collection counts and explain relevant skips/xfails. For each new
architectural invariant, identify an existing check to extend or explain
the part that still needs human review. A list of tests is not execution evidence.
