# Add a GUI plugin

中文说明：[接入一个 GUI 插件](../../zh/how-to/add-gui.md).

This guide covers the plugin contract, shared runtime, and test boundary for adding an independent GUI to
JiuwenSymbiosis. Product-specific interface design belongs to the plugin itself. See the runnable
[example package](../../../examples/gui_plugin/README.md).

## Plugin boundary

A GUI plugin is an independently installed Python distribution registered through the
`jiuwensymbiosis.gui` entry-point group. It uses the lightweight `jiuwensymbiosis_gui.plugin` contract from the
core package, returns a `GuiSpec`, and provides `main(options)` with common `LaunchOptions`:

```python
from jiuwensymbiosis_gui.plugin import GUI_API_VERSION, GuiSpec, LaunchOptions


def describe() -> GuiSpec:
    return GuiSpec(
        key="example",
        display_name="Example GUI",
        api_version=GUI_API_VERSION,
        launch_target="my_gui.app:main",
    )


def main(options: LaunchOptions) -> int:
    ...
```

Declare the entry point in the plugin's `pyproject.toml`. The entry-point name must match `GuiSpec.key`:

```toml
[project.entry-points."jiuwensymbiosis.gui"]
example = "my_gui.plugin:describe"
```

The launcher calls `describe()` while listing and validating plugins, so it should import only the contract and
lightweight modules. Defer NiceGUI, Qt, WebGL, point-cloud, or other UI-specific imports to the selected plugin's
startup path. Put framework dependencies in that distribution's `dependencies` or an extra, not in the core
package. Raise an actionable `StartupError` when an incompatible plugin API is detected.

The launcher discovers installed entry points in the current Python environment. Plugin dependencies can be
declared separately, but one process still has one version of each Python package. If two interfaces require
incompatible versions of the same dependency, install them in separate virtual environments and start each GUI
from its matching environment. The launcher does not create an isolated interpreter per plugin.

## Install and select

From the repository root, install the core development package and the example plugin:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e examples/gui_plugin
python -m jiuwensymbiosis_gui --list-guis
python -m jiuwensymbiosis_gui --gui example \
  --gui-config examples/gui_plugin/jiuwensymbiosis_example_gui/configs/self-check.json \
  --workspace /tmp/jiuwensymbiosis-example --no-browser
```

The workbench plugin separately requires NiceGUI; install `pip install -e ".[gui]"` to use it. Plugins that do
not require NiceGUI do not need that extra. See [GUI configuration and startup](configure-gui.md) and the
[CLI reference](../reference/cli.md) for common options, the default workbench, and the temporary compatibility
entry point.

For actual task execution, point `--config` at a robot runtime YAML:

```bash
python -m jiuwensymbiosis_gui --gui example \
  --config configs/piper/piper.yaml \
  --workspace /tmp/jiuwensymbiosis-example
```

`--gui-config` belongs to the selected GUI and can hold its layout, theme, or plugin-specific options. It does
not replace `--config` or turn a real adapter into a mock. The example's `self-check` only verifies plugin
selection and an idle Runtime lifecycle; it does not connect to a robot. Real task execution requires a valid
`--config` and the runtime dependencies for the selected adapter and model.

After selection, the launcher calls the plugin's `main(options)`. The plugin loads its UI dependencies, creates
the `Runtime`, validates its configuration and prepares a binding, then starts its UI. On shutdown it must call
`Runtime.close()` and wait for active work to perform safe cleanup. A dropped browser connection must not release
hardware resources automatically.

## Execute tasks through the shared Runtime

The interface should submit user commands through the common Runtime instead of building an Agent, Planner,
Rails, or `RobotSession` itself. Different GUIs share task state and local resource admission through Runtime.
Official CLI entry points that have joined admission currently share only local resource admission; they keep
their own execution and output flow and do not write Runtime job records.

In the example below, `start_ui_event_loop` represents the plugin's own UI lifetime. Its page callbacks call
`submit`, `read_job`, and `cancel`, while Runtime stays open for the full UI lifetime:

```python
from uuid import uuid4

from jiuwensymbiosis.runtime import Runtime

runtime = Runtime(workspace=options.workspace)
try:
    binding = runtime.prepare_binding(options.config_path)
    def submit(query, request_id=None):
        return runtime.submit_task(
            binding.binding_id,
            request_id=request_id or uuid4().hex,
            query=query,
        )

    def read_job(job_id, after_seq):
        return {
            "snapshot": runtime.get_job(job_id),
            "events": runtime.read_events(job_id, after_seq=after_seq, limit=128),
        }

    def cancel(job_id):
        return runtime.cancel(job_id)

    start_ui_event_loop(submit, read_job, cancel)  # plugin-specific; handlers return promptly
finally:
    closing = runtime.close(timeout=5.0)
    if not closing["closed"]:
        raise RuntimeError("cleanup is unconfirmed; keep this host stopped and disable automatic restart")
```

`submit()`, `read_job()`, and `cancel()` are short operations for UI handlers. Do not call blocking
`wait_for_job()` from an event-loop handler; it is suitable for command-line scripts and tests. The example waits
again if work is still running during shutdown. If Runtime ultimately returns `closed=false`, it reports an error
and requires the host to disable automatic restart. A `close()` timeout does not release resources, and an
operation with unconfirmed cleanup is not a clean shutdown.

`prepare_binding()` validates and freezes configuration without connecting hardware. Runtime performs resource
admission on each task submission, and only creates a session after admission succeeds. A binding fixes the
effective configuration, source paths, workspace, and device identity so a later page edit or environment change
cannot silently switch the target device. GUIs should use the same standard robot configuration and workspace
semantics.

When no workspace is supplied, Runtime and bindings share `runtime.default_workspace()`. Plugins that display
the default directory should call this API instead of copying the path literal.

For a rerun or an edited configuration, pass both the snapshot and its current source path to
`prepare_binding()`; do not reuse another configuration's source directory. Device-related environment values
are captured when building the effective configuration, and drivers and subprocesses must use those captured
values. For example, Cruzr's `ros_domain_id` defaults to `ROS_DOMAIN_ID` at that time, or 0 when unset.

`request_id` makes network retries idempotent; reusing an ID with different content is a conflict. Distinguish
`ResourceBusyError` (a resource is in use) from `ResourceBlockedError` (cleanup from a previous operation has not
been confirmed). A blocked state requires checking and confirming real cleanup; do not delete a lock record to
force another operation. Cancellation is cooperative, and the application must wait for cleanup after requesting
it.

A terminal phase or `run_finished` describes the execution result. Resource reuse is confirmed separately by
`cleanup.released=true` in the job snapshot. Runtime registers artifacts, saves the result, and removes its task
log handler before releasing admission. A cancelled startup with confirmed cleanup can release normally;
unconfirmed child-process exit or torque recovery stays blocked with the specific reason and operator guidance.

Read events with `read_events(job_id, after_seq, limit)` and continue from `next_seq`. Runtime retains at most the
latest 10,000 events per job; `gap=true` means earlier events have expired, so reload the current job snapshot.
Runtime also exposes `latest_frame()` and `read_artifact()` for opaque preview/trace references: at most 128 step
frames are stored per job, and each artifact read is limited to 32 MiB. Read previews only as needed and bound any
client-side cache.

The read limit does not guarantee UI latency. Display decoding and encoding belong to the plugin. The workbench
currently handles frames synchronously during UI polling; move this work to a display-side thread pool if measured
latency warrants it, keeping it off the core action thread.

Resource admission covers Runtime tasks and official CLI entry points that have joined the common admission
service; CLI execution state and output remain with those entry points and do not enter the Runtime job store.
Maintenance operations intended only for trusted Python workflows should coordinate through the
Runtime maintenance API; do not expose its leases, arbitrary callbacks, or maintenance calls to browser clients.
Browser clients should submit Runtime-managed single tasks and read snapshots, events, and artifacts.

The example HTTP page binds only to `127.0.0.1`, checks `Host` and an exact same-origin `Origin` on writes, and
does not enable CORS. A plugin using another transport must define its listening scope, write-origin checks, and
authentication boundary explicitly; do not extend a local-trust assumption into a network service.

## Dependencies and fault isolation

The entry-point `describe()` function cannot depend on the selected UI framework, so `--list-guis` can list a
plugin when NiceGUI or another optional UI package is not installed. The launcher imports the selected plugin's
`main` and its UI-specific components only after selection. If a selected plugin dependency is missing, display
an actionable install command.

If a new GUI needs a different Python version or conflicting dependency versions, install that GUI and the core
package in a separate virtual environment. Entry-point discovery is scoped to the current interpreter; it does
not import plugins from other environments automatically.

## Test boundary

- `tests/unit_tests/` covers core Agent, Runtime, adapters, admission, and behavior without GUI dependencies; run
  it with `make test-core`.
- `tests/gui/` covers the launcher, installed plugin discovery, and interface/contract behavior requiring GUI
  extras; run it with `make test-gui`.
- `make test` runs both no-hardware suites. Workbench GUI tests require `pip install -e ".[gui]"`.
  `make test-all` also collects integration tests.
- Plugin-specific view/layout tests go in `tests/gui/<plugin>/components/`; other GUI/plugin tests go in
  `tests/gui/<plugin>/unit/`. Runtime, resource admission, and core behavior remain in
  `tests/unit_tests/runtime/`.
- Tests moved from `jiuwensymbiosis.gui` should import `jiuwensymbiosis_gui.workbench`; only compatibility-entry
  tests should execute the old module forwarder. Core unit tests must not import NiceGUI or workbench pages.

The old `python -m jiuwensymbiosis.gui` entry remains temporarily as a migration shim. New installs, docs, and
tests should use `python -m jiuwensymbiosis_gui` or `jiuwensymbiosis-gui`; removal of the shim will be announced
with migration guidance.
