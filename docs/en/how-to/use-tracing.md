# Record and Replay Execution Traces

> Category: How-to. The [Chinese source](../../zh/how-to/use-tracing.md) is authoritative.

Tracing records each Agent tool call, its parameters, result, duration, observation, Rail events, warnings, and optional
camera frame. It is disabled by default and does not change tool or Env implementations.

## 1. Design goals

Tracing provides structured records, one-file-per-invoke persistence, HTML/text replay, zero changes to robot tools,
default-off behavior, and bounded entry/frame retention.

## 2. Quick start

### Enable Trace

There are two equivalent configuration paths. YAML is recommended for reproducible deployments.

#### Option 1: Configuration file (recommended)

Add an `agent:` block to the task YAML:

```yaml
# configs/piper/piper.yaml
agent:
  enable_tracing: true
  trace_save_frames: true
  trace_console: true
  trace_max_entries: 200
  trace_max_frames: 50
  # trace_dir: ./traces
  # trace_capture_loggers: ["jiuwensymbiosis"]
  # enable_diagnosis: true
```

`build_robot_agent()` creates the `TraceRail`, connects the built-in Rail event sinks, and installs the warning-log
handler. Omitting the block keeps tracing off.

#### Option 2: Python code

The equivalent Python configuration is:

```python
from jiuwensymbiosis.agent import RobotAgentConfig

config = RobotAgentConfig(
    enable_tracing=True,
    trace_save_frames=True,
    trace_console=True,
)
agent = build_robot_agent(session, config)
```

Unknown Agent configuration keys raise `TypeError`; misspellings do not fail silently. CLI options can override the YAML
for one run.

### Where Trace files are stored

Workspace resolution follows this order:

1. explicit `config.workspace`;
2. `JIUWENSYMBIOSIS_WORKSPACE`;
3. `~/.jiuwensymbiosis/settings.json`;
4. `~/.jiuwensymbiosis/<session-name>_workspace/`.

Traces normally appear under `<workspace>/traces/`:

```text
traces/
  <run-token>.json
  frames/
    <run-token>/
      step_000.jpg
      step_001.jpg
      ...
```

Each invoke gets its own run token, JSON file, and frame directory. `step_000.jpg` is the initial frame. A later step's
before-frame is the previous step's after-frame, so tracing does not capture two images for every action.

Set `trace_dir` to override only the trace output directory. Set `trace_save_frames: false` when image evidence is not
needed or storage is constrained.

With `trace_console: true`, each tool call prints a compact start/result line:

```text
[trace] #1 goto_xyzr({'x': 150, 'y': 0, 'z': 80}) …
[trace]   └ ✅ 0.80s
```

This is an operational dashboard, not the persisted source of truth. Use JSON or replay for full parameters, Rail
events, warnings, and frames.

### Replay

Generate a self-contained HTML replay:

```bash
jiuwensymbiosis-replay path/to/trace.json
```

The command writes `<run-token>.html` beside the JSON when possible and prints its path. Images are embedded as base64,
so the HTML can be moved or shared without the original frame directory.
Add `--open` to launch the generated HTML in your default browser automatically.

For a terminal timeline:

```bash
jiuwensymbiosis-replay path/to/trace.json --text
```

The text view shows step status, parameters, duration, errors, observation summaries, Rail events, warning logs, and
frame paths.

Common evidence:

| Evidence | Meaning |
| --- | --- |
| `success=false`, `error=...` | The tool or a pre-tool Rail failed |
| `SafetyRail/reject` | A workspace, Z-floor, or joint-limit check blocked the action |
| `RecoveryRail/recover` | Recovery attempted home and end-effector release after failure |
| `VisualFeedback/inject_frame` | A frame was staged for the next model call |
| `log_events` | `WARNING+` records emitted during this step |
| `trace_log` | Captured records emitted outside an active step |

Raw RGB and depth arrays are never stored in JSON. Observations retain only pose, joints, and lightweight extra fields.

Operational constraints:

- Tracing cannot be combined with `parallel_tool_calls=True`; step attribution uses serial Rail context.
- Motion/grasp sessions reject parallel tool dispatch independently because concurrent physical actions are unsafe.
- `trace_max_entries` drops the oldest in-memory entries when the cap is exceeded.
- `trace_max_frames` includes the initial frame.
- Session teardown flushes a pending trace and detaches the warning handler even if normal invoke finalization did not run.

Use the [tracing reference](../reference/tracing.md) for the complete schema and configuration table, and
[design/tracing.md](../../../design/tracing.md) for lifecycle and event-attribution decisions.
