# Execution Tracing Reference

> Category: Reference. The [Chinese source](../../zh/reference/tracing.md) is authoritative.

For operational steps, see [Record and Replay Execution Traces](../how-to/use-tracing.md). Internal lifecycle decisions
are documented in [design/tracing.md](../../../design/tracing.md).

## 1. Configuration

Paths in the table are relative to `agent`: `modules.tracing.*`, `modules.diagnosis.*` and `logging.*`; see [complete agent configuration](agent-config.md). Diagnosis is stepagent-only and enabling it without tracing is an error.

| Field | Default | Meaning |
| --- | --- | --- |
| `modules.tracing.enabled` | `False` | Master Trace switch |
| `modules.tracing.max_entries` | `200` | Maximum retained tool steps |
| `modules.tracing.max_frames` | `50` | Maximum JPEG frames per invoke, including the initial frame |
| `modules.tracing.save_frames` | `False` | Save frames under `frames/<run-token>/` |
| `modules.tracing.console` | `False` | Print the live per-step dashboard |
| `modules.tracing.dir` | `None` | Override `<workspace>/traces` |
| `modules.tracing.capture_loggers` | `["jiuwensymbiosis"]` | Logger namespaces captured by `TraceLogHandler` |
| `modules.diagnosis.enabled` | `False` | Inject online failure evidence; requires tracing |
| `modules.diagnosis.max_chars` | `1500` | Soft maximum diagnosis length |
| `modules.diagnosis.history_steps` | `3` | Maximum related historical steps |
| `modules.diagnosis.history_kinds` | `("reject", "recover")` | Rail event kinds treated as related history |
| `logging.level` | `"INFO"` | Framework logging level |
| `logging.dir` | `"./logs"` | Framework file directory; `None` is console-only |

## 2. Core abstractions

### Data-flow overview

`before_invoke` creates an `ExecutionTrace`; `before_tool_call` creates the current entry; safety, recovery, visual
feedback, and logging sinks attach evidence; `after_tool_call` completes the entry; and `after_invoke` persists one
JSON file. Session teardown performs final cleanup if normal invoke finalization did not run.

### Three-layer data structure

#### `TraceEntry` (one tool call)

```python
@dataclass
class TraceEntry:
    step: int
    tool_name: str
    input_params: dict
    success: bool
    error: str | None
    started_at: float
    duration_s: float
    observation: dict | None
    frame_path: str | None
    output_summary: str
    rail_events: list[dict]
    log_events: list[dict]
```

`tool_name` is the effective action after unwrapping `robot_control`. `observation` contains pose, joints, and extra but
not raw RGB/depth. Verbose outputs are truncated. Rail events are structured; log events contain captured `WARNING+`
records.

#### `ExecutionTrace` (one complete invoke)

```python
@dataclass
class ExecutionTrace:
    conversation_id: str
    robot_name: str
    query: str | None
    started_at: float
    entries: list[TraceEntry]
    trace_log: list[dict]
    workspace: str
    initial_frame_path: str | None
```

> **Per-step "before + after" frame pairing**: each step stores only one **after-frame** (the observation once the
> action completes, `entry.frame_path`) and never grabs a separate before-frame — across consecutive steps, step N's
> after-frame *is* step N+1's before-frame (nothing moves in between, so the environment is unchanged). Grabbing one
> **initial frame** at the start of the invoke (`initial_frame_path`) is therefore enough to give every step a
> before/after pair: step 1's before-frame is the initial frame, and for step N>1 it is the previous step's
> after-frame. HTML replay uses this to present adjacent frames side by side as "before action → after action". The
> initial frame consumes 1 of the `max_frames` budget.

Important methods:

- `new_entry(tool_name, input_params, started_at)` creates a 1-based step and attaches unscoped pending events.
- `record_rail_event(rail_name, kind, detail, success, step=None)` targets the current or explicit step.
- `record_log_event(logger_name, level, msg, ts, step=None)` targets a step or the trace-level log.
- `run_token()` returns the shared JSON/frame-directory identifier.
- `to_dict()`, `to_json()`, and `save(traces_dir)` serialize the trace.

#### `TraceRail(AgentRail)` (parallel collection Rail)

```python
class TraceRail(AgentRail):
    priority = 100
```

The high priority creates an active step before SafetyRail can reject a call. Lifecycle callbacks are:

| Callback | Effect |
| --- | --- |
| `before_invoke` | Create `ExecutionTrace`, bind log sink, optionally save `step_000.jpg` |
| `before_tool_call` | Unwrap the action and create `TraceEntry` |
| `on_tool_exception` | Mark the active entry failed |
| `after_tool_call` | Fill result, duration, observation, and optional frame |
| `after_invoke` | Save one JSON and unbind the sink |

`finalize()` ends one invoke while keeping the handler installed for another invoke. `close()` finalizes and detaches
the handler at Session teardown.

Event sink protocols:

```python
class TraceEventSink(Protocol):
    def record_rail_event(
        self, *, rail_name: str, kind: str, detail: dict, success: bool
    ) -> None: ...


class StepAwareTraceEventSink(TraceEventSink, Protocol):
    def record_rail_event_at_step(
        self, *, rail_name: str, kind: str,
        detail: dict, success: bool, step: int
    ) -> None: ...
```

Safety and Recovery publish synchronous events to the current step. VisualFeedback may publish later and uses the
step-aware extension to retain the original step association.

## 3. Typical Trace JSON structure

```json
{
  "conversation_id": "conv-1",
  "robot_name": "piper",
  "query": "pick the red box",
  "started_at": 1719207351.3,
  "entries": [
    {
      "step": 1,
      "tool_name": "goto_xyzr",
      "input_params": {"x": 150, "y": 0, "z": 80, "r": 0},
      "success": true,
      "error": null,
      "duration_s": 0.82,
      "observation": {"pose": {"x": 150.0, "y": 0.0, "z": 80.0}},
      "frame_path": "/workspace/traces/frames/<run-token>/step_001.jpg",
      "output_summary": "{\"ok\": true}",
      "rail_events": [],
      "log_events": []
    }
  ],
  "trace_log": [],
  "workspace": "/workspace",
  "initial_frame_path": "/workspace/traces/frames/<run-token>/step_000.jpg"
}
```

## 4. Related files

| File | Role |
| --- | --- |
| [agent/trace.py](../../../jiuwensymbiosis/agent/trace.py) | Trace records, Rail, frames, and persistence |
| [agent/trace_html.py](../../../jiuwensymbiosis/agent/trace_html.py) | `render_trace_html()`: trace → self-contained HTML renderer (frames base64-embedded) |
| [agent/config.py](../../../jiuwensymbiosis/agent/config.py) | The trace fields of `RobotAgentConfig` |
| [agent/builder.py](../../../jiuwensymbiosis/agent/builder.py) | `build_robot_agent` assembles TraceRail + sinks |
| [agent/session.py](../../../jiuwensymbiosis/agent/session.py) | `disconnect` calls `close()` |
| [rails/safety.py](../../../jiuwensymbiosis/rails/safety.py) / [recovery.py](../../../jiuwensymbiosis/rails/recovery.py) / [visual_feedback.py](../../../jiuwensymbiosis/rails/visual_feedback.py) | Receive `trace_sink` and push Rail events |
| [utils/logging.py](../../../jiuwensymbiosis/utils/logging.py) | `TraceLogHandler` (see the [logging guide](../how-to/configure-logging.md)) |
| [cli.py](../../../jiuwensymbiosis/cli.py) | `replay` / `replay_html` / `replay_main` (HTML by default + a clickable path printed; `--text` for plain text) |
| [test_trace.py](../../../tests/unit_tests/rails/test_trace.py) | Unit tests |
