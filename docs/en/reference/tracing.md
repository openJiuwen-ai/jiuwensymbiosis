# Execution Tracing Reference

> Category: Reference. The [Chinese source](../../zh/reference/tracing.md) is authoritative.

For operational steps, see [Record and Replay Execution Traces](../how-to/use-tracing.md). Internal lifecycle decisions
are documented in [design/tracing.md](../../../design/tracing.md).

## 1. Configuration

| Field | Default | Meaning |
| --- | --- | --- |
| `enable_tracing` | `False` | Master Trace switch |
| `trace_max_entries` | `200` | Maximum retained tool steps |
| `trace_max_frames` | `50` | Maximum JPEG frames per invoke, including the initial frame |
| `trace_save_frames` | `False` | Save frames under `frames/<run-token>/` |
| `trace_console` | `False` | Print the live per-step dashboard |
| `trace_dir` | `None` | Override `<workspace>/traces` |
| `trace_capture_loggers` | `["jiuwensymbiosis"]` | Logger namespaces captured by `TraceLogHandler` |
| `enable_diagnosis` | `False` | Inject online failure evidence; requires tracing |
| `diagnosis_max_chars` | `1500` | Soft maximum diagnosis length |
| `diagnosis_history_steps` | `3` | Maximum related historical steps |
| `diagnosis_history_kinds` | `("reject", "recover")` | Rail event kinds treated as related history |
| `log_level` | `"INFO"` | Framework logging level |
| `log_dir` | `"./logs"` | Framework file directory; `None` is console-only |

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

Important methods:

- `new_entry(tool_name, input_params, started_at)` creates a 1-based step and attaches unscoped pending events.
- `record_rail_event(..., step=None)` targets the current or explicit step.
- `record_log_event(..., step=None)` targets a step or the trace-level log.
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
| [agent/trace_html.py](../../../jiuwensymbiosis/agent/trace_html.py) | Self-contained HTML renderer |
| [agent/config.py](../../../jiuwensymbiosis/agent/config.py) | Trace and diagnosis configuration |
| [agent/builder.py](../../../jiuwensymbiosis/agent/builder.py) | Rail, sink, and handler assembly |
| [agent/session.py](../../../jiuwensymbiosis/agent/session.py) | Final Trace cleanup ownership |
| [utils/logging.py](../../../jiuwensymbiosis/utils/logging.py) | `TraceLogHandler` |
| [cli.py](../../../jiuwensymbiosis/cli.py) | HTML and text replay commands |
| [test_trace.py](../../../tests/unit_tests/rails/test_trace.py) | Unit tests |
