# Record and Replay Execution Traces

> Category: How-to. The [Chinese source](../../zh/how-to/use-tracing.md) is authoritative.

> LLM-driven robot manipulation is a multi-turn "perceive-plan-execute-observe-feedback" loop. This module provides a
> **parallel rail**, `TraceRail`, which collects the complete information of every tool call through openjiuwen's
> lifecycle hooks, persists it as a single JSON, and supports CLI replay — making one embodied-agent run recordable,
> persistable, and reviewable.

This page covers enabling, locating, and replaying traces end to end. For fields, data structures, and the JSON format
see the [Tracing reference](../reference/tracing.md); for the implementation mechanism and design trade-offs see the
[tracing internals](../../../design/tracing.md).

## 1. Design goals

| Goal | Meaning |
|------|------|
| **Structured record** | Per turn: `tool_name` / `input_params` / output summary / `success`/`error` / `duration_s` / `observation` snapshot / Rail events / key logs |
| **Persistence** | One JSON per invoke written to `<workspace>/traces/`, vision frames to `frames/{run_token}/` |
| **Replayable** | `jiuwensymbiosis-replay <trace.json>` replays a text timeline, optionally showing frames |
| **Zero intrusion** | Changes no `@implements`, no env, and no existing behavior of any other rail |
| **Off by default** | `modules.tracing.enabled=False`; zero overhead when off, breaking no existing deployment |
| **Bounded overhead** | `max_entries` / `max_frames` truncate; frame persistence is capped per frame budget |

---

## 2. Quick start

### Enable Trace

There are two equivalent ways; **the configuration file is recommended** (declarative, no code change, version-controllable).

#### Option 1: Configuration file (recommended)

Add an `agent:` block to the task YAML. It sits beside `env:` (hardware), `model:` (model) and `api_servers:`
(detection service) as the declarative entry point for agent behavior; every field is optional and tracing defaults to off:

```yaml
agent:
  modules:
    tracing:
      enabled: true
      save_frames: true
      console: true
      max_entries: 200
      max_frames: 50
      capture_loggers: ["jiuwensymbiosis"]
```

`build_robot_agent` reads this block, assembles the `TraceRail`, injects the sink into three rails, and attaches
`TraceLogHandler` — no extra wiring by hand. The `agent:` block is **entirely optional and purely additive**: leave it
out and tracing stays off; other modules use their own defaults.

> Field names must match the schema (for instance `modules.tracing.enabled`). A misspelling
> raises `TypeError` at load time instead of being silently ignored — deliberately, to avoid the hidden "configured but
> not in effect" trap.

Command-line switches (such as `--mode`, `--no-skill`, `--max-iter`, `--workspace`) layer on top of the `agent:` block
without conflict: YAML sets the baseline, the CLI makes a temporary adjustment.

See [complete agent configuration](../reference/agent-config.md) for all fields and constraints. Diagnosis requires `execution.mode: stepagent` with both `modules.diagnosis.enabled` and `modules.tracing.enabled` enabled.

#### Option 2: Python code

Pass the fields directly when constructing `RobotAgentConfig`; this is equivalent:

```python
from jiuwensymbiosis.agent.config import RobotAgentConfig

config = RobotAgentConfig(
    execution={"mode": "stepagent"},
    modules={"tracing": {"enabled": True, "save_frames": True, "console": True}},
)
agent = build_robot_agent(session, config)
```

`RobotAgentConfig.from_dict(mapping)` is the shared foundation under both paths: it feeds a dict (i.e. the YAML
`agent:` block) into the dataclass, rejects Python-only `model`/`model_spec` keys (YAML model settings belong to the top-level `model:` block), and
raises on unknown keys. The configuration-file path is simply the demo calling it internally.

### Where Trace files are stored

The default directory resolution order (the same as workspace resolution):

```
explicit config.workspace
  > $JIUWENSYMBIOSIS_WORKSPACE
  > "workspace" in ~/.jiuwensymbiosis/settings.json
  > ~/.jiuwensymbiosis/{session.name}_workspace/      ← final default
```

The most typical landing path is `~/.jiuwensymbiosis/<robot-name>_workspace/traces/`. In that directory:

```text
traces/
  {run_token}.json          ← one per invoke
  frames/
    {run_token}/            ← one subdirectory per invoke
      step_000.jpg
      step_001.jpg
      ...
```

- **Trace JSON**: `{run_token}.json`, one per invoke.
- **Frame images** (only with `modules.tracing.save_frames=True`): `traces/frames/{run_token}/step_NNN.jpg`, in a **separate
  subdirectory per invoke**, so step numbers never overwrite each other across runs.

`run_token` = `{safe_cid}_{timestamp}_{microseconds}_{pid}`, exactly matching that invoke's JSON filename — so the
frames referenced by any historical trace stay valid forever.

`step_000.jpg` is the initial frame. A later step's before-frame is the previous step's after-frame, so tracing does not
capture two images for every action.

Set `modules.tracing.dir` to override only the trace output directory. Set `modules.tracing.save_frames: false` when image evidence is not
needed or storage is constrained.

With `modules.tracing.console: true`, each tool call prints a compact start/result line:

```text
[trace] #1 goto_xyzr({'x': 150, 'y': 0, 'z': 80}) …
[trace]   └ ✅ 0.80s
```

This is an operational dashboard, not the persisted source of truth. Use JSON or replay for full parameters, Rail
events, warnings, and frames.

### Replay

```bash
jiuwensymbiosis-replay <trace.json>                  # default: generate HTML + print a clickable path (no browser auto-open)
jiuwensymbiosis-replay <trace.json> --text           # fall back to a plain-text timeline (frames shown as paths only)
```

Default behavior: write a **self-contained HTML** (`{run_token}.html`) in the **same directory** as the trace JSON, with
each step's JPEG frame embedded as base64 and fused into one card together with that step's parameters / error / rail
events / logs, then print the file path. The HTML depends on no external image file, so it can be moved or shared; when
the directory is not writable it falls back to the system temp directory.
Add `--open` to launch the generated HTML in your default browser automatically.

`--text` falls back to the original plain-text timeline, printing frames as paths only.

Example text-timeline output:

```text
=== Execution Trace: conv-1_20260624_105551_693633_149333.json ===
robot=test_robot  conversation=conv-1
query: pick the red box

[  1] ✅ goto_xyzr({"x": 150, "y": 0, "z": 80})
       dur=0.80s
       pose: {'x': 150, 'y': 0, 'z': 80}
[  2] ❌ close_gripper({"force_n": 10})
       dur=1.20s
       error: ValueError: gripper timeout
       rail: [ok] RecoveryRail/recover {'home_ok': True, 'released_ok': True}
       log:  [WARNING] jiuwensymbiosis.rails.recovery: home() retried

2 step(s) recorded.
```

Characteristics:

- HTML mode: frame and key events on one card, base64-embedded, a self-contained single file; the path is clickable.
- Text mode: paths are clickable in terminals or IDEs that support file links; `rail_events` and `log_events` are shown
  in separate groups; a missing field degrades to `"?"`.

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
- `modules.tracing.max_entries` drops the oldest in-memory entries when the cap is exceeded.
- `modules.tracing.max_frames` includes the initial frame.
- Session teardown flushes a pending trace and detaches the warning handler even if normal invoke finalization did not run.

Use the [tracing reference](../reference/tracing.md) for the complete schema and configuration table, and
[design/tracing.md](../../../design/tracing.md) for lifecycle and event-attribution decisions.
