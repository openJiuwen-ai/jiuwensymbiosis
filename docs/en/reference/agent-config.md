# Complete Agent Configuration Reference

[中文](../../zh/reference/agent-config.md) | [Documentation](../README.md)

This page describes agent settings only. Hardware, detection services, speech backends and model endpoints retain their own configuration blocks.
The canonical `RobotAgentConfig` structure is `execution`, `modules` and `logging`. The CLI, Runtime and workbench share parsing and defaults.

## Minimal configuration and defaults

```yaml
agent:
  execution:
    mode: fastagent
```

Fastagent uses skill composition with action-composition fallback. Disabling skills selects action composition without removing the action dispatcher.
Safety checks and recovery default on, tracking is authorized subject to available capabilities, and tracing defaults off.
Per-step visual feedback and failure diagnosis currently support stepagent only; explicitly enabling them for fastagent is rejected.

When stepagent is selected and module switches are omitted, skills defaults to false and visual_feedback to true. Other defaults stay the same.
Changing the execution mode on an existing Python object does not reset its resolved module values; reparse or explicitly update them when switching modes.

## Complete example

All serializable fields are shown with fastagent defaults. `execution.stepagent` may store step-mode parameters but does not change fast planning.
`workspace: null` shows the complete Python/CLI schema; omit this key in Runtime/workbench submissions, where the binding supplies the workspace.
These software defaults do not replace the body's control constraints.

```yaml
agent:
  execution:
    mode: fastagent
    stepagent:
      mode: hybrid
      max_iterations: 15
      system_prompt: null
      parallel_tool_calls: false
    fastagent:
      max_replans: 2
      tracking:
        enabled: true
        detect_hz: 5.0
        first_target_timeout_s: 8.0
        settle_grip_s: 0.5
        max_re_align_iters: 1
        max_grasp_retries: 1
        servo:
          control_hz: 30.0
          max_lin_step_mm: 6.0
          max_ang_step_deg: 5.0
          pos_tol_mm: 4.0
          ang_tol_deg: 3.0
          settle_ticks: 3
          timeout_s: 20.0
          absolute_timeout_s: 60.0
          progress_pos_epsilon_mm: 0.5
          progress_ang_epsilon_deg: 0.5
          lost_target_grace_s: 3.0
        mask_tracking:
          enabled: true
          min_score: 0.35
          dilation_px: 4
          static_containment: 0.85
          min_visible_ratio: 0.25
          occlusion_area_ratio: 0.98
          max_static_centroid_shift_px: 3.0
          max_static_depth_delta_mm: 10.0
          max_depth_span_mm: 40.0
          min_valid_depth_ratio: 0.5
          motion_min_area_ratio: 0.65
          motion_confirm_frames: 2
          max_motion_step_mm: 35.0
  modules:
    skills:
      enabled: true
    safety:
      enabled: true
    recovery:
      enabled: true
    visual_feedback:
      enabled: false
    diagnosis:
      enabled: false
      max_chars: 1500
      history_steps: 3
      history_kinds:
      - reject
      - recover
    tracing:
      enabled: false
      max_entries: 200
      max_frames: 50
      save_frames: false
      console: false
      dir: null
      capture_loggers:
      - jiuwensymbiosis
  logging:
    level: INFO
    dir: ./logs
    motion_dir: null
  workspace: null
  strict_capabilities: false
```

## Fields

All paths are relative to `agent`. Booleans must be YAML `true/false`, not strings; numeric values must be finite.
Unknown fields, invalid types and out-of-range values are rejected. Disabled modules may retain validated settings for future use.

### Execution `execution`

| Field | Default | Meaning and constraints |
|---|---|---|
| `mode` | `"fastagent"` | Execution mechanism: fastagent or stepagent |
| `stepagent.mode` | `"hybrid"` | Tool strategy: tool, code or hybrid; stepagent decisions only |
| `stepagent.max_iterations` | `15` | Maximum stepagent loop iterations, integer ≥1; does not cap fast sequences |
| `stepagent.system_prompt` | `null` | Override the stepagent system prompt; null uses the built-in prompt; not passed to the fast compiler |
| `stepagent.parallel_tool_calls` | `false` | Parallel stepagent tool calls; incompatible with tracing and subject to builder motion-capability checks |
| `fastagent.max_replans` | `2` | Maximum runtime replans, integer ≥0; zero disables replanning |

### Tracking execution `execution.fastagent.tracking`

| Field | Default | Meaning and constraints |
|---|---|---|
| `detect_hz` | `5.0` | Background detection rate cap in Hz, >0 |
| `first_target_timeout_s` | `8.0` | First-target wait timeout in seconds, >0 |
| `settle_grip_s` | `0.5` | Wait after ordinary fast-runner gripper commands, seconds ≥0; also applies with tracking disabled |
| `max_re_align_iters` | `1` | Maximum post-descent realignment passes, integer ≥0 |
| `max_grasp_retries` | `1` | Maximum retries after unconfirmed tracked grasps, integer ≥0 |
| `enabled` | `true` | Authorize track_detect/track_grasp, subject to capability and interface support |

### Servo execution settings `execution.fastagent.tracking.servo`

| Field | Default | Meaning and constraints |
|---|---|---|
| `control_hz` | `30.0` | Servo loop frequency in Hz, range 1–200 |
| `max_lin_step_mm` | `6.0` | Maximum linear command step per tick in mm, >0 |
| `max_ang_step_deg` | `5.0` | Maximum angular command step per tick in degrees, >0 |
| `pos_tol_mm` | `4.0` | Position tolerance in mm, >0 |
| `ang_tol_deg` | `3.0` | Angular tolerance in degrees, >0 |
| `settle_ticks` | `3` | Consecutive in-tolerance ticks, integer ≥1 |
| `timeout_s` | `20.0` | No-progress timeout in seconds, >0; measurable progress refreshes it |
| `absolute_timeout_s` | `60.0` | Absolute per-phase timeout in seconds, >0; null disables this cap |
| `progress_pos_epsilon_mm` | `0.5` | Position improvement threshold for progress timing in mm, >0 |
| `progress_ang_epsilon_deg` | `0.5` | Angular improvement threshold for progress timing in degrees, >0 |
| `lost_target_grace_s` | `3.0` | Lost-target grace period in seconds, >0 |

### Mask tracking `execution.fastagent.tracking.mask_tracking`

| Field | Default | Meaning and constraints |
|---|---|---|
| `enabled` | `true` | Fixed-camera mask tracking filter; used only when the sampling interface is available |
| `min_score` | `0.35` | Minimum detection score, 0–1 |
| `dilation_px` | `4` | Trusted-mask dilation radius in pixels, integer ≥0 |
| `static_containment` | `0.85` | Static mask containment threshold, 0–1 |
| `min_visible_ratio` | `0.25` | Minimum visible-area ratio, 0–1 |
| `occlusion_area_ratio` | `0.98` | Occlusion area-ratio threshold, between min_visible_ratio and 1 |
| `max_static_centroid_shift_px` | `3.0` | Maximum static centroid shift in pixels, >0 |
| `max_static_depth_delta_mm` | `10.0` | Maximum static depth change in mm, >0 |
| `max_depth_span_mm` | `40.0` | Maximum accepted depth span in mm, >0 |
| `min_valid_depth_ratio` | `0.5` | Minimum valid-depth ratio, 0–1 |
| `motion_min_area_ratio` | `0.65` | Minimum area ratio for movement confirmation, >0 and between min_visible_ratio and 1 |
| `motion_confirm_frames` | `2` | Frames required to confirm movement, integer ≥1 |
| `max_motion_step_mm` | `35.0` | Maximum accepted target displacement in mm, >0 |

### Skills `modules.skills`

| Field | Default | Meaning and constraints |
|---|---|---|
| `enabled` | `true` | Use skill knowledge; defaults true for fast and false for step; does not gate the fast action dispatcher |

### Safety `modules.safety`

| Field | Default | Meaning and constraints |
|---|---|---|
| `enabled` | `true` | Attach agent pre-action checks; does not disable constraints in other layers |

### Recovery `modules.recovery`

| Field | Default | Meaning and constraints |
|---|---|---|
| `enabled` | `true` | Automatic recovery; controls RecoveryRail and fast-runner failure retreat, not resource cleanup |

### Visual Feedback `modules.visual_feedback`

| Field | Default | Meaning and constraints |
|---|---|---|
| `enabled` | `false` | Post-action image feedback; stepagent only; defaults false for fast and true for step; requires an image-capable model |

### Diagnosis `modules.diagnosis`

| Field | Default | Meaning and constraints |
|---|---|---|
| `enabled` | `false` | Failure diagnosis feedback; stepagent only; requires tracing.enabled=true |
| `max_chars` | `1500` | Soft diagnosis text limit, integer ≥1 |
| `history_steps` | `3` | Number of related history entries, integer ≥0 |
| `history_kinds` | `["reject", "recover"]` | string list of event kinds included in related history |

### Tracing `modules.tracing`

| Field | Default | Meaning and constraints |
|---|---|---|
| `enabled` | `false` | Record execution traces; fast traces discrete actions, not every servo tick |
| `max_entries` | `200` | Maximum trace entries, integer ≥1 |
| `max_frames` | `50` | Maximum saved frames, integer ≥0 |
| `save_frames` | `false` | Save action-observation JPEGs when tracing is enabled |
| `console` | `false` | Print live trace summaries when tracing is enabled |
| `dir` | `null` | Trace directory; null uses workspace/traces |
| `capture_loggers` | `["jiuwensymbiosis"]` | Logger names whose WARNING-and-higher records enter the trace |

### Logging `logging`

| Field | Default | Meaning and constraints |
|---|---|---|
| `level` | `"INFO"` | Logging level: DEBUG/INFO/WARNING/ERROR/CRITICAL/NOTSET; WARN/FATAL aliases accepted |
| `dir` | `"./logs"` | Framework log directory; null selects console-only logging |
| `motion_dir` | `null` | Per-run motion artifact root; null uses ./jiuwen_motion_log; applied before connection |

### Run metadata

| Field | Default | Meaning and constraints |
|---|---|---|
| `workspace` | `null` | Workspace for Python/CLI; Runtime uses its binding workspace and rejects this field in the agent block |
| `strict_capabilities` | `false` | Strict capability consistency checking; must be applied before connection; wired by official entry points |

## Dependencies, execution scope and errors

- `diagnosis.enabled=true` requires tracing. Invalid combinations now raise instead of silently disabling diagnosis after a warning.
- Fastagent rejects enabled visual feedback or diagnosis. Official hosts validate before connecting; task dispatch validates again.
- Stepagent parallel tool calls conflict with tracing; the builder also checks motion-capability conflicts.
- Disabling recovery affects RecoveryRail and fast-runner automatic failure retreat, not cancellation, cleanup or action-internal control checks. Internal tracked-grasp retries are independently controlled by `max_grasp_retries`.
- Disabling tracking removes tracking special operations; ordinary actions still run. `settle_grip_s` still applies to ordinary gripper commands.
- `max_replans` caps replanning only. After exhausting the budget the runner still dispatches pending steps through existing execution checks. It is not a step or wall-time limit.
- Tracing covers discrete actions and events, not each servo tick or fast planning before a trace is established.
- Spatial memory and context enhancement are not implemented. `modules.spatial_memory` and `modules.context` are rejected as unknown fields. Foundational execution memory is not optional.

## Overrides and paths

Effective settings combine mode defaults, explicit YAML values and explicit caller overrides. Overrides merge individual leaves recursively.
Omitted CLI flags `--mode`, `--max-iter`, `--control-hz` and `--servo-step-mm` do not replace YAML values.
`--stepagent` / `--mock` select stepagent; `--no-skill` disables skill knowledge only; `--no-visual-feedback` disables per-step image feedback.
The GUI edits grouped paths without implicitly enabling tracing. Task `agent_defaults` fill missing leaves without replacing explicit user values, including null.
Task defaults and the runtime YAML's agent settings are merged before checking module dependencies; for example, diagnosis enabled by a task can use tracing enabled in YAML. Failed merge validation leaves the original configuration unchanged.

Relative logging, motion-artifact and explicit trace directories are relative to the process working directory, not the YAML location.
Workspace precedence remains explicit value, `JIUWENSYMBIOSIS_WORKSPACE`, user settings, then default. Runtime uses the binding workspace.

### Body-specific servo settings in shipped configurations

Framework defaults remain 30 Hz / 6 mm. These YAML files explicitly override `execution.fastagent.tracking.servo`:

| Body | `control_hz` | `max_lin_step_mm` | Configuration sources |
|---|---|---|---|
| SO-101 | 20.0 | 3.0 | `configs/so101/so101.yaml` and the corresponding packaged workbench YAML |
| Piper | 10.0 | 5.0 | `configs/piper/piper.yaml` and the corresponding packaged workbench YAML |

SO-101's 20 Hz / 3 mm is an explicit tuning choice aligned with the driver's default 20 Hz and 60 mm/s limits.
The packaged Piper configuration now shares the source configuration's explicit 10 Hz / 5 mm settings.

These values do not constitute hardware acceptance testing. The agent loop rate is distinct from the hardware dispatch rate: SO-101 independently limits send intervals and velocity.
CLI, Runtime and workbench use the same values for the same YAML. Overriding only `--control-hz` preserves its configured step size.
Custom YAML files must specify any desired tuning themselves; the loader does not select values by body name or derive them from hardware profiles.

## Configuration format

Only the grouped structure on this page is accepted. Flat fields, old Python properties and `exec_config` have been removed; old keys raise errors instead of being converted.
CLI, Runtime and workbench share `RobotAgentConfig.from_dict` for merging and parsing, including defaults.
If a file cannot be loaded or validated, the workbench reports its source and error and blocks execution without substituting defaults.
Changing execution mode refreshes form controls and unset defaults while preserving explicit settings. Explicit modules unsupported by fastagent must be disabled before running.

## Adding future modules

For spatial memory or context enhancement, add a typed field to `ModulesConfig` with `enabled` and module-owned parameters. Optional enhancements should default to disabled.
The builder should create and attach the module only when enabled, skipping its reads, writes and resource initialization when disabled. Validate dependencies centrally in `validate()`.
Core execution state, action preconditions and location freshness remain mandatory. Add switch-behavior tests and document the new fields here; these modules and placeholder switches do not exist yet.

## Python usage

```python
from jiuwensymbiosis.agent import RobotAgentConfig

config = RobotAgentConfig.from_dict(
    {"execution": {"mode": "fastagent"}},
    overrides={"modules": {"tracing": {"enabled": True}}},
)
config.validate(for_execution=True)
# session = ...  # an unconnected RobotSession
config.prepare_session(session)  # apply strict_capabilities / motion_dir before connect
# with session:
#     result = run_robot_task(session, query, config)
```

The execution, modules and logging blocks also accept their dataclasses. `to_dict()` exports effective settings without injected objects.
`model`, `model_spec`, `extra_tools` and `extra_rails` are Python construction dependencies; YAML model parameters stay in the top-level model block.
All four Python-only dependencies raise errors in `from_dict` input.
`build_robot_agent` builds an invoke-able step agent; `build_robot_agent_config` builds a multi-robot sub-agent configuration.
Both use stepagent defaults when `config` is omitted (skills off, visual feedback on). Explicit configurations must set `execution.mode="stepagent"`.
Passing `RobotAgentConfig()` raises `ValueError` because that constructor defaults to fastagent. A mode mismatch fails before constructing models or tools; builders neither rewrite the mode nor recompute module defaults.
Explicit module switches are preserved. See [Framework API](framework-api.md#multi-robot-sub-agent-configuration-scope) for differences in the two builders' supported features.

```python
from jiuwensymbiosis.agent import build_robot_agent, build_robot_agent_config

agent = build_robot_agent(session)  # stepagent defaults
step_config = RobotAgentConfig(
    execution={"mode": "stepagent"},
    modules={"skills": {"enabled": True}, "visual_feedback": {"enabled": False}},
)
agent = build_robot_agent(session, step_config)
subagent = build_robot_agent_config(session, config=step_config)
```

Use `run_robot_task` to dispatch by `execution.mode`. These entry points do not connect the session.

Related: [Framework API](framework-api.md), [Tracing](tracing.md), [CLI](cli.md).
