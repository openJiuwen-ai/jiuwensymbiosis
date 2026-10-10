# Agent and Framework API Reference

> Category: Reference. The [Chinese source](../../zh/reference/framework-api.md) is authoritative. This page is based on
> the public exports and function signatures of `jiuwensymbiosis/__init__.py` and `jiuwensymbiosis/agent/`.

This page summarizes the stable Agent construction, configuration, Session, and task-running interfaces.

## Imports

Common entry points are available from the root package:

```python
from jiuwensymbiosis import (
    ModelSpec,
    RobotSession,
    build_model,
    build_robot_agent,
    build_robot_agent_config,
    run_fast_task,
    run_robot_task,
)
from jiuwensymbiosis.agent import RobotAgentConfig
```

The root package also re-exports selected openjiuwen abstractions such as `AgentRail`, `Tool`, `ToolCard`,
`LocalFunction`, and `ToolOutput`; their behavior remains defined by openjiuwen.

## `ModelSpec`

```python
ModelSpec(
    provider="OpenAI",
    api_base="http://127.0.0.1:8110/v1",
    api_key="EMPTY",
    model_name="Qwen/Qwen3-VL-32B-Instruct",
    temperature=0.3,
    max_tokens=2048,
    verify_ssl=False,
    extra_request_kwargs={},
)
```

`build_model(spec=None)` creates the openjiuwen `Model` from that configuration. `api_base` should end at the API root and must not include
`/chat/completions`.

## `RobotAgentConfig`

See [Complete Agent Configuration Reference](agent-config.md) for every field, default and dependency.

| Group | Fields and important defaults |
| --- | --- |
| Execution | `execution.mode="fastagent"`; step settings under `execution.stepagent`, tracking under `execution.fastagent.tracking` |
| Model | Python injection: `model=None`, `model_spec=None`; step prompt under `execution.stepagent.system_prompt` |
| Modules | `modules.skills/safety/recovery/visual_feedback/diagnosis/tracing`, each with an `enabled` switch |
| Extension | `extra_tools=None`, `extra_rails=None`, `workspace=None`, `strict_capabilities=False` |
| Logging | `logging.level="INFO"`, `logging.dir="./logs"`, `logging.motion_dir=None` |

`RobotAgentConfig.from_dict(data, overrides=...)` parses the YAML `agent:` mapping with recursive explicit overrides and type validation.
Only grouped fields are accepted; `to_dict()` emits the same structure. Old fields raise errors.
`parallel_tool_calls=True` is rejected for motion/grasp hardware and cannot be combined with tracing.

Fastagent defaults enable skills/safety/recovery and disable visual_feedback/diagnosis/tracing.
The fast path needs an explicit `model_spec`; the action dispatcher is registered regardless of the skills switch.

## `RobotSession`

```python
RobotSession(
    env,
    api,
    name="robot",
    sidecar_starters=[],
    extra_globals={},
    strict_capabilities=False,
)
```

| Method | Purpose |
| --- | --- |
| `connect()` | Idempotently open registered resources, start optional owned local processes, and connect Env |
| `disconnect()` | Flush tracing, disconnect Env, and close owned clients and local processes |
| `globals_provider()` | Return `env`, `api`, `np`, and extra globals for the code tool |
| `describe()` | Return robot name and Env/Api/effective capabilities |
| `attach_trace_rail(rail)` | Transfer final Trace cleanup ownership to the Session |

The internal `sidecar_starters` interface also registers client contexts and cleanup callbacks.
In detection mode `local`, Session starts/stops a managed local subprocess (sidecar).
In `remote`, it closes only its HTTP client and never stops the external server, including a manually started localhost server.

Prefer context management:

```python
with session:
    result = run_robot_task(session, "pick the red box", config)
```

When `strict_capabilities=True`, Api capabilities missing from the Env cause connection failure. Env-only capabilities
remain a warning because they describe hardware that has no exposed Api tool.

## Agent construction and execution

Agent construction:

```python
build_robot_agent(session, config=None) -> Any

build_robot_agent_config(
    session,
    *,
    config=None,
    name=None,
    description=None,
) -> Any
```

`build_robot_agent()` creates an immediately usable Agent instance. `build_robot_agent_config()` returns the openjiuwen
`SubAgentConfig` for a multi-robot top-level Agent. It supports the subset of `RobotAgentConfig` described below.

Both builders use stepagent defaults when `config` is omitted. Explicit configurations must set `execution.mode="stepagent"`.
`RobotAgentConfig()` defaults to fastagent, so passing it directly raises `ValueError` before constructing models or tools.
Builders preserve supplied module switches without recomputing defaults. Use `run_robot_task` for mode-based dispatch.

### Multi-robot sub-agent configuration scope

`build_robot_agent_config` builds tools from `execution.stepagent.mode` and `extra_tools`, configures the model, iteration limit and parallel
setting, and attaches capability-gated SafetyRail, RecoveryRail, VisualFeedbackRail and `extra_rails`.
`modules.skills.enabled=True` adds `RobotControlTool`, but does not automatically attach `SkillUseRail`.

This function currently does not generate a default system prompt, resolve or create a workspace, configure logging,
attach TraceRail/DiagnosisRail, or propagate `config.strict_capabilities` to the Session. It passes `execution.stepagent.system_prompt`
unchanged to `SubAgentConfig` (default `None`). The caller constructing the top-level Agent must configure those features
when needed. Enable strict capability checks through `RobotSession(strict_capabilities=True)` or the Session attribute
before connecting. Setting the sub-agent's `modules.tracing.enabled` or `modules.diagnosis.enabled` fields alone does not enable those features.

Task execution:

```python
run_robot_task(
    session,
    query,
    config=None,
    *,
    conversation_id=None,
    cancel_token=None,
) -> Any

run_fast_task(
    session,
    query,
    config,
    *,
    conversation_id=None,
    cancel_token=None,
) -> dict
```

- `build_robot_agent` builds a single-robot DeepAgent; the Session lifecycle stays the caller's responsibility.
- `build_robot_agent_config` returns the `SubAgentConfig` used by a multi-robot top-level Agent.
- `run_robot_task` picks the ordinary Agent or the fast path according to `config.execution.mode`.
- `run_fast_task` requires the configuration to be passed explicitly; when the fast path cannot be built it returns a
  result dictionary carrying `ok=False`.
- Both execution functions accept an optional `cancel_token`, used by the GUI and other hard-cancel callers.

The caller owns Session connection; use `with session:` to guarantee cleanup.

### Minimal fast-path configuration

This assumes an adapter has constructed `session` and the model service is available. The caller still owns connection:

```python
config = RobotAgentConfig(
    execution={"mode": "fastagent"},
    model_spec=ModelSpec(
        api_base="http://127.0.0.1:8110/v1",
        api_key="EMPTY",
        model_name="Qwen/Qwen3-VL-32B-Instruct",
    ),
)
with session:
    result = run_robot_task(session, "pick the red box", config)
```

Fast-path task parsing and planning read the HTTP endpoint parameters directly from `model_spec`.
Providing only `config.model` returns `{"ok": False, "reason": "no_model_spec", ...}`. When injecting a custom or offline
model through `model` alone, select `execution.mode="stepagent"`. Setting `modules.skills.enabled=False` selects
action composition directly without disabling ordinary action execution.

Workspace resolution:

For the single-robot builder `build_robot_agent`, workspace selection follows:

1. explicit configuration;
2. `JIUWENSYMBIOSIS_WORKSPACE`;
3. `~/.jiuwensymbiosis/settings.json`;
4. `~/.jiuwensymbiosis/<session-name>_workspace/`.

Tracing and other run artifacts resolve paths from the selected workspace unless their own output directory is set.
