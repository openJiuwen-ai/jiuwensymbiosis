# Agent and Framework API Reference

> Category: Reference. The [Chinese source](../../zh/reference/framework-api.md) is authoritative.

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

`build_model(spec=None)` creates the openjiuwen Model. `api_base` should end at the API root and must not include
`/chat/completions`.

## `RobotAgentConfig`

| Group | Fields and important defaults |
| --- | --- |
| Execution | `mode="hybrid"`, `max_iterations=15`, `parallel_tool_calls=False` |
| Model | `model=None`, `model_spec=None`, `system_prompt=None` |
| Rails | `enable_visual_feedback=True`, `enable_safety=True`, `enable_recovery=True`, `enable_skill=False` |
| Extension | `extra_tools=None`, `extra_rails=None`, `workspace=None`, `strict_capabilities=False` |
| Trace | `enable_tracing=False`, `trace_max_entries=200`, `trace_max_frames=50`, `trace_save_frames=False`, `trace_console=False`, `trace_dir=None` |
| Diagnosis | `enable_diagnosis=False`, `diagnosis_max_chars=1500`, `diagnosis_history_steps=3` |
| Logging | `log_level="INFO"`, `log_dir="./logs"` |
| Fast path | `exec_mode="agent"`, `exec_config=None` |

`RobotAgentConfig.from_dict(data)` consumes the YAML `agent:` mapping. Unknown fields raise `TypeError`.
`parallel_tool_calls=True` is rejected for motion/grasp hardware and cannot be combined with tracing.

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
| `connect()` | Idempotently start sidecars and connect the Env |
| `disconnect()` | Flush tracing, disconnect the Env, and close sidecars |
| `globals_provider()` | Return `env`, `api`, `np`, and extra globals for the code tool |
| `describe()` | Return robot name and Env/Api/effective capabilities |
| `attach_trace_rail(rail)` | Transfer final Trace cleanup ownership to the Session |

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
configuration object for callers that own final Agent construction. Both resolve tools, Rails, system prompt, workspace,
logging, tracing, skills, and capability constraints from the same `RobotAgentConfig`.

Task execution:

```python
run_robot_task(
    session,
    query,
    config=None,
    *,
    conversation_id=None,
) -> Any

run_fast_task(
    session,
    query,
    config,
    *,
    conversation_id=None,
) -> dict
```

`run_robot_task()` selects the configured Agent or fast execution path. `run_fast_task()` executes the deterministic
fast runner directly and returns a dictionary. The caller owns Session connection; use `with session:` to guarantee
cleanup.

Workspace resolution:

Workspace selection follows:

1. explicit configuration;
2. `JIUWENSYMBIOSIS_WORKSPACE`;
3. `~/.jiuwensymbiosis/settings.json`;
4. `~/.jiuwensymbiosis/<session-name>_workspace/`.

Tracing and other run artifacts resolve paths from the selected workspace unless their own output directory is set.
