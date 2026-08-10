# Agent 与框架 API 参考

> 类别：Reference。本页以 `jiuwensymbiosis/__init__.py`、`jiuwensymbiosis/agent/` 的公开导出和函数签名为基线。

## 导入入口

常用入口可直接从根包导入：

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

根包还重导出 `AgentRail`、`Tool`、`ToolCard`、`LocalFunction` 和 `ToolOutput` 等 openjiuwen 抽象；这些抽象的行为以上游 API 为准。

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

`build_model(spec=None)` 根据该配置构造 openjiuwen `Model`。`api_base` 不应包含 `/chat/completions`。

## `RobotAgentConfig`

主要字段按用途分组如下：

| 分组 | 字段与默认值 |
| --- | --- |
| 执行 | `mode="hybrid"`、`max_iterations=15`、`parallel_tool_calls=False` |
| 模型 | `model=None`、`model_spec=None`、`system_prompt=None` |
| Rails | `enable_visual_feedback=True`、`enable_safety=True`、`enable_recovery=True`、`enable_skill=False` |
| 扩展 | `extra_tools=None`、`extra_rails=None`、`workspace=None`、`strict_capabilities=False` |
| Trace | `enable_tracing=False`、`trace_max_entries=200`、`trace_max_frames=50`、`trace_save_frames=False`、`trace_console=False`、`trace_dir=None` |
| Diagnosis | `enable_diagnosis=False`、`diagnosis_max_chars=1500`、`diagnosis_history_steps=3` |
| 日志 | `log_level="INFO"`、`log_dir="./logs"` |
| Fast path | `exec_mode="agent"`、`exec_config=None` |

`RobotAgentConfig.from_dict(data)` 从 YAML 的 `agent:` 映射构造配置；未知字段会触发 `TypeError`。

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

| 方法 | 作用 |
| --- | --- |
| `connect()` / `disconnect()` | 幂等地管理 Env、sidecar 和 Trace 生命周期 |
| `globals_provider()` | 返回代码工具每次执行时注入的 `env`、`api`、`np` 等对象 |
| `describe()` | 返回名称以及 Env/API/有效 Capability 摘要 |
| `attach_trace_rail(rail)` | 绑定由 Session 负责最终清理的 TraceRail |

推荐始终使用 `with session:` 管理连接。

## Agent 构建与运行

```python
build_robot_agent(session, config=None) -> Any

build_robot_agent_config(
    session,
    *,
    config=None,
    name=None,
    description=None,
) -> Any

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

- `build_robot_agent` 构造单机器人 DeepAgent，Session 生命周期仍由调用方负责。
- `build_robot_agent_config` 返回用于多机器人顶层 Agent 的 `SubAgentConfig`。
- `run_robot_task` 根据 `config.exec_mode` 选择普通 Agent 或 fast path。
- `run_fast_task` 要求显式传入配置；无法构建 fast path 时返回带 `ok=False` 的结果字典。
