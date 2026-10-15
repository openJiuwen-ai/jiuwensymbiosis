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

完整字段、默认值和依赖见 [Agent 全量配置参考](agent-config.md)。规范结构如下：

| 分组 | 字段与默认值 |
| --- | --- |
| 执行 | `execution.mode="fastagent"`；逐步设置在 `execution.stepagent`，跟踪设置在 `execution.fastagent.tracking` |
| 模型 | Python 注入 `model=None`、`model_spec=None`；逐步提示在 `execution.stepagent.system_prompt` |
| 模块 | `modules.skills/safety/recovery/visual_feedback/diagnosis/tracing`；每个模块用 `enabled` 控制 |
| 扩展 | `extra_tools=None`、`extra_rails=None`、`workspace=None`、`strict_capabilities=False` |
| 日志 | `logging.level="INFO"`、`logging.dir="./logs"`、`logging.motion_dir=None` |

`RobotAgentConfig.from_dict(data, overrides=...)` 从 YAML 的 `agent:` 映射构造配置，递归合并显式覆盖并校验类型。
仅接受分组字段，序列化使用 `to_dict()` 输出同一结构；旧字段会报错。

默认 fastagent 启用 skills/safety/recovery，关闭 visual_feedback/diagnosis/tracing。
Fast path 需要提供 `model_spec`；无论 skills 是否开启，均注册动作分派工具。

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
| `connect()` / `disconnect()` | 幂等地管理 Env、客户端、可选本地托管进程和 Trace 生命周期 |
| `globals_provider()` | 返回代码工具每次执行时注入的 `env`、`api`、`np` 等对象 |
| `describe()` | 返回名称以及 Env/API/有效 Capability 摘要 |
| `attach_trace_rail(rail)` | 绑定由 Session 负责最终清理的 TraceRail |

推荐始终使用 `with session:` 管理连接。

`sidecar_starters` 是沿用的内部资源登记接口，也用于客户端上下文和关闭回调。
检测配置为 `local` 时，Session 启停本地托管子进程（sidecar）；配置为 `remote` 时，
Session 只关闭自有 HTTP 客户端，不停止外部服务，本机手动启动的 HTTP 服务也遵循此规则。

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

- `build_robot_agent` 构造单机器人 DeepAgent，Session 生命周期仍由调用方负责。
- `build_robot_agent_config` 返回用于多机器人顶层 Agent 的 `SubAgentConfig`。
- `run_robot_task` 根据 `config.execution.mode` 选择普通 Agent 或 fast path。
- `run_fast_task` 要求显式传入配置；无法构建 fast path 时返回带 `ok=False` 的结果字典。
- `cancel_token` 是可选的取消令牌（GUI 强停等场景用）；两个运行函数均接受。

两个 builder 省略 `config` 时均使用 stepagent 默认值；显式配置必须满足 `execution.mode="stepagent"`。
`RobotAgentConfig()` 默认是 fastagent，直接传给这两个 builder 会在构建模型和工具前抛 `ValueError`。
模块开关按传入值生效，不会因 builder 的用途而重新计算；需要按模式分发时使用 `run_robot_task`。

### 多机器人子 Agent 的配置范围

`build_robot_agent_config` 使用 `execution.stepagent.mode`、`extra_tools` 构建工具，配置模型、迭代上限和并行开关，
并按能力和开关装配 SafetyRail、RecoveryRail、VisualFeedbackRail 及 `extra_rails`。
`modules.skills.enabled=True` 会添加 `RobotControlTool`，但不会自动附加 `SkillUseRail`。

以下单机器人构建行为目前没有在该函数中实现：生成默认系统提示词、解析或创建工作区、
调用日志配置、装配 TraceRail/DiagnosisRail，以及把 `config.strict_capabilities` 写入 Session。
`execution.stepagent.system_prompt` 按原值传给 `SubAgentConfig`（默认 `None`）。需要这些功能时，由顶层 Agent 的调用方
负责配置；严格能力检查应在连接前通过 `RobotSession(strict_capabilities=True)` 或 Session 属性启用。
不能仅设置子 Agent 的 `modules.tracing.enabled`、`modules.diagnosis.enabled` 等字段就认为对应功能已经开启。

### Fast path 最小配置

下面假设 `session` 已由适配器构造，模型服务可用；Session 连接仍由调用方管理：

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
    result = run_robot_task(session, "抓取红色盒子", config)
```

Fast path 的任务解析和规划直接读取 `model_spec` 中的 HTTP 端点参数。
只提供 `config.model` 会返回 `{"ok": False, "reason": "no_model_spec", ...}`；
自定义或离线模型仅通过 `model` 注入时，应选择 `execution.mode="stepagent"`。
`modules.skills.enabled=False` 使 fast planner 直接组合动作，不影响普通动作执行。
