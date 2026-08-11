# 记录和回放执行轨迹

> 类别：How-to。内容以治理前原始文档为基线重组。

> LLM 驱动的机器人操作是多轮「感知-规划-执行-观测-反馈」闭环。本模块提供一个**平行 rail** `TraceRail`，通过 openjiuwen 的生命周期钩子采集每轮工具调用的完整信息，落盘为单个 JSON，并支持 CLI 回放——让一次具身 Agent 运行可记录、可持久化、可复盘。

本文集中说明开启、定位和回放轨迹的完整操作。字段、数据结构和 JSON 格式见[执行轨迹参考](../reference/tracing.md)，实现机制和设计取舍见[执行轨迹内部设计](../../../design/tracing.md)。

## 一、设计目标

| 目标 | 说明 |
|------|------|
| **结构化记录** | 每轮：`tool_name` / `input_params` / 输出摘要 / `success`/`error` / `duration_s` / `observation` 快照 / Rail 事件 / 关键日志 |
| **持久化** | 一次 invoke 写一个 JSON 到 `<workspace>/traces/`，视觉帧到 `frames/{run_token}/` |
| **可回放** | `jiuwensymbiosis-replay <trace.json>` 纯文本时间线回放，可选弹窗显示帧 |
| **零侵入** | 不改任何 `@robot_tool`、不改 env、不改其它 rail 的既有行为 |
| **默认关闭** | `enable_tracing=False`，关闭时零开销，不破坏既有部署 |
| **可控开销** | `max_entries` / `max_frames` 截断；帧落盘按帧限 |

---

## 二、快速上手

### 开启 trace

有两种等价方式，**推荐用配置文件**（声明式、无需改代码、可纳入版本管理）。

#### 方式一：配置文件（推荐）

在任务 YAML 里加一个 `agent:` 块即可。它与 `env:`（硬件）、`model:`（模型）、`api_servers:`（检测服务）并列，是 agent 行为的声明式入口；所有字段都可选，缺省即默认关闭：

```yaml
# configs/piper/piper.yaml
agent:
  enable_tracing: true        # 总开关（默认 False）
  trace_save_frames: true     # 保存 JPEG 帧到 traces/frames/{run_token}/
  trace_console: true         # 运行时实时打印逐轮缩略图到 stdout
  trace_max_entries: 200      # 最多记录步数（超出丢最旧）
  trace_max_frames: 50        # 每次 invoke 最多保存帧数
  # log_level: INFO           # 日志级别（见 logging.md）
  # log_dir: ./logs           # 默认写入 ./logs；设为 null 时仅控制台
  # trace_dir: ./traces       # 覆盖 trace 目录（默认 <workspace>/traces）
  # trace_capture_loggers: ["jiuwensymbiosis"]  # TraceLogHandler 捕获哪些 logger 的 WARNING+
  # enable_diagnosis: true    # 在线诊断：失败步后把「当前参数+相关历史+系统状态」回灌下一轮 LLM（依赖 enable_tracing）
  # diagnosis_max_chars: 1500 # 诊断消息软上限；超限先丢历史，保当前步+系统状态
  # diagnosis_history_steps: 3  # 因果链回看步数（同工具或同类 rail 事件）
  # diagnosis_history_kinds: ["reject", "recover"]  # 视为相关的 rail_events kind
```

`build_robot_agent` 会读这个块、装配 `TraceRail`、向三个 rail 注入 sink、挂 `TraceLogHandler`，无需手写额外接线。`agent:` 块**全可选、纯增量**——不写它，既有 YAML 照样按默认（全关）运行。

> 字段名必须与 `RobotAgentConfig` 严格一致（如 `enable_tracing` 不是 `enable_trace`）。拼错会在加载时抛 `TypeError`，而不是静默忽略——这是有意的，避免「配了不生效」的隐蔽坑。

命令行开关（如 `--mode`、`--no-skill`、`--max-iter`、`--workspace`）会覆盖在 `agent:` 块之上，二者不冲突：YAML 定基调、CLI 做临时微调。

#### 方式二：Python 代码

在 `RobotAgentConfig` 构造时直接传字段，等效：

```python
from jiuwensymbiosis.agent.config import RobotAgentConfig

config = RobotAgentConfig(
    enable_tracing=True,
    trace_save_frames=True,
    trace_console=True,
)
agent = build_robot_agent(session, config)
```

`RobotAgentConfig.from_dict(mapping)` 是上述两者的统一底层：它把一个 dict（即 YAML 的 `agent:` 块）喂给 dataclass，自动剥离 `model`/`model_spec`（这两个归 `model:` 块管），未知键抛错。配置文件方式就是 demo 在内部调用它。

### trace 文件在哪

默认目录解析优先级（与 workspace 一致）：

```
显式 config.workspace
  > $JIUWENSYMBIOSIS_WORKSPACE
  > ~/.jiuwensymbiosis/settings.json 里的 "workspace"
  > ~/.jiuwensymbiosis/{session.name}_workspace/      ← 最终默认
```

最典型的落地路径：`~/.jiuwensymbiosis/<机器人名>_workspace/traces/`。目录里：

- **trace JSON**：`{run_token}.json`，每次 invoke 一个。
- **帧图片**（仅 `trace_save_frames=True`）：`traces/frames/{run_token}/step_NNN.jpg`，**每次 invoke 独立子目录**，步号跨运行不互相覆盖。

`run_token` = `{safe_cid}_{时间戳}_{微秒}_{pid}`，与该次 invoke 的 JSON 文件名完全一致——任意历史 trace 引用的帧都永久有效。

### 回放

```bash
jiuwensymbiosis-replay <trace.json>                  # 默认：生成 HTML + 打印可点击路径（不自动开浏览器）
jiuwensymbiosis-replay <trace.json> --text           # 回退纯文本时间线（帧仅显示路径）
```

默认行为：在 trace JSON **同目录**写一个**自包含 HTML**（`{run_token}.html`），每一步的 JPEG 帧以 base64 内嵌进页面，与该步的参数 / 错误 / rail 事件 / 日志融在同一张卡片里，并打印文件路径。HTML 不依赖外部图片文件，可移动/分享；目录不可写时回退到系统临时目录。

`--text` 回退到原来的纯文本时间线，帧只打印路径。

文本时间线输出示例：

```
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

特点：
- HTML 模式：帧与关键事件同卡，base64 内嵌，自包含单文件；路径可点击。
- 文本模式：路径在支持文件链接的终端或 IDE 里可点击打开；`rail_events` 与 `log_events` 分组显示；缺字段退化为 `"?"`。


---
