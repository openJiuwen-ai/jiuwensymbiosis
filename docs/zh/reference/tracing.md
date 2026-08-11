# 执行轨迹参考

> 类别：Reference。内容以治理前原始文档为基线重组。

本页集中查询 Trace 配置、核心数据结构和 JSON 格式。实际操作见[记录和回放执行轨迹](../how-to/use-tracing.md)，内部生命周期见[执行轨迹内部设计](../../../design/tracing.md)。

## 一、配置

`RobotAgentConfig` 的 trace 相关字段（全部默认关闭/保守值）：

| 字段 | 默认 | 说明 |
|------|------|------|
| `enable_tracing` | `False` | 总开关 |
| `trace_max_entries` | `200` | 最多记录步数（超则丢最旧） |
| `trace_max_frames` | `50` | 每次 invoke 最多保存帧数 |
| `trace_save_frames` | `False` | 是否保存 JPEG 帧到 `frames/{run_token}/` |
| `trace_console` | `False` | 是否打印逐轮 dashboard 到 stdout |
| `trace_dir` | `None` | 覆盖 trace 目录（默认 `<workspace>/traces`） |
| `trace_capture_loggers` | `["jiuwensymbiosis"]` | `TraceLogHandler` 挂哪些 logger 前缀 |
| `enable_diagnosis` | `False` | 在线诊断开关；失败步后向下一轮 LLM 注入诊断消息（依赖 `enable_tracing`，关 tracing 时自动禁用并 warning） |
| `diagnosis_max_chars` | `1500` | 诊断消息软上限；超限按「历史→系统状态」顺序丢弃，保当前步 |
| `diagnosis_history_steps` | `3` | 因果链回看步数（同工具名或 rail 事件 kind 命中） |
| `diagnosis_history_kinds` | `("reject","recover")` | 视为与当前失败相关的 `rail_events` kind |
| `log_level` | `"INFO"` | 日志级别（见[日志指南](../how-to/configure-logging.md)） |
| `log_dir` | `"./logs"` | 日志文件目录；`None` 时仅控制台（见[日志指南](../how-to/configure-logging.md)） |

---

## 二、核心抽象

### 数据流总览

```
agent.invoke()
   │
   ├─ before_invoke ──→ 新建 ExecutionTrace（抓 conversation_id/query）
   │      └─ set_sink(trace_rail)  ← 恢复 TraceLogHandler 绑定
   │      └─ [save_frames] 抓「首帧」存 step_000.jpg → trace.initial_frame_path
   │
   ├─ 每轮工具调用:
   │    ├─ before_tool_call ──→ new_entry(tool_name, params, started_at)
   │    │      └─ 挂到 ctx.extra["trace_current_step"]
   │    │      └─ [console] 打印 "第 N 轮: tool(params) …"
   │    ├─ [其它 rail 可能触发 SafetyRail 拒绝 / RecoveryRail 恢复 / VisualFeedback 注入帧]
   │    │      └─ 通过 TraceEventSink 推事件 → 归入当前 entry.rail_events
   │    └─ after_tool_call ──→ 填 duration/output/observation/可选帧
   │      └─ [console] 打印 "✅/❌ 耗时"
   │
   └─ after_invoke ──→ finalize()：写盘 1 次 JSON，sink 置 None（handler 仍挂着，供下次 invoke）

session.disconnect() ──→ close()：finalize + detach_log_handler（彻底清理，无悬挂）
```

### 三层数据结构

#### `TraceEntry`（一步工具调用）

```python
@dataclass
class TraceEntry:
    step: int                          # 1-based 步序
    tool_name: str                     # 解包 robot_control 后的实际动作名
    input_params: dict                # 调用参数
    success: bool                      # 是否成功
    error: Optional[str]               # 失败时的异常信息
    started_at: float                  # 开始时间戳
    duration_s: float                   # 耗时（秒）
    observation: Optional[dict]         # pose/joints/extra 快照（不含原始 rgb/depth）
    frame_path: Optional[str]          # 保存的 JPEG 帧路径（若有）
    output_summary: str                # 截断的工具输出摘要
    rail_events: list[dict]            # 本步内 Rail 触发事件（通知钩子推来）
    log_events: list[dict]             # 本步内 WARNING+ 日志行（TraceLogHandler 捕获）
```

#### `ExecutionTrace`（一次 invoke 的完整轨迹）

```python
@dataclass
class ExecutionTrace:
    conversation_id: str
    robot_name: str
    query: Optional[str]
    started_at: float
    entries: list[TraceEntry]
    trace_log: list[dict]              # 无对应 step 时的日志（trace 级）
    workspace: str
    initial_frame_path: Optional[str] # invoke 开始时抓的「首帧」(step_000.jpg)，仅 save_frames 时有
    # 内部: _pending_events / _step_counter
```

> **每步「前+后」帧对比**：每步只存一张**后帧**（动作完成后观测，`entry.frame_path`），不额外抓前帧——因为连续多步里第 N 步后帧 = 第 N+1 步前帧（中间无动作、环境不变）。只需在 invoke 开始时抓一张**首帧**（`initial_frame_path`），即可让每步都凑出前后对比：step 1 的前帧 = 首帧，step N>1 的前帧 = 上一步后帧。HTML replay 据此把相邻帧并排呈现「动作前→动作后」。首帧占 `max_frames` 预算 1 张。


方法：
- `new_entry(tool_name, input_params, started_at)` — 创建一条目，flush pending 事件。
- `record_rail_event(rail_name, kind, detail, success, step=None)` — 归入当前或指定 step。
- `record_log_event(logger_name, level, msg, ts, step=None)` — 同上，用于日志。
- `to_json()` / `save(traces_dir)` — 序列化与落盘。

#### `TraceRail(AgentRail)`（平行 rail，采集器）

```python
class TraceRail(AgentRail):
    priority = 100  # openjiuwen higher = runs first，保证 before_tool_call 先于 SafetyRail 记录
```

---

## 三、典型 trace JSON 结构

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
      "started_at": 1719207351.4,
      "duration_s": 0.82,
      "observation": {"pose": {"x": 150.0, "y": 0.0, "z": 80.0, "r": 0.0}},
      "frame_path": "/path/traces/frames/conv-1_20260624_105551_693633_149333/step_001.jpg",
      "output_summary": "{\"ok\": true}",
      "rail_events": [],
      "log_events": []
    }
  ],
  "trace_log": [
    {"logger": "jiuwensymbiosis.detector", "level": "WARNING", "msg": "unreachable", "ts": 0.0}
  ],
  "workspace": "/home/user/.jiuwensymbiosis/piper_workspace",
  "initial_frame_path": "/path/traces/frames/conv-1_20260624_105551_693633_149333/step_000.jpg"
}
```

---

## 四、相关文件

| 文件 | 角色 |
|------|------|
| [jiuwensymbiosis/agent/trace.py](../../../jiuwensymbiosis/agent/trace.py) | 本模块实现 |
| [jiuwensymbiosis/agent/trace_html.py](../../../jiuwensymbiosis/agent/trace_html.py) | `render_trace_html()`：trace → 自包含 HTML 渲染器（帧 base64 内嵌） |
| [jiuwensymbiosis/agent/config.py](../../../jiuwensymbiosis/agent/config.py) | `RobotAgentConfig` 的 trace 字段 |
| [jiuwensymbiosis/agent/builder.py](../../../jiuwensymbiosis/agent/builder.py) | `build_robot_agent` 装配 TraceRail + sinks |
| [jiuwensymbiosis/agent/session.py](../../../jiuwensymbiosis/agent/session.py) | `disconnect` 调 `close()` |
| [jiuwensymbiosis/rails/safety.py](../../../jiuwensymbiosis/rails/safety.py) / [recovery.py](../../../jiuwensymbiosis/rails/recovery.py) / [visual_feedback.py](../../../jiuwensymbiosis/rails/visual_feedback.py) | 接收 `trace_sink`，推送 Rail 事件 |
| [jiuwensymbiosis/utils/logging.py](../../../jiuwensymbiosis/utils/logging.py) | `TraceLogHandler`（见[日志指南](../how-to/configure-logging.md)） |
| [jiuwensymbiosis/cli.py](../../../jiuwensymbiosis/cli.py) | `replay` / `replay_html` / `replay_main`（默认 HTML + 打印可点击路径；`--text` 纯文本） |
| [tests/unit_tests/rails/test_trace.py](../../../tests/unit_tests/rails/test_trace.py) | 单元测试 |
