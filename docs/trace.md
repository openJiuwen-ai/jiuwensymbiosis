# 执行轨迹模块（`jiuwensymbiosis/agent/trace.py`）

> LLM 驱动的机器人操作是多轮「感知-规划-执行-观测-反馈」闭环。本模块提供一个**平行 rail** `TraceRail`，通过 openjiuwen 的生命周期钩子采集每轮工具调用的完整信息，落盘为单个 JSON，并支持 CLI 回放——让一次 具身agent 运行可记录、可持久化、可复盘。


---

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
  # log_dir: ./logs           # 不写=仅控制台；写则落盘
  # trace_dir: ./traces       # 覆盖 trace 目录（默认 <workspace>/traces）
  # trace_capture_loggers: ["jiuwensymbiosis"]  # TraceLogHandler 捕获哪些 logger 的 WARNING+
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

## 三、配置

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
| `log_level` | `"INFO"` | 日志级别（见 [logging.md](logging.md)） |
| `log_dir` | `None` | 日志文件目录（见 [logging.md](logging.md)） |

---

## 四、核心抽象

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

## 五、生命周期：三段式设计

handler 的 attach/detach 必须区分"invoke 之间"与"session 终点"。

| 阶段 | 触发 | 动作 | handler 状态 |
|------|------|------|-------------|
| **invoke 间** | `before_invoke` | `set_sink(self)` 恢复绑定 | 挂着、sink 绑定 |
| | `after_invoke` → `finalize()` | 写盘 + `set_sink(None)` | 挂着、sink 解除（下次 invoke 可恢复） |
| **session 终点** | `disconnect` → `close()` | `finalize()` + `detach_log_handler()` | 彻底移除 |
| **重复 build** | `build_robot_agent` | builder 先 purge 旧 `TraceLogHandler` 再挂新 | 防御性清理 |

### 完整方法清单

| 方法 | 职责 | 幂等 |
|------|------|------|
| `attach_log_handler(handler, loggers)` | 绑定 handler，记录它挂在哪些 logger 上 | — |
| `detach_log_handler()` | 从记录的 logger 上 `removeHandler` | ✅（handler=None 直接返回） |
| `finalize()` | 写盘 + `set_sink(None)`（不 detach） | ✅（trace=None 返回 None） |
| `close()` | `finalize()` + `detach_log_handler()` | ✅ |
| `before_invoke(ctx)` | 新建 trace + `set_sink(self)` 恢复 handler | — |
| `after_invoke(ctx)` | `_finalize()`（包 try/except 保护） | — |

---

## 六、两种事件采集机制

本模块用**互补**的两套机制采集 Rail 相关信息，覆盖面互补：

### 机制 1：`TraceEventSink` 通知钩子（结构化、精确）

定义两个 Protocol，让其它 rail 主动推送**真实结果**：**基础** `TraceEventSink`（必选，归入当前 entry）+ **step-aware** `StepAwareTraceEventSink`（继承基础，归入指定 entry）。拆成两个 Protocol 是因为 `@runtime_checkable` 把 body 里声明的方法都当结构必选——若 `record_rail_event_at_step` 和基础方法同在一个 Protocol 里，只实现 4 参接口的旧 sink 的 `isinstance(..., TraceEventSink)` 会变 `False`，mypy 也会拒：

```python
@runtime_checkable
class TraceEventSink(Protocol):
    # 基础接口（必选）：归入当前 entry（entries[-1]）
    def record_rail_event(
        self, *, rail_name: str, kind: str, detail: dict, success: bool,
    ) -> None: ...

@runtime_checkable
class StepAwareTraceEventSink(TraceEventSink, Protocol):
    # step-aware 接口（可选）：归入指定 step 的 entry
    def record_rail_event_at_step(
        self, *, rail_name: str, kind: str, detail: dict, success: bool, step: int,
    ) -> None: ...
```

`TraceRail` 同时实现两者（满足 `StepAwareTraceEventSink`）。调用方按能力分发：sink 有 `record_rail_event_at_step` 且事件带 step → 走 step-aware；否则回退基础接口。**向后兼容**：自定义 sink 只实现基础 `TraceEventSink`（4 参）也能收到所有事件——调用方 duck-type 探测 `record_rail_event_at_step`，缺席就回退，不会因未知 `step` kwarg 抛 `TypeError` 丢事件。

其他 rail 各加 `trace_sink` 构造参数，在关键点调用：

| Rail | 触发点 | 推送内容 | 走哪个接口 |
|------|--------|---------|-----------|
| `SafetyRail` | 拒绝（抛 ValueError 前） | `("SafetyRail", "reject", {tool_name, reason}, success=False)` | 基础（同步，当前 entry） |
| `RecoveryRail` | 恢复后 | `("RecoveryRail", "recover", {tool_name, home_ok, released_ok}, success=home_ok)` | 基础（同步，当前 entry） |
| `VisualFeedbackRail` | flush 帧后（`before_model_call`） | `("VisualFeedback", "inject_frame", {tool_name, frame_path}, success, step)` | step-aware（带暂存 step） |

> **`step` 精确定位**：VisualFeedbackRail 延迟到 `before_model_call` 才 flush，此时 `entries[-1]` 可能已是后续步——所以它在 `after_tool_call` 暂存当时的 `trace_step`（`_PendingFrame.trace_step`），flush 时若 sink 支持 step-aware 就显式传回，事件落到正确的 entry（多 tool calls 一轮迭代也能分别对位）；若 sink 是旧 4 参实现则回退基础接口（归入 `entries[-1]`，精度退化但不丢事件）。同步 rail（Safety/Recovery）不发 step、走基础接口，事件归当前 entry（正确）。

显式指定的 `step` 若已不存在（已被 `trace_max_entries` 淘汰，或尚未创建）则丢弃该事件；只有未指定 step 且当前尚无 entry 的事件才进入 pending、等待首个 entry。这样不会把迟到或无效目标的事件误挂到后续步骤，`trace_log` 也继续只承载其既有日志 schema。

`trace_sink=None` 时零开销。`build_robot_agent` 在 tracing + 对应 rail 都启用时把 TraceRail 注入为 sink。

> **TraceRail 与并行互斥**：`build_robot_agent` / `build_robot_agent_config` 在 `parallel_tool_calls=True` 且 `enable_tracing=True` 时直接 `raise ValueError`。TraceRail 用共享 `ctx.extra` 的 `_TRACE_CURRENT_KEY` 定位当前步、用 `entries[-1]` 兼容旧 sink，两者在并行下都会把事件钉到错步。需要并行就用顺序 trace（关 tracing），需要 trace 就别开并行。

### 机制 2：`TraceLogHandler` 日志捕获

通知钩子只覆盖三个 rail。但 detector、相机、piper 驱动等模块的 `logger.warning(...)` 也有调试价值。`TraceLogHandler`（见 [logging.md](logging.md)）挂到 `trace_capture_loggers`（默认 `["jiuwensymbiosis"]`）指定的 logger 上，把 `WARNING`+ 日志记成 `log_events`。

**互补关系**：
- 钩子：结构化事件（有 `kind`/`detail`/`success` 字段），适合程序化分析。
- handler：裸日志行（有 `level`/`logger`/`msg`），适合覆盖未加钩子的模块。

无当前 step 时，日志落 trace 级 `trace_log`；有当前 step 时落该 step 的 `log_events`。

---

## 七、`robot_control` 透明解包

当用 `RobotControlTool` 时，所有动作走单一 `robot_control` 入口，`action`/`params` 藏在参数里。TraceRail 复用了其它 rail 的同一套 unwrap 模式，使 trace 条目名是实际动作（`goto_xyzr`）而非 `robot_control`：

```python
def _unwrap_robot_control(tool_name, tool_args):
    if tool_name == "robot_control" and isinstance(tool_args, dict):
        action = tool_args.get("action", "")
        params = tool_args.get("params", {})
        if action:
            return str(action), params
    return tool_name, tool_args
```

---

## 八、序列化与落盘

### `_filename_base()` 与 `run_token`

JSON 文件名与帧子目录共用同一个 run token：

```
{safe_cid}_{stamp}_{usec:06d}_{pid}
```

- `safe_cid`：conversation_id 净化（非字母数字→`_`，截 64 字符），空则 `noinv`。
- `stamp`：`%Y%m%d_%H%M%S`（秒级）。
- `usec`：微秒（避免同进程同秒两次 finalize 碰撞）。
- `pid`：进程 ID。

JSON 写到 `traces/{run_token}.json`；帧写到 `traces/frames/{run_token}/step_NNN.jpg`。一次 invoke 写一个文件，`save()` 内部 `mkdir`。

### `_json_safe(obj)` 递归归一

把任意对象转为 JSON 可序列化形式：
- `numpy.ndarray`：>64 元素 → `<ndarray shape=... dtype=...>`；否则 `.tolist()`。
- `numpy.generic` → `.item()`。
- `bytes` → base64 字符串。
- `dict`/`list`/`tuple`/`set` → 递归。
- 有 `__dict__` 的对象（dataclass 等）→ 取非下划线属性。
- 深度 >8 或异常 → `repr()` 截断。

### `_summarise_output(value)`

工具输出超 2000 字符截断，避免巨型 JSON。

---

## 九、builder 集成

`build_robot_agent` 在 tracing 开启时：

```python
trace_rail = TraceRail(
    session,
    workspace=workspace,
    max_entries=config.trace_max_entries,
    max_frames=config.trace_max_frames,
    save_frames=config.trace_save_frames,
    console=config.trace_console,
    capture_loggers=tuple(config.trace_capture_loggers),
    traces_dir=Path(trace_dir),  # config.trace_dir 优先，否则 <workspace>/traces
)
_inject_trace_sinks(rails, trace_rail)  # 注入 trace_sink + frame_sink 到三 rail
log_handler = _attach_trace_log_handlers(trace_rail, loggers, level)  # purge 旧 handler 后挂新
trace_rail.attach_log_handler(log_handler, tuple(loggers))
session._trace_rail = trace_rail  # session.disconnect 时调 close()
rails.insert(0, trace_rail)  # 列表位置不决定执行顺序；TraceRail.priority=100 保证先执行
```

**frame_sink 注入**：当 `VisualFeedbackRail` 启用**且** `trace_save_frames=True` 时，`_make_frame_sink(trace_rail)` 返回一个 `(rgb, tool_name) -> path` 的 callable，让 VisualFeedbackRail 注入到 agent 上下文的帧**同时**落盘到 trace 的 `frames/`——保证注入帧与落盘帧是**同一帧**。`trace_save_frames=False`（默认）时**不安装** `frame_sink`（且显式置 `None` 防复用残留），`TraceRail.save_frame_for_sink` 自身也再判一次 `save_frames`——双层防护，确保用户关掉帧落盘就不会有 JPEG 写盘。

---

## 十、设计决策与权衡

### 为什么 trace 默认关闭？

机器人控制循环对延迟敏感。trace 涉及每步 `get_observation()` 快照、可能的帧编码、JSON 序列化——虽是 best-effort，但不该在默认路径上引入。`enable_tracing=False` 保证既有部署零开销。

### 为什么 TraceRail `priority=100`？

openjiuwen 当前 `register_callback` 语义是 **higher = runs first**（回调列表按 `priority` 倒序执行），而默认 `AgentRail.priority=50`。`TraceRail.priority=100` 让 TraceRail 的 `before_tool_call` 先于 SafetyRail 运行（先创建当前 step，安全拒绝事件才能落到正确 entry），也让 `after_tool_call` 先于 VisualFeedbackRail 运行（先记录动作后观测，再让反馈 rail 暂存/flush 帧事件）。

### 为什么采集与持久化解耦？

内存累积，一次 invoke 写一次盘。若每步写盘，长 episode 下磁盘 I/O 会影响控制循环时序。`after_invoke` 一次性 flush，I/O 降到最小。

### 为什么 `observation` 不含原始 rgb/depth？

RGB 帧可能数 MB，放 JSON 会让 trace 文件膨胀到不可用。帧单独存 `frames/` 并在 entry 记 `frame_path` 引用；observation 只留 `pose`/`joints`/`extra` 等小数据。

### 为什么 `finalize()` 不 detach，而 `close()` 才 detach？

`finalize()` 是"invoke 之间"用的——一次 invoke 结束写盘，但 handler 要留着给下一次 invoke 用（`before_invoke` 会 `set_sink(self)` 重新绑定）。如果 finalize 就 detach，第二次 invoke 的日志就捕获不到。

`disconnect` 是 session 真正终点，应彻底清理：既写盘（finalize）又把 handler 从 logger 上移除（detach），否则 `TraceLogHandler` 会永远挂在 `jiuwensymbiosis` logger 上，进程长期运行时是资源悬挂。所以 `close = finalize + detach`。

### 为什么 `after_tool_call` 不重新推断已被 `on_tool_exception` 标记的失败？

openjiuwen 的 `@rail` 装饰器在 `before_tool_call` 抛异常（如 SafetyRail 拒绝）时，会先触发 `ON_TOOL_EXCEPTION`（except 块）、再在 `finally` 触发 `AFTER_TOOL_CALL`，且二者共享同一个 `ctx.extra`。`on_tool_exception` 先正确记下 `success=False`/`error`；若 `after_tool_call` 无脑重算 `success`（无 `tool_result` 时默认 `True`），会把失败覆盖回成功——破坏了 trace 记录失败的核心价值。因此 `after_tool_call` 进入时若 entry 已失败，就跳过 success 重算，只补 `output_summary`/`observation`/帧。

---

## 十一、典型 trace JSON 结构

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

## 十二、相关文件

| 文件 | 角色 |
|------|------|
| [jiuwensymbiosis/agent/trace.py](../jiuwensymbiosis/agent/trace.py) | 本模块实现 |
| [jiuwensymbiosis/agent/trace_html.py](../jiuwensymbiosis/agent/trace_html.py) | `render_trace_html()`：trace → 自包含 HTML 渲染器（帧 base64 内嵌） |
| [jiuwensymbiosis/agent/config.py](../jiuwensymbiosis/agent/config.py) | `RobotAgentConfig` 的 trace 字段 |
| [jiuwensymbiosis/agent/builder.py](../jiuwensymbiosis/agent/builder.py) | `build_robot_agent` 装配 TraceRail + sinks |
| [jiuwensymbiosis/agent/session.py](../jiuwensymbiosis/agent/session.py) | `disconnect` 调 `close()` |
| [jiuwensymbiosis/rails/safety.py](../jiuwensymbiosis/rails/safety.py) / [recovery.py](../jiuwensymbiosis/rails/recovery.py) / [visual_feedback.py](../jiuwensymbiosis/rails/visual_feedback.py) | 接收 `trace_sink`，推送 Rail 事件 |
| [jiuwensymbiosis/utils/logging.py](../jiuwensymbiosis/utils/logging.py) | `TraceLogHandler`（见 [logging.md](logging.md)） |
| [jiuwensymbiosis/cli.py](../jiuwensymbiosis/cli.py) | `replay` / `replay_html` / `replay_main`（默认 HTML + 打印可点击路径；`--text` 纯文本） |
| [tests/unit_tests/rails/test_trace.py](../tests/unit_tests/rails/test_trace.py) | 单元测试 |
