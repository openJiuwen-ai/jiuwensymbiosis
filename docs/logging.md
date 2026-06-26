# 日志使用指南（`jiuwensymbiosis/utils/logging.py`）

本模块是整个 jiuwensymbiosis 框架**唯一的日志配置入口**：一处 `configure_logging` 统一格式 + 可选文件输出，一处 `get_logger` 呼出，外加一个把关键日志推入执行轨迹的 `TraceLogHandler`。

---

## 一、快速上手

### 1.1 在你的代码里记日志

直接用 `get_logger` 拿一个 logger，按需 `debug/info/warning/error`：

```python
from jiuwensymbiosis.utils import get_logger   # 或 from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger(__name__)

logger.info("已连接 CAN %s", can_port)
logger.warning("目标越界，已钳制到 (%.1f, %.1f, %.1f)", x, y, z)
logger.error("使能超时：%s", exc)
```

- `get_logger(__name__)` 与标准库 `logging.getLogger(__name__)` 完全等价——**既有的 `logging.getLogger(__name__)` 调用无需改动**，照样享受统一配置。
- `get_logger()`（无参）会通过 `inspect.currentframe` best-effort 探测调用方模块名，与 `logging.getLogger(__name__)` 惯用法对齐（CPython 专属，失败回退到 `"jiuwensymbiosis"`）。
- **日志格式统一**：全框架走同一个 formatter
  `%(asctime)s %(levelname)s %(name)s: %(message)s`（常量 `DEFAULT_FMT`）。你**不需要**、也不应该自己 `setFormatter` 或 `basicConfig`——否则会和 `configure_logging` 重复挂 handler，导致控制台每条日志打印两次（见 §三.3）。

> ⚠️ **不要在模块顶层做任何全局日志配置**。`build_robot_agent` 会在合适时机调一次 `configure_logging`；重复配置或 `basicConfig` 会造成 handler 叠加。

### 1.2 默认行为：开箱即落盘到 `./logs`

无需任何配置，框架日志就会同时输出到：
- **控制台（stderr）**——一个 `StreamHandler`，统一格式，**不过滤**：`jiuwensymbiosis.*` 和 openjiuwen 用标准库 `logging` 打的日志都会显示，便于调试时看全貌。
- **文件** `<log_dir>/jiuwensymbiosis.log`——`RotatingFileHandler`，单文件 5 MB、保留 3 个备份。默认 `log_dir="./logs"`。**只记录 `jiuwensymbiosis.*` 名空间的日志**（文件 handler 上挂了 `_FrameworkFilter`，挡掉 openjiuwen 冒泡到 root 的标准库日志，保持文件干净）。

默认 `./logs` 是刻意选择的，让框架日志与 Piper 命令日志同处一个 `logs/` 根目录。openjiuwen 自有日志后端因实现原因会落在 `logs/logs/` 下（见下方说明），彼此独立、互不干扰：

```
logs/
├── jiuwensymbiosis.log              ← 【我们的】框架日志（agent / rails / 各模块）
├── motion/<时间戳>/commands.log    ← 【我们的】Piper 每次运行的命令轨迹
└── logs/                            ← 【openjiuwen】自有日志后端（落点见下说明）
    ├── run/jiuwen.log               ← 运行日志
    ├── runner.log                   ← runner 日志
    ├── interface/jiuwen_interface.log
    └── performance/jiuwen_performance.log
```

**日志归属说明**：

| 来源 | 落点 | 是否进 `jiuwensymbiosis.log` |
|------|------|------------------------------|
| `jiuwensymbiosis.*`（我们的 agent/rails/适配器代码） | `logs/jiuwensymbiosis.log` | ✅ 是 |
| Piper 运动命令日志 | `logs/motion/<stamp>/commands.log` | ❌（独立 per-run 文件） |
| openjiuwen 自有日志后端（json/trace_id 那套） | `logs/logs/run/`、`logs/logs/interface/`、`logs/logs/performance/` 下各文件 | ❌（绕过标准库 logging） |
| openjiuwen 用标准库 `logging` 打的日志（如初始化时的 `Registered parser ...`） | 控制台可见，**不落 `jiuwensymbiosis.log`** | ❌（被 `_FrameworkFilter` 挡掉） |

> 简言之：`jiuwensymbiosis.log` 只存我们自己的日志；openjiuwen 的日志在它自己的 `logs/logs/` 子目录里（或仅控制台可见）。控制台则全量显示。

> `./logs` 是**相对当前工作目录**的路径。从不同目录运行，日志落在不同位置——固定运行目录即可。

---

## 二、通过配置控制日志

日志参数通过 `RobotAgentConfig` 的 `log_level` / `log_dir` 两个字段控制，它们最终传给 `build_robot_agent → configure_logging`。两种设置方式：**YAML `agent:` 块**（声明式，推荐）和 **CLI 参数**（临时覆盖）。两者可叠加，CLI 优先级更高。

### 2.1 YAML `agent:` 块（声明式配置）

在任务配置文件（如 `configs/piper/pick_box.yaml`）里写 `agent:` 块：

```yaml
agent:
  log_level: INFO            # 根 logger 级别：INFO / DEBUG / WARNING / ERROR ...
  log_dir: ./logs            # 日志文件目录；设为 null（或省略）则仅控制台输出
```

加载链路：

```
YAML → raw["agent"] → RobotAgentConfig.from_dict(raw["agent"])   # piper_pick_demo.py
                      → build_robot_agent(config=...)
                          → configure_logging(level=config.log_level, log_dir=config.log_dir)
```

- `from_dict` 用 `cls(**data)` 原样透传，YAML 里写什么 `log_level`/`log_dir` 就用什么。
- **未知键会抛 `TypeError`**（[agent/config.py](../jiuwensymbiosis/agent/config.py) `RobotAgentConfig.from_dict`）——拼错（如 `enable_trace` 写成 `enable_trace`）会在加载期立刻报错，而不是被静默忽略。

`log_level` / `log_dir` 字段：

| 字段 | 默认 | 说明 |
|------|------|------|
| `log_level` | `"INFO"` | 根 logger 级别（`logging` 的 level 名或 int） |
| `log_dir` | `"./logs"` | 日志文件目录；`None`/`null` = 仅控制台。默认 `./logs`（openjiuwen 日志因其实现落在 `logs/logs/`，与本目录独立） |

#### 关掉文件落盘（仅控制台）

```yaml
agent:
  log_dir: null      # 或直接省略 agent 块里这行，靠代码里显式 RobotAgentConfig(log_dir=None)
```

#### 调到 DEBUG 抓更详细日志

```yaml
agent:
  log_level: DEBUG
  log_dir: ./logs
```

### 2.2 CLI 参数覆盖（demo 临时调级）

`examples/piper_pick_demo.py` 提供了 `--debug`，它在 `RobotAgentConfig.from_dict(...)` 之后把 `log_level` 改成 `DEBUG`，**优先级高于 YAML**：

```python
agent_cfg = RobotAgentConfig.from_dict(raw.get("agent"))
if args.debug:
    agent_cfg.log_level = "DEBUG"
agent = build_robot_agent(session, config=agent_cfg)
```

```bash
# 临时看 DEBUG 级日志（覆盖 YAML 的 log_level），不必改配置文件
python examples/piper_pick_demo.py --config configs/piper/pick_box.yaml --mock --debug
```

> demo **不在 `main()` 里调 `logging.basicConfig`**——根日志完全由 `build_robot_agent → configure_logging` 接管。这点很重要：`basicConfig` 会另挂一个不被 `configure_logging` 识别的 handler，导致控制台重复打印（见 §三.3）。

### 2.3 与执行轨迹（trace）联动

日志还能进执行轨迹（`enable_tracing` 开启时）。相关配置项在 `agent:` 块里（详见 [docs/trace.md](trace.md)）：

```yaml
agent:
  enable_tracing: true                 # 开启 TraceRail，记录每轮工具调用
  trace_capture_loggers: ["jiuwensymbiosis"]   # 哪些 logger 的 WARNING+ 进 trace
  # capture_log_level 目前固定 WARNING（见 §三.2），不开放配置
```

---

## 三、API 与机制参考

### 3.1 `configure_logging(level="INFO", *, log_dir=None, fmt=DEFAULT_FMT)`

幂等地配置根 logger。**普通开发者通常不直接调用**——`build_robot_agent` 会替你调。仅在你脱离 `build_robot_agent` 独立使用框架时才手动调一次。

```python
from jiuwensymbiosis.utils.logging import configure_logging

configure_logging()                                         # 控制台 + 默认行为
configure_logging(level="DEBUG", log_dir="/var/log/js")     # 控制台 + 指定目录的文件
configure_logging(level="INFO", log_dir=None)               # 仅控制台（关闭文件）
```

**幂等机制**：每个由本模块创建的 handler 都被打上 `_OWNED_TAG = "_jiuwensymbiosis_owned"` 标记。重复调用时：
- StreamHandler：存在则只更新 formatter，不新增。
- FileHandler：`log_dir` 从无→有则新增；从有→无则移除并关闭旧的；有→换路径会先移除旧的再建新的。

```
configure_logging(level, log_dir)
  │
  ├─ root.setLevel(int_level)
  ├─ _owned_handlers() 为空？
  │    ├─ 是 → 新建 StreamHandler（不过滤），打 _OWNED_TAG，加到 root
  │    └─ 否 → 更新已有 owned handler 的 formatter
  └─ log_dir 给定？
       ├─ 是 且 无 owned FileHandler → 新建 RotatingFileHandler(5MB, 3 backup)
       │                                  + 挂 _FrameworkFilter（只放行 jiuwensymbiosis.*）
       └─ 否 且 有 owned FileHandler → 移除并 close
```

> **文件 handler 的过滤范围**：`_FrameworkFilter` 只加在 `RotatingFileHandler` 上，**不加在 StreamHandler 上**——所以 `jiuwensymbiosis.log` 只存我们自己的日志，而控制台仍显示 openjiuwen 等全部来源的日志（调试时看全貌）。openjiuwen 自有的日志后端（写 `logs/run/` 等）根本不走标准库 logging，不受此影响。

### 3.2 `get_logger(name=None)`

`logging.getLogger` 的薄封装，留作未来加结构化字段的单一入口。用法见 §一.1。

### 3.3 `TraceLogHandler`（把 WARNING+ 推入 trace）

把 `WARNING`+ 的日志记录转发给一个绑定的 trace sink（通常是 `TraceRail`）。这是"关键日志进 trace"的实现核心——只要把 handler 挂到对应 logger 上，业务模块里既有的 `logger.warning(...)` 就自动成为 trace 的 `log_events`，**无需改业务代码**。

```python
from jiuwensymbiosis.utils.logging import TraceLogHandler

handler = TraceLogHandler(sink=trace_rail, level=logging.WARNING)
logging.getLogger("jiuwensymbiosis").addHandler(handler)
# 此后 jiuwensymbiosis.* 下任何 logger.warning(...) 都会进入 trace
```

- **捕获级别固定 `WARNING`**：在 `build_robot_agent` 里硬编码（`capture_log_level=_logging.WARNING`），不开放配置。理由见 §四.3。
- **emit 行为**：`sink is None` 时 no-op（可提前构造）；否则组装 `{logger, level, msg, ts}` 调 `sink.record_log_event(...)`；sink 异常被精确类型 `(AttributeError, TypeError, ValueError)` 吞掉——日志 handler 绝不能抛。
- **生命周期**由 `TraceRail` 管理（详见 [docs/trace.md](trace.md)）：`set_sink(sink)` 切换 sink（每个 invoke 开始时绑回、结束后置 None）。

### 3.4 常量

| 常量 | 值 | 用途 |
|------|----|------|
| `DEFAULT_FMT` | `"%(asctime)s %(levelname)s %(name)s: %(message)s"` | 默认 formatter 格式 |
| `_OWNED_TAG` | `"_jiuwensymbiosis_owned"` | 标记本模块创建的 handler，用于幂等识别 |
| `_FrameworkFilter` | `logging.Filter` 子类 | 只放行 `jiuwensymbiosis.*` 记录；挂在文件 handler 上，使 `jiuwensymbiosis.log` 不混入 openjiuwen 日志 |

### 3.5 handler 所有权模型

本模块通过 `_OWNED_TAG` 区分"自己创建的 handler"与"外部（pytest、应用代码）注入的 handler"：

```
root.handlers = [
  StreamHandler(_owned=True)                       ← 不过滤，控制台全量显示
  RotatingFileHandler(_owned=True, _FrameworkFilter)   ← 只记 jiuwensymbiosis.*
  LogCaptureHandler                                ← pytest 注入，configure_logging 不碰
]
```

这保证了：
- `configure_logging` 不会误删 pytest 的日志捕获 handler。
- 重复调用不会堆叠 owned handler。
- `log_dir` 切换时只动 owned FileHandler。

---

## 四、集成点与设计决策

### 4.1 `build_robot_agent` 集成

[agent/builder.py](../jiuwensymbiosis/agent/builder.py) 在构造开头调一次：

```python
configure_logging(level=config.log_level, log_dir=config.log_dir)
```

tracing 开启时还会挂 `TraceLogHandler`（§3.3）。

### 4.2 Piper 命令日志

Piper 驱动的 `_attach_cmd_log_handler`（[adapters/piper/lowlevel.py](../jiuwensymbiosis/adapters/piper/lowlevel.py)）每运行一个时间戳子目录，复用 `configure_logging` 的统一格式 + 同款 formatter，**额外**挂一个 per-run 的 `commands.log` 文件 handler：

```python
from jiuwensymbiosis.utils.logging import _OWNED_TAG, DEFAULT_FMT, configure_logging

configure_logging(level="DEBUG", log_dir=None)   # 统一控制台 + DEBUG
handler = logging.FileHandler(path, mode="w", encoding="utf-8")
handler.setFormatter(logging.Formatter(DEFAULT_FMT, datefmt="%H:%M:%S"))
setattr(handler, _OWNED_TAG, True)   # 标记为 owned，格式与全框架一致
logger.addHandler(handler)
```

**保留的向后兼容环境变量**：

| 环境变量 | 作用 |
|----------|------|
| `JIUWEN_PIPER_CMD_LOG=0` | 禁用 Piper 命令日志 |
| `JIUWEN_PIPER_CMD_LOG_DIR` / `JIUWEN_CMD_LOG_DIR` | 覆盖命令日志输出目录（默认 `./logs/motion`） |

命令日志默认写到 `./logs/motion/<stamp>/commands.log`，与框架日志同处 `logs/` 根目录（openjiuwen 日志因实现原因落在 `logs/logs/`，彼此独立）；`commands.log` 文件名与每运行一个时间戳子目录的结构不变。

### 4.3 为什么 `TraceLogHandler` 默认 `WARNING`？

`INFO` 级别在机器人控制循环里量很大（每步运动、每次检测都 log），全量进 trace 会淹没真正的关键信号。`WARNING`+ 聚焦"需要关注的异常"（home 失败、检测不可达、编码失败等）。目前捕获级别在 `build_robot_agent` 硬编码为 `WARNING`，未作为配置项开放。

### 4.4 为什么 `jiuwensymbiosis.log` 只存我们自己的日志？

`configure_logging` 把文件 handler 挂在 root logger 上，而 Python logging 的传播机制让**所有**子 logger（含 openjiuwen 用标准库打的那部分）的记录都冒泡到 root，被同一个文件 handler 接住。不加过滤的话，openjiuwen 初始化时的一大堆 `Registered parser ...` 会淹没我们的框架日志。

`_FrameworkFilter` 只放行 `jiuwensymbiosis.*` 记录，解决这个噪声问题。设计取舍：

- **只过滤文件、不过滤控制台**：文件是给人/工具按名空间检索的，要干净；控制台是调试时看的，openjiuwen 的初始化线索（如哪个 parser 注册了）有诊断价值，应保留全量。
- **只挡标准库通道的 openjiuwen 日志**：openjiuwen 自有的日志后端（json/trace_id 那套，写 `logs/logs/run/jiuwen.log` 等）根本不经过 root logger，与我们的文件 handler 互不相干——所以过滤与否都不影响它，它一直落在自己的子目录里。

### 4.5 为什么不用 `logging.config.dictConfig`？

`dictConfig` 功能更强但更重，且对"幂等更新已有 handler 的 formatter"支持不直观。本模块用显式的 `_owned_handlers()` + 标记位实现幂等，逻辑更透明、更易测试。

---

## 五、相关文件

| 文件 | 角色 |
|------|------|
| [jiuwensymbiosis/utils/logging.py](../jiuwensymbiosis/utils/logging.py) | 本模块实现 |
| [jiuwensymbiosis/utils/\_\_init\_\_.py](../jiuwensymbiosis/utils/__init__.py) | re-export `configure_logging` / `get_logger` / `TraceLogHandler` / `DEFAULT_FMT` |
| [jiuwensymbiosis/agent/config.py](../jiuwensymbiosis/agent/config.py) | `RobotAgentConfig.log_level` / `log_dir` 字段 + `from_dict`（YAML 透传） |
| [jiuwensymbiosis/agent/builder.py](../jiuwensymbiosis/agent/builder.py) | `build_robot_agent` 调 `configure_logging`；tracing 开启时挂 `TraceLogHandler` |
| [examples/piper_pick_demo.py](../examples/piper_pick_demo.py) | `--debug` 覆盖 `log_level`；演示 YAML `agent:` 块加载链路 |
| [jiuwensymbiosis/adapters/piper/lowlevel.py](../jiuwensymbiosis/adapters/piper/lowlevel.py) | Piper `_attach_cmd_log_handler` 复用 `configure_logging` |
| [docs/trace.md](trace.md) | 执行轨迹设计文档（`TraceLogHandler` 的消费者） |
