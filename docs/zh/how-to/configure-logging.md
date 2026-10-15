# 配置和使用日志

Symbiosis 与 openjiuwen 分别控制日志级别。默认保留 Symbiosis 的 `INFO` 执行摘要；openjiuwen 保持其当前策略，只有显式设置 `openjiuwen_level` 才统一调整其终端和文件级别；任务上下文、完整计划、工具参数和结果在 Symbiosis 的 `DEBUG` 日志中。

## 配置

在机器人 YAML 中设置：

```yaml
agent:
  logging:
    level: INFO                         # Symbiosis 终端与主文件
    openjiuwen_level: null              # 默认保留上游策略；显式 WARNING 同时限制终端和文件
    dir: ./logs                         # Symbiosis 文件目录；null 关闭主日志文件
    motion_dir: null                    # 默认 ./jiuwen_motion_log
```

两个级别相互独立。CLI `--debug` 只覆盖 `logging.level=DEBUG`，不会自动开启 openjiuwen 的 INFO。需要上游详细日志时，显式修改 `openjiuwen_level: INFO`，终端和文件一起生效；上游部分 INFO 日志包含请求和上下文。

省略该字段或设置 `null` 时，不调用上游配置接口，不增删其 handler，也不改变 logger/sink 级别。`null` 保留当前状态，不恢复更早的快照；例如显式 WARNING 后再传 `null`，仍保持 WARNING。仓库机器人 YAML 中已有的显式 WARNING 保留，它是该示例的选择，不是字段默认值。

显式设置该字段时，通过上游公开配置接口统一设置默认 logger、per-logger 和 sink 级别，保留后端、输出目标、格式、路由及轮转配置。改变上游级别可能重新打开其日志文件；重复应用相同配置不会重置。旧的终端专用字段 `openjiuwen_console_level` 已移除，请改用 `openjiuwen_level`。

字段取值与校验见 [Agent 全量配置](../reference/agent-config.md)。相对路径按进程工作目录解析。`logging.dir: null` 只关闭 Symbiosis 主文件，不关闭 openjiuwen 文件、Piper 命令文件或执行轨迹。

## 日志归属与输出

| 来源 | 默认级别 / 内容 | 输出位置 |
|---|---|---|
| Symbiosis 核心、CLI、标定、工作台 | INFO：连接状态、执行进度、LLM 请求/完成和耗时；DEBUG：上下文、计划、参数、结果、伺服 tick | stderr 与 `<logging.dir>/jiuwensymbiosis.log` |
| openjiuwen 终端 | 上游当前策略；显式配置后使用 openjiuwen_level | 上游自身的终端 handler / stdout、stderr sink |
| openjiuwen 文件 | 上游当前策略；显式配置后与上游终端使用同一级别 | 上游文件 handler / sink |
| Piper 命令记录 | DEBUG，独立保留运动诊断细节 | `<motion_dir>/<时间戳>/commands.log` |
| 执行轨迹 | 结构化工具调用；按 `capture_loggers` 捕获 WARNING+ | `<workspace>/traces/` 或 `modules.tracing.dir` |

Symbiosis 主日志文件只接收 `jiuwensymbiosis`、`jiuwensymbiosis_gui` 及其子命名空间。标定使用 `jiuwensymbiosis.calibration.*`，任务 CLI 使用 `jiuwensymbiosis.cli.run_task`。两个框架命名空间共用一组 handler，并设置 `propagate=False`，终端与主文件都只接收框架日志。openjiuwen 使用自己的出口。显式设置上游级别时，少数直接使用 `openjiuwen.*` 标准库 logger 的记录，由独立适配层设置 `openjiuwen` 命名空间级别，并在该节点补充终端 handler；传播路径上已有其他控制台 handler（包括宿主 root）时不补充。它们不进入 Symbiosis 主文件。

主日志使用统一格式 `%(asctime)s %(levelname)s %(name)s: %(message)s`，单文件 5 MB，保留 3 个备份。openjiuwen 的格式、目录、轮转和输出目标仍由上游管理；部分版本默认文件位于 `logs/logs/`。显式级别配置统一调整上游日志级别，不重写其日志内容或路径，也不会启用原本关闭的输出目标。

Piper 命令文件沿用同一格式，连接时定位到当前运行目录。同一运行目录内重连采用追加写入，保留此前的命令记录；新运行仍使用独立目录。它不修改全局级别或关闭主日志文件。Piper 的 DEBUG 内容默认只进入 `commands.log`；只有 `logging.level: DEBUG` 时才同时进入主文件和控制台。`JIUWEN_PIPER_CMD_LOG=0` 禁用附加并移除已有命令 handler；目录通过 `logging.motion_dir` 配置。最后一个使用者关闭驱动时释放文件、恢复 logger 原级别（不覆盖外部后续调整），构造失败也会回滚；关闭失败会报告并保留句柄供重试。切换目录失败不会继续写入上一轮文件。

`modules.tracing.console: true` 的实时步骤摘要也通过标准日志输出，遵循 `logging.level`，完整参数和结果只用 DEBUG。Trace JSON 仍按结构化轨迹配置记录参数和结果；日志级别不用于删除轨迹内容。

最终回答是 CLI 的 stdout 用户输出，不受文本日志级别裁剪。fastagent 成功且没有 `output` / `content` 文本时，按 `steps_done` 输出“任务完成，共执行 N 步”；已有文本回答优先。语音模式每条指令输出一次反馈后再交给 TTS；`--tts null` 下仍可见。输入上下文和完整执行结果继续使用 DEBUG。

任务返回 `ok: false` 时，CLI 单独用 ERROR 记录任务原因及失败步骤的序号、操作和原因；语音模式也在生成回复前记录该摘要。这样仅通过 `reason` 或 `steps[].reason` 返回的失败在 INFO/WARNING 配置下仍可见，完整参数、计划和结果继续保留在 DEBUG。

## 初始化与 Python API

官方 CLI 先合并全部命令行覆盖，再构造、校验 Agent 配置，检查 query 后初始化日志，避免有效覆盖被提前校验拦截。CLI 和 Runtime 均在构建/连接会话、语音监听与首次规划之前应用配置。`prepare_session()`、Agent builder 和直接调用 `run_fast_task()` 的路径均会幂等配置，规划失败也能留下日志。缺少 query 时，CLI 向 stderr 报错并返回 2，不创建 Symbiosis 日志文件；adapter 导入之前已完成日志初始化，导入失败仍会落盘。包导入不主动配置上游日志，导入期输出遵循上游策略；显式 YAML 级别在入口读取配置后才生效。适配层通过 `get_log_config_snapshot()` / `configure_log_config()` 配置 default 和 Loguru 后端，无私有接口或生命周期挂钩。上游配置变化时，公开接口会重置其 logger 和输出；之后惰性创建的 logger 使用新级别。

嵌入式宿主还需区分其他依赖的初始化行为：当前依赖链中的 `pymilvus` 首次导入会调用全局 `logging.dictConfig()`，可能关闭已有 handler；这不是日志适配层直接修改 root 造成的，也不由此适配层接管。

```python
from jiuwensymbiosis.agent import RobotAgentConfig

config = RobotAgentConfig.from_dict(raw.get("agent"))
config.prepare_session(session)
with session:
    ...
```

独立使用日志模块时：

```python
from jiuwensymbiosis.utils.logging import configure_logging, get_logger

configure_logging("INFO", log_dir="./logs")  # 默认保留上游配置
logger = get_logger("jiuwensymbiosis.my_module")
logger.info("执行完成，步数=%d", count)
logger.debug("完整参数=%s", params)
```

YAML 的 `openjiuwen_level` 接受级别字符串或 `null`；公开 Python API `configure_logging()` 还接受标准整数级别，如 `logging.WARNING`。`NOTSET` 表示允许所有标准级别，不能用来替代“不干预”的 `None`。

项目模块使用 `get_logger(__name__)`；已有 `logging.getLogger(__name__)` 同样有效，它们返回同一个标准库 logger，记录沿框架命名空间传给统一 handler。业务代码不直接调用 handler。应用脚本及可用 `python -m` 执行的模块使用稳定的 Symbiosis 命名空间，避免日志名称变成 `__main__`。不要另调 `basicConfig()` 或在驱动中配置 root。

`configure_logging()` 只管理 `jiuwensymbiosis` 和 `jiuwensymbiosis_gui` 根节点上带 `_jiuwensymbiosis_owned` 标记的 handler：

- 重复调用更新级别和格式，不增加 handler。
- 控制台与文件 handler 都设置级别，显式 DEBUG 子 logger 不会绕过 INFO 阈值。
- 从目录 A 切换到 B，先准备 B，再统一切换配置并关闭 A；设为 `None` 会关闭文件输出。目录创建或文件打开失败会保留原配置；旧文件关闭失败时新配置已经完整生效，错误仍向调用者报告，并在下次配置时重试清理。
- 不修改 root 级别、handler、formatter 或 filter。外部 handler 保持原样，外部输出策略由宿主负责。
- 仅在显式设置上游级别时，独立适配层才通过公开配置接口统一设置终端和文件级别。后续显式调用上游配置接口会覆盖这项策略，直到再次通过 Symbiosis 显式设置上游级别；适配层不拦截上游重配置。

由于框架日志不再传播到 root，宿主聚合器和 pytest `caplog` 需要直接向 `jiuwensymbiosis` / `jiuwensymbiosis_gui` 附加捕获 handler。Runtime 与 Trace 本来就使用命名空间捕获；仓库的 `caplog` fixture 也已适配。测试调整源头级别时指定 `caplog.at_level("DEBUG", logger="jiuwensymbiosis")`。

两个框架命名空间设置自己的源头级别；显式配置上游时，`openjiuwen` 命名空间也设置独立级别，不依赖宿主 root 的阈值。宿主自己附加的 handler 仍按宿主策略过滤。NOTSET 允许全部标准日志级别；上游 Loguru 后端映射为其支持的最低标准级别 DEBUG。

日志配置作用于进程；运行期间官方 Runtime 串行执行任务。自行嵌入多个并行任务时不能依靠这组进程级配置获得各任务独立的级别和文件目录。

## 兼容性与迁移

调用 `configure_logging()` 后，框架命名空间停止向 root 传播。仅在 root 上采集日志的宿主应把采集 handler 同时附加到两个框架命名空间；采集结束时从相同位置移除：

```python
import logging

for name in ("jiuwensymbiosis", "jiuwensymbiosis_gui"):
    logging.getLogger(name).addHandler(collector)
```

外部 pytest 测试同样需要附加 `caplog.handler`，并在测试结束时移除；调整源头级别时使用 `caplog.at_level("DEBUG", logger="jiuwensymbiosis")`。仓库内的 fixture 已处理这一步。

legacy 标定命令 `scripts/calibrate/calibrate_hand_eye.py`（包括 `--selftest`）的日志由 stdout 改为 stderr，采用带时间、级别、logger 名称的统一格式。原来读取 stdout 的脚本应改读 stderr，并调整文本匹配；若只判断自检是否通过，使用退出码。

## 执行轨迹中的告警

`TraceLogHandler` 把指定 logger 的 WARNING+ 转成 trace 的 `log_events`。默认捕获 `jiuwensymbiosis`，包含其标定子命名空间，不默认捕获 openjiuwen 或工作台：

```yaml
agent:
  modules:
    tracing:
      enabled: true
      capture_loggers: [jiuwensymbiosis]
```

捕获级别固定 WARNING。invoke 结束解除 sink，Session 结束移除 handler。详细数据格式和生命周期见 [执行轨迹参考](../reference/tracing.md) 与 [日志设计](../../../design/logging.md)。
