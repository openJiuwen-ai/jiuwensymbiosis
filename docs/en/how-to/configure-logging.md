# Configure and Use Logging

> Category: How-to. The [Chinese source](../../zh/how-to/configure-logging.md) is authoritative.

JiuwenSymbiosis centralizes console, rotating-file, Piper command, and Trace-event logging while remaining compatible
with existing `logging.getLogger()` calls.

## 1. Quick start

### 1.1 Log from application code

New code should use the shared entry point:

```python
from jiuwensymbiosis.utils import get_logger

logger = get_logger(__name__)
logger.info("robot connected")
logger.warning("detector unavailable")
```

Legacy `logging.getLogger(__name__)` calls remain valid because configuration is installed on the root logger. Do not
call `basicConfig()` or install global handlers at module import time; that can duplicate console output.

### 1.2 Default behavior: write to `./logs` out of the box

`build_robot_agent()` configures a uniform console stream and, by default, a rotating file at
`./logs/jiuwensymbiosis.log`. The file rotates at 5 MB and retains three backups. Repeated configuration updates
framework-owned handlers instead of stacking duplicates.

| Output | Contents | Default location |
| --- | --- | --- |
| Console | All standard-library logs propagated to root | stderr |
| Framework rotating file | Only `jiuwensymbiosis.*` records | `./logs/jiuwensymbiosis.log` |
| Piper command log | Per-run Piper motion records | `./logs/motion/<timestamp>/commands.log` |
| openjiuwen backend | openjiuwen-owned logs | `./logs/logs/...` |

The framework file uses `_FrameworkFilter`, so openjiuwen initialization records may appear on the console without
polluting `jiuwensymbiosis.log`. Relative paths are resolved from the current working directory.

## 2. Control logging through configuration

`RobotAgentConfig.log_level` and `log_dir` flow through `build_robot_agent()` into `configure_logging()`. Use YAML for
the deployed default and the demo CLI for a one-run override.

### 2.1 Declarative YAML `agent:` block

```yaml
agent:
  log_level: INFO
  log_dir: ./logs
```

| Field | Default | Meaning |
| --- | --- | --- |
| `log_level` | `"INFO"` | Root logger level name or integer |
| `log_dir` | `"./logs"` | Framework log directory; `null` means console only |

Unknown `agent` keys raise `TypeError` during configuration loading rather than being silently ignored.

#### Disable file output and keep console only

```yaml
agent:
  log_dir: null
```

#### Switch to DEBUG for detailed diagnostics

```yaml
agent:
  log_level: DEBUG
  log_dir: ./logs
```

### 2.2 CLI override for temporary demo debugging

`examples/piper_pick_demo.py --debug` changes the loaded Agent configuration to `DEBUG`, taking precedence over YAML:

```bash
# Temporarily enable DEBUG without changing the YAML log_level
python examples/piper_pick_demo.py \
  --config configs/piper/piper.yaml --mock --debug
```

Keep the normal deployment level in YAML and use this flag only for the current run.

### 2.3 Link warnings to execution traces

```yaml
agent:
  enable_tracing: true
  trace_capture_loggers: ["jiuwensymbiosis"]
```

When tracing is enabled, `TraceLogHandler` forwards `WARNING` and higher records from configured namespaces into the
active `ExecutionTrace`. Records emitted during a tool step appear in that entry's `log_events`; records with no active
step appear in `trace_log`. The capture threshold is intentionally fixed at `WARNING`.

See the [execution-tracing reference](../reference/tracing.md) for the event schema and lifecycle.

## 3. API and mechanism reference

### 3.1 `configure_logging(level="INFO", *, log_dir=None, fmt=DEFAULT_FMT)`

```python
from jiuwensymbiosis.utils.logging import configure_logging

configure_logging()
configure_logging(level="DEBUG", log_dir="/var/log/js")
configure_logging(level="INFO", log_dir=None)
```

The function sets the root level, maintains one framework-owned `StreamHandler`, and adds or removes the owned
`RotatingFileHandler` according to `log_dir`. Once an owned file handler exists, changing directly from one non-null
directory to another does not replace its path. Call `configure_logging(log_dir=None)` first and then call it with the
new directory, or restart the process. Only handlers marked with `_OWNED_TAG` are changed; handlers installed by pytest
or a host application remain untouched.

If duplicate lines appear, check for host handlers or another `basicConfig()` call. If the file is empty while console
output exists, verify that `log_dir` is non-null and the logger name starts with `jiuwensymbiosis`.

### 3.2 `get_logger(name=None)`

```python
get_logger(name: str | None = None) -> logging.Logger
```

This is a thin `logging.getLogger` entry point. With no name it attempts to infer the caller module and falls back to
`"jiuwensymbiosis"`; passing `__name__` explicitly is clearest in reusable modules.

### 3.3 `TraceLogHandler`: forward WARNING+ records to Trace

```python
handler = TraceLogHandler(sink=trace_rail, level=logging.WARNING)
handler.set_sink(trace_rail_or_none)
# Subsequent WARNING+ records from the selected loggers enter the active Trace
```

The Agent builder owns this handler. A missing sink is a no-op, and expected sink errors are contained so logging cannot
fail a robot task. If warnings do not appear in Trace, confirm `enable_tracing`, `trace_capture_loggers`, and a record
level of at least `WARNING`.

### 3.4 Constants

| Constant | Value or role |
| --- | --- |
| `DEFAULT_FMT` | `%(asctime)s %(levelname)s %(name)s: %(message)s` |
| `_OWNED_TAG` | Marks handlers managed by this module |
| `_FrameworkFilter` | Allows only `jiuwensymbiosis.*` into the rotating file |

### 3.5 Handler ownership model

```text
root.handlers
├── StreamHandler                 framework-owned; unfiltered console
├── RotatingFileHandler           framework-owned; _FrameworkFilter
└── host or pytest handlers       externally owned; never changed
```

This ownership model makes reconfiguration idempotent and lets `log_dir` change without deleting host logging setup.
Piper command logging remains independent: disable it with `JIUWEN_PIPER_CMD_LOG=0` or change its root with
`JIUWEN_PIPER_CMD_LOG_DIR` (`JIUWEN_CMD_LOG_DIR` remains a compatibility fallback).

## 4. Related files

- [`jiuwensymbiosis/utils/logging.py`](../../../jiuwensymbiosis/utils/logging.py): implementation and constants.
- [`jiuwensymbiosis/agent/config.py`](../../../jiuwensymbiosis/agent/config.py): Agent logging fields.
- [`jiuwensymbiosis/agent/builder.py`](../../../jiuwensymbiosis/agent/builder.py): startup configuration and Trace handler.
- [`examples/piper_pick_demo.py`](../../../examples/piper_pick_demo.py): YAML loading and `--debug` override.
- [Logging design](../../../design/logging.md): internal ownership and lifecycle decisions.
