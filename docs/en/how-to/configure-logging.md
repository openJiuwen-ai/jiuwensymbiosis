# Configure and use logging

Symbiosis and openjiuwen have independent verbosity settings. Symbiosis defaults to INFO progress summaries. Upstream keeps its current policy unless `openjiuwen_level` explicitly sets a shared console/file level. Symbiosis contexts, complete plans, tool parameters, results and servo ticks use DEBUG.

## Configuration

```yaml
agent:
  logging:
    level: INFO
    openjiuwen_level: null              # Preserve upstream; explicit WARNING limits console and files
    dir: ./logs                         # null disables the Symbiosis main log file
    motion_dir: null                    # Defaults to ./jiuwen_motion_log
```

CLI `--debug` overrides only `logging.level`. It does not enable upstream INFO. Set `openjiuwen_level: INFO` when needed; it applies to console and files together. Some upstream INFO statements include requests and context. Accepted levels and validation are described in the [agent configuration reference](../reference/agent-config.md).

Omitting the field or setting `null` skips upstream configuration entirely: no logger/sink levels or handlers change. It preserves current state rather than restoring an earlier snapshot; explicit WARNING followed by `null` stays at WARNING. Existing robot YAML examples explicitly select WARNING; that choice is not the field default.

For an explicit level, public upstream configuration APIs unify default, per-logger and sink levels while preserving backend, destinations, format, routes and rotation. Changing the upstream level may reopen its files; identical settings skip the reset. The former console-only `openjiuwen_console_level` field has been removed; use `openjiuwen_level`.

Paths are relative to the process working directory. Setting `logging.dir: null` does not disable upstream files, Piper command files or execution traces.

## Ownership and outputs

| Source | Default contents | Destination |
|---|---|---|
| Symbiosis core, CLI, calibration and workbench | INFO connection/execution progress and fast-planner LLM request/completion timing; DEBUG payload details | stderr and `<logging.dir>/jiuwensymbiosis.log` |
| openjiuwen console | Current upstream policy, or explicit openjiuwen_level | Upstream console handlers / stdout and stderr sinks |
| openjiuwen files | Current upstream policy, or the same explicit level as the console | Upstream file handlers / sinks |
| Piper commands | DEBUG motion diagnostics | `<motion_dir>/<timestamp>/commands.log` |
| Execution traces | Structured tool calls and captured WARNING+ records | `<workspace>/traces/` or `modules.tracing.dir` |

The Symbiosis main file accepts only `jiuwensymbiosis`, `jiuwensymbiosis_gui` and their descendants. Calibration uses `jiuwensymbiosis.calibration.*`; the task CLI uses `jiuwensymbiosis.cli.run_task`. The two framework namespace roots share the same console/file handler instances and use `propagate=False`. Upstream keeps its own outputs. When an upstream level is explicit, a separate adapter sets the `openjiuwen` namespace level and attaches a fallback console handler for plain stdlib loggers. It defers to any other console handler on the propagation path, including host root handlers. Upstream records never enter the Symbiosis main file.

Symbiosis uses `%(asctime)s %(levelname)s %(name)s: %(message)s`. Its main file rotates at 5 MB with three backups. Upstream retains its own format, directories, rotation and destinations; some versions place default files under `logs/logs/`. Explicit upstream configuration only changes severity settings; disabled destinations remain disabled.

Piper keeps the same format in its per-run file without changing root severity or closing the main log. Reconnecting within the same run directory appends to the existing command file; new runs still use separate directories. DEBUG motion records enter the main file/console only when `logging.level: DEBUG`. Set `JIUWEN_PIPER_CMD_LOG=0` to disable attachment and detach an existing command handler; use `logging.motion_dir` to select its parent directory. The last owning driver closes the shared file and restores its logger-level override without overwriting later external changes. Failed construction rolls back the attachment; failed cleanup remains available for retry. A failed directory change never keeps writing to the previous run.

With `modules.tracing.console: true`, live step summaries use standard logging and respect `logging.level`; full parameters and output use DEBUG. The structured trace JSON still contains its configured tool data independently of text-log verbosity.

Final CLI answers are stdout user output, independent of logging severity. Successful fastagent results without `output` / `content` text print “任务完成，共执行 N 步” using `steps_done`; existing text answers take precedence. Voice mode prints each command’s reply once before passing it to TTS, so replies remain visible with `--tts null`. Input context and full execution results remain DEBUG.

For task results with `ok: false`, the CLI separately logs the task reason and each failed step's index, operation and reason at ERROR. Voice mode does this before reducing the result to a reply. Failures returned only through `reason` or `steps[].reason` remain visible at INFO/WARNING settings; complete parameters, plans and results remain DEBUG.

## Initialization and Python API

The CLI merges all command-line overrides before constructing and validating Agent settings once, then checks for a query before configuring logging from those final settings. Both CLI and Runtime do this before session construction/connection, voice listening and initial planning. `prepare_session()`, agent builders and direct `run_fast_task()` calls apply configuration idempotently, so early planning failures are logged. A missing query produces a stderr error and return code 2 without creating a Symbiosis log file. Logging is initialized before adapter import, so import failures still reach the file. Package import does not configure upstream logging; import-time messages follow upstream policy, and explicit YAML settings apply only after the entry point reads configuration. The adapter uses public `get_log_config_snapshot()` / `configure_log_config()` APIs for default and Loguru backends, with no private APIs or lifecycle hooks. A changed upstream configuration resets its loggers and outputs; subsequent lazy creation uses the new policy.

Embedding hosts should distinguish other dependency initialization: the current `pymilvus` dependency calls global `logging.dictConfig()` on first import, which can close existing handlers. That behavior is separate from the adapter, which does not directly configure root.

```python
from jiuwensymbiosis.agent import RobotAgentConfig

config = RobotAgentConfig.from_dict(raw.get("agent"))
config.prepare_session(session)
with session:
    ...
```

For standalone logging:

```python
from jiuwensymbiosis.utils.logging import configure_logging, get_logger

configure_logging("INFO", log_dir="./logs")  # Preserve upstream configuration
logger = get_logger("jiuwensymbiosis.my_module")
logger.info("Completed %d steps", count)
logger.debug("Parameters: %s", params)
```

YAML accepts level strings or `null` for `openjiuwen_level`; the public Python `configure_logging()` API also accepts standard numeric levels such as `logging.WARNING`. NOTSET admits all standard severities; use `None` to leave upstream untouched.

Project modules use `get_logger(__name__)`; existing `logging.getLogger(__name__)` calls return the same stdlib logger and already route records to the shared namespace handlers. Business code must not invoke handlers directly. Application scripts and modules executable with `python -m` use stable Symbiosis namespace names instead of `__main__`. Do not add `basicConfig()` or configure root from a driver.

`configure_logging()` manages only handlers tagged `_jiuwensymbiosis_owned` on `jiuwensymbiosis` and `jiuwensymbiosis_gui`:

- Repeated calls update format and severity without stacking handlers.
- Both handlers enforce severity; an explicit DEBUG child cannot bypass INFO output thresholds.
- Changing directory A to B prepares B before switching all outputs and closing A. `None` closes the main file. Directory/file preparation errors preserve prior policy. Old-file cleanup errors propagate after the new configuration is active; failed handlers are retried on the next configuration call.
- Root severity, handlers, formatters and filters remain untouched. Foreign handlers retain their own output policies.
- An explicit upstream level invokes a separate adapter using public configuration APIs for console and files together. Later explicit upstream configuration takes precedence until an explicit upstream level is applied through Symbiosis again; the adapter does not intercept upstream reconfiguration.

Framework records no longer propagate to root. Host collectors and pytest `caplog` must attach their capture handler directly to `jiuwensymbiosis` / `jiuwensymbiosis_gui`. Runtime and Trace already capture there; this repository’s `caplog` fixture does too. To adjust source severity in tests, specify `caplog.at_level("DEBUG", logger="jiuwensymbiosis")`.

The two framework namespaces have independent source levels, as does `openjiuwen` when explicitly configured, without inheriting host root severity. Host-owned handlers still enforce their own policy. NOTSET allows all standard severities; the upstream Loguru backend maps it to its lowest supported standard level, DEBUG.

Configuration is process-wide. Official Runtime operations are serialized. Embedded concurrent tasks cannot use these settings for independent per-task levels or files.

## Compatibility and migration

After `configure_logging()`, framework namespaces stop propagating to root. Hosts that collect only from root must also attach their collector to both framework namespaces; detach it from the same locations when collection ends:

```python
import logging

for name in ("jiuwensymbiosis", "jiuwensymbiosis_gui"):
    logging.getLogger(name).addHandler(collector)
```

External pytest suites must likewise attach `caplog.handler` and remove it after each test. Set source severity with `caplog.at_level("DEBUG", logger="jiuwensymbiosis")`. This repository's fixture already handles attachment.

The legacy `scripts/calibrate/calibrate_hand_eye.py` command, including `--selftest`, now sends logs to stderr instead of stdout, with timestamps, levels and logger names. Scripts consuming stdout must read stderr and adjust text matching; use the exit code when only checking whether self-tests passed.

## Warning capture in execution traces

`TraceLogHandler` forwards WARNING+ from configured loggers into trace `log_events`. The default `jiuwensymbiosis` namespace includes calibration, but not upstream or workbench loggers:

```yaml
agent:
  modules:
    tracing:
      enabled: true
      capture_loggers: [jiuwensymbiosis]
```

The capture threshold is fixed at WARNING. Invocation completion unbinds the sink; Session cleanup removes the handler. See [execution traces](../reference/tracing.md) and the [logging design](../../../design/logging.md).
