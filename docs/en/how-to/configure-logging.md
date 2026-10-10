# Configure and Use Logging

> Category: How-to. The [Chinese source](../../zh/how-to/configure-logging.md) is authoritative.

This module is the **single logging configuration entry point** for the whole jiuwensymbiosis framework: one
`configure_logging` for a uniform format plus optional file output, one `get_logger` to call out, and a
`TraceLogHandler` that pushes key records into the execution trace.

## 1. Quick start

### 1.1 Log from application code

Just take a logger from `get_logger` and use `debug/info/warning/error` as needed:

```python
from jiuwensymbiosis.utils import get_logger   # or: from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger(__name__)

logger.info("connected to CAN %s", can_port)
logger.warning("target out of bounds, clamped to (%.1f, %.1f, %.1f)", x, y, z)
logger.error("enable timed out: %s", exc)
```

- `get_logger(__name__)` is exactly equivalent to the standard library's `logging.getLogger(__name__)` — **existing
  `logging.getLogger(__name__)` calls need no change** and still get the unified configuration.
- `get_logger()` (no argument) best-effort detects the caller's module name via `inspect.currentframe`, matching the
  `logging.getLogger(__name__)` idiom (CPython-specific; falls back to `"jiuwensymbiosis"`).
- **One log format everywhere**: the whole framework goes through one formatter,
  `%(asctime)s %(levelname)s %(name)s: %(message)s` (the constant `DEFAULT_FMT`). You **do not need** — and should not —
  call `setFormatter` or `basicConfig` yourself; doing so stacks a second handler alongside `configure_logging` and every
  console line prints twice (see §3.3).

> ⚠️ **Never do global logging configuration at module top level.** `build_robot_agent` calls `configure_logging` once
> at the right moment; reconfiguring or calling `basicConfig` stacks handlers.

### 1.2 Default behavior: write to `./logs` out of the box

With no configuration at all, framework logs go to both:

- **Console (stderr)** — one `StreamHandler`, uniform format, **unfiltered**: both `jiuwensymbiosis.*` and whatever
  openjiuwen emits through the standard library `logging` show up, so debugging sees the whole picture.
- **File** `<log_dir>/jiuwensymbiosis.log` — a `RotatingFileHandler`, 5 MB per file, 3 backups kept. Default
  `log_dir="./logs"`. **Only records in the `jiuwensymbiosis.*` namespace** (the file handler carries a
  `_FrameworkFilter` that blocks standard-library records bubbling up from openjiuwen, keeping the file clean).

The `./logs` default is deliberate. openjiuwen's own logging backend lands under `logs/logs/` for implementation
reasons (see below); the two are independent and do not interfere:

```
logs/
├── jiuwensymbiosis.log              ← [ours] framework logs (agent / rails / modules)
└── logs/                            ← [openjiuwen] its own logging backend
    ├── run/jiuwen.log               ← run log
    ├── runner.log                   ← runner log
    ├── interface/jiuwen_interface.log
    └── performance/jiuwen_performance.log
```

Each run's motion artifacts get their own directory (configured by `agent.logging.motion_dir`, default
`./jiuwen_motion_log`):

```
jiuwen_motion_log/<timestamp>/       ← one directory per run
├── commands.log                     ← [ours] Piper per-run command trail (piper only)
└── grasp_debug/                     ← [ours] detection / grasp-pose debug dumps (det/raw/info_NNN)
```

**Where each log goes**:

| Source | Destination | Enters `jiuwensymbiosis.log`? |
|------|------|------------------------------|
| `jiuwensymbiosis.*` (our agent/rails/adapter code) | `logs/jiuwensymbiosis.log` | ✅ yes |
| Piper command log + grasp debug dumps | `<motion_log_dir>/<stamp>/{commands.log, grasp_debug/}` (default `./jiuwen_motion_log`) | ❌ (separate per-run directory) |
| openjiuwen's own logging backend (the json/trace_id one) | files under `logs/logs/run/`, `logs/logs/interface/`, `logs/logs/performance/` | ❌ (bypasses standard-library logging) |
| openjiuwen records emitted through standard-library `logging` (e.g. `Registered parser ...` at init) | visible on console, **not in `jiuwensymbiosis.log`** | ❌ (blocked by `_FrameworkFilter`) |

> In short: `jiuwensymbiosis.log` holds only our own logs; openjiuwen's live in its own `logs/logs/` subdirectory (or
> are console-only). The console shows everything.

> `./logs` is **relative to the current working directory**. Running from different directories puts logs in different
> places — pin the run directory.

---

## 2. Control logging through configuration

Logging is controlled by two `RobotAgentConfig` fields, `logging.level` and `logging.dir`, which are eventually passed to
`build_robot_agent → configure_logging`. Two ways to set them: the **YAML `agent:` block** (declarative, recommended)
and **CLI options** (temporary override). They compose, with the CLI taking precedence.

Canonical fields are `logging.level` / `logging.dir` / `logging.motion_dir`; only grouped fields are accepted. See [complete agent configuration](../reference/agent-config.md). Omitting the directory still writes to `./logs`; only explicit `null` disables file output.

### 2.1 Declarative YAML `agent:` block

Write an `agent:` block in the task configuration file (e.g. `configs/piper/piper.yaml`):

```yaml
agent:
  logging:
    level: INFO
    dir: ./logs
```

The loading chain:

```
YAML → raw["agent"] → RobotAgentConfig.from_dict(raw["agent"])   # run_task.py / jiuwensymbiosis-run
                      → build_robot_agent(config=...)
                          → configure_logging(level=config.logging.level, log_dir=config.logging.dir)
```

- `from_dict` recursively merges overrides and validates grouped fields and types.
- **An unknown key raises `TypeError`** ([agent/config.py](../../../jiuwensymbiosis/agent/config.py)
  `RobotAgentConfig.from_dict`) — a typo (writing `enable_trace` for `modules.tracing.enabled`) fails immediately at load time
  instead of being silently ignored.

The `logging.level` / `logging.dir` fields:

| Field | Default | Meaning |
|------|------|------|
| `logging.level` | `"INFO"` | Root logger level (a `logging` level name string) |
| `logging.dir` | `"./logs"` | Log file directory; `None`/`null` = console only. Defaults to `./logs` (openjiuwen logs land in `logs/logs/` by its own implementation, independent of this directory) |

#### Disable file output and keep console only

```yaml
agent:
  logging:
    dir: null
```

#### Switch to DEBUG for detailed diagnostics

```yaml
agent:
  logging:
    level: DEBUG
    dir: ./logs
```

### 2.2 CLI override for temporary demo debugging

`examples/run_task.py` provides `--debug`, which sets `logging.level` to `DEBUG` through
`RobotAgentConfig.from_dict(..., overrides=...)`, **taking precedence over YAML**:

```python
overrides = {"logging": {"level": "DEBUG"}} if args.debug else {}
agent_cfg = RobotAgentConfig.from_dict(raw.get("agent"), overrides=overrides)
result = run_robot_task(session, args.query, agent_cfg)
```

```bash
# Temporarily enable DEBUG without changing the YAML logging.level
python examples/run_task.py --config configs/piper/piper.yaml --mock --debug
```

> The demo **does not call `logging.basicConfig` inside `main()`** — root logging is handed entirely to
> `build_robot_agent → configure_logging`. This matters: `basicConfig` installs a second handler that
> `configure_logging` does not recognize, so the console prints every line twice (see §3.3).
> <sup>run_task.py does set one `logging.basicConfig` at the top of `main()` so the voice-listening phase is visible,
> after which `configure_logging` takes over the root logger.</sup>

### 2.3 Link warnings to execution traces

Logs can also enter the execution trace (when `modules.tracing.enabled` is on). The related settings live in the `agent:` block
(see the [Tracing reference](../reference/tracing.md)):

```yaml
agent:
  modules:
    tracing:
      enabled: true
      capture_loggers: ["jiuwensymbiosis"]
```

Records emitted during a tool step appear in that entry's `log_events`; records with no active step appear in
`trace_log`.

---

## 3. API and mechanism reference

### 3.1 `configure_logging(level="INFO", *, log_dir=None, fmt=DEFAULT_FMT)`

Idempotently configures the root logger. **Ordinary developers usually never call it** — `build_robot_agent` does it for
you. Call it by hand only when using the framework outside `build_robot_agent`.

```python
from jiuwensymbiosis.utils.logging import configure_logging

configure_logging()                                         # console + defaults
configure_logging(level="DEBUG", log_dir="/var/log/js")     # console + file in a given directory
configure_logging(level="INFO", log_dir=None)               # console only (file off)
```

**Idempotency mechanism**: every handler this module creates is tagged with `_OWNED_TAG = "_jiuwensymbiosis_owned"`. On
a repeat call:

- StreamHandler: if one exists, only the formatter is updated; none is added.
- FileHandler: added when `log_dir` goes none→set, removed and closed when it goes set→none; switching directly from one
  non-empty path to another does **not** replace the existing handler. Call `configure_logging(log_dir=None)` first and
  then pass the new path, or restart the process.

```
configure_logging(level, log_dir)
  │
  ├─ root.setLevel(int_level)
  ├─ _owned_handlers() empty?
  │    ├─ yes → create StreamHandler (unfiltered), tag _OWNED_TAG, add to root
  │    └─ no  → update the formatter of the existing owned handler
  └─ log_dir given?
       ├─ yes and no owned FileHandler → create RotatingFileHandler(5MB, 3 backups)
       │                                   + attach _FrameworkFilter (allow only jiuwensymbiosis.*)
       └─ no and an owned FileHandler exists → remove and close it
```

> **Scope of the file handler's filter**: `_FrameworkFilter` is attached only to the `RotatingFileHandler`, **not to the
> StreamHandler** — so `jiuwensymbiosis.log` holds only our own logs while the console still shows every source,
> openjiuwen included (the whole picture while debugging). openjiuwen's own logging backend (which writes `logs/run/`
> and friends) does not go through standard-library logging at all and is unaffected.

If duplicate lines appear, check for host handlers or another `basicConfig()` call. If the file is empty while console
output exists, verify that `log_dir` is non-null and the logger name starts with `jiuwensymbiosis`.

### 3.2 `get_logger(name=None)`

A thin wrapper over `logging.getLogger`, kept as the single entry point for adding structured fields later. Usage in
§1.1.

### 3.3 `TraceLogHandler`: forward WARNING+ records to Trace

Forwards `WARNING`+ records to a bound trace sink (normally `TraceRail`). This is the core of "key logs enter the
trace" — attach the handler to the corresponding logger and existing `logger.warning(...)` calls in business modules
automatically become the trace's `log_events`, **with no business-code change**.

```python
from jiuwensymbiosis.utils.logging import TraceLogHandler

handler = TraceLogHandler(sink=trace_rail, level=logging.WARNING)
logging.getLogger("jiuwensymbiosis").addHandler(handler)
# Subsequent WARNING+ records from the selected loggers enter the active Trace
```

- **Capture level fixed at `WARNING`**: hard-coded in `build_robot_agent` (`capture_log_level=_logging.WARNING`), not
  configurable. Rationale in the [logging design](../../../design/logging.md).
- **emit behavior**: a no-op when `sink is None` (so it can be constructed early); otherwise it assembles
  `{logger, level, msg, ts}` and calls `sink.record_log_event(...)`; sink errors are swallowed by the precise types
  `(AttributeError, TypeError, ValueError)` — a logging handler must never raise.
- **Lifetime** is managed by `TraceRail` (see the [tracing internals](../../../design/tracing.md)): `set_sink(sink)`
  swaps the sink (bound back at the start of each invoke, set to None at the end).

If warnings do not appear in Trace, confirm `modules.tracing.enabled`, `modules.tracing.capture_loggers`, and a record level of at least
`WARNING`.

### 3.4 Constants

| Constant | Value | Purpose |
|------|----|------|
| `DEFAULT_FMT` | `"%(asctime)s %(levelname)s %(name)s: %(message)s"` | Default formatter format |
| `_OWNED_TAG` | `"_jiuwensymbiosis_owned"` | Marks handlers created by this module, for idempotency |
| `_FrameworkFilter` | a `logging.Filter` subclass | Allows only `jiuwensymbiosis.*` records; attached to the file handler so `jiuwensymbiosis.log` never mixes in openjiuwen logs |

### 3.5 Handler ownership model

This module uses `_OWNED_TAG` to tell "handlers I created" apart from "handlers injected externally (pytest,
application code)":

```
root.handlers = [
  StreamHandler(_owned=True)                           ← unfiltered, full console output
  RotatingFileHandler(_owned=True, _FrameworkFilter)   ← only jiuwensymbiosis.*
  LogCaptureHandler                                    ← injected by pytest, configure_logging never touches it
]
```

This guarantees that:

- `configure_logging` never removes pytest's log-capture handler by mistake.
- Repeat calls never stack owned handlers.
- Turning `log_dir` on and off only touches the owned FileHandler; switching between two non-empty paths requires
  explicitly closing the old handler first.

Per-run motion artifacts live in their own directory, one per run: `<motion_log_dir>/<stamp>/` holds the Piper
`commands.log` and the `grasp_debug/` detection dumps side by side. Set the root with `agent.logging.motion_dir`
(default `./jiuwen_motion_log`); disable the Piper command log with `JIUWEN_PIPER_CMD_LOG=0`.

---

## 4. Related files

| File | Role |
|------|------|
| [jiuwensymbiosis/utils/logging.py](../../../jiuwensymbiosis/utils/logging.py) | This module's implementation |
| [jiuwensymbiosis/utils/\_\_init\_\_.py](../../../jiuwensymbiosis/utils/__init__.py) | re-exports `configure_logging` / `get_logger` / `TraceLogHandler` / `DEFAULT_FMT` |
| [jiuwensymbiosis/agent/config.py](../../../jiuwensymbiosis/agent/config.py) | `RobotAgentConfig.logging.level` / `logging.dir` fields + `from_dict` (YAML pass-through) |
| [jiuwensymbiosis/agent/builder.py](../../../jiuwensymbiosis/agent/builder.py) | `build_robot_agent` calls `configure_logging`; attaches `TraceLogHandler` when tracing is on |
| [examples/run_task.py](../../../examples/run_task.py) | `--debug` overrides `logging.level`; demonstrates the YAML `agent:` loading chain |
| [jiuwensymbiosis/adapters/piper/lowlevel.py](../../../jiuwensymbiosis/adapters/piper/lowlevel.py) | Piper `_attach_cmd_log_handler` reuses `configure_logging` |
| [Tracing reference](../reference/tracing.md) | The consumer of `TraceLogHandler` and its data format |
| [Logging design](../../../design/logging.md) | Internal ownership and lifecycle decisions |
