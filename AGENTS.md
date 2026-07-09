# AGENTS.md

Shared instructions for AI coding assistants working in `jiuwensymbiosis`.
Keep this file cross-tool (Cursor / Copilot / Claude all read it). Prefer
nearby code and tests over assumptions.

`pyproject.toml` is the canonical source of truth for Python/tooling
settings. `CLAUDE.md` imports this file (`@AGENTS.md`) and adds
Claude-specific pointers only.

## Project Overview

JiuwenSymbiosis is an embodied agent framework built on `openjiuwen` for robotics. It provides hardware-agnostic tools, safety policies, and multi-agent collaboration. The core design principle is **Capability Mixin architecture**: a single codebase adapts to different robot form factors (SCARA, 6-DoF, suction cup, gripper) through mixin composition — new hardware only needs a YAML config + 6 adapter files.

## Build & Test Commands

```bash
# One-stop entry points (see Makefile); defaults to conda env "jiuwensymbiosis",
# override with `make check CONDA_ENV=` to use plain PATH.
make check        # ruff format --check + ruff check + mypy on staged files (mypy advisory)
make fix          # ruff format + ruff check --fix on staged files
make test         # pytest tests/unit_tests/ (no hardware/GPU)
make test-all     # pytest (incl. integration)
# Use COMMITS=N to check files changed in the last N commits instead of staged.

# Install in editable mode
pip install -e ".[dev]"                                    # core + test deps
pip install -e ".[full]" --extra-index-url https://download.pytorch.org/whl/cu128  # + vision/GPU deps
pip install -e ".[piper]"                                  # + piper hardware SDK

# Run tests
pytest                                                     # all unit tests (no hardware needed)
pytest tests/unit_tests/                                   # unit tests only (no hardware/GPU)
pytest -m integration                                      # integration tests (needs hardware/GPU)
pytest tests/unit_tests/agent/test_builder.py              # single test file
pytest -k "test_capabilities"                              # filter by test name

# Validate a hardware adapter
python scripts/validate_adapter.py --module jiuwensymbiosis.adapters.my_robot

# Run demo (mock mode: no hardware, no real LLM — uses MockArmEnv + MockModel).
# Task is not in the config — pass it via --query (or --voice at run time).
python examples/piper_pick_demo.py --config configs/piper/piper.yaml --mock --query "<任务>"
# CLI entry point (after pip install)
piper-pick-demo --config configs/piper/piper.yaml --mock --query "<任务>"

# Lint / format / type-check (tools not installed by default; install: pip install ruff mypy)
ruff format .           # format (Black-compatible drop-in)
ruff check .            # lint; ruff check --fix . to auto-fix
mypy jiuwensymbiosis/
```

## Critical: Proxy Hygiene

`clear_proxy_env()` (defined in `jiuwensymbiosis/utils/proxy.py`, exported from `jiuwensymbiosis.utils` and `jiuwensymbiosis`) **must** be called before `import openjiuwen`. HTTP proxy env vars cause `httpx` to require `socksio` and route localhost through proxy, breaking local vLLM/detection calls. The root `conftest.py` does this automatically for tests.

## Centralised Logging

`jiuwensymbiosis.utils.logging` provides one choke point for all logging:

- `configure_logging(level="INFO", *, log_dir=None)` — idempotent root-logger setup: one `StreamHandler` with a uniform format (`%(asctime)s %(levelname)s %(name)s: %(message)s`) plus an optional `RotatingFileHandler` (`<log_dir>/jiuwensymbiosis.log`, 5 MB / 3 backups). `build_robot_agent` calls it with `RobotAgentConfig.log_level` / `log_dir`.
- `get_logger(name=None)` — thin alias over `logging.getLogger`; new code should use it. Legacy `logging.getLogger(__name__)` calls remain valid.
- The Piper driver's per-run `commands.log` (`_attach_cmd_log_handler`) now routes through `configure_logging` + a tagged `FileHandler` with the same format. Disable with `JIUWEN_PIPER_CMD_LOG=0`; override dir with `JIUWEN_PIPER_CMD_LOG_DIR`.
- `TraceLogHandler` — a `logging.Handler` that forwards `WARNING`+ records from `RobotAgentConfig.trace_capture_loggers` (default `["jiuwensymbiosis"]`) into the active execution trace, so rail warnings / detector failures land in the trace with no business-code changes.

## Architecture: Layered Capability-Gated Design

The framework has 7 layers, with data flowing top-down for commands and bottom-up for observations:

```
Agent Layer       RobotSession + build_robot_agent() + RobotAgentConfig
Safety Rails      SafetyRail / RecoveryRail / VisualFeedbackRail / SkillUseRail (before_tool_call hooks); TraceRail (parallel, optional)
Tool Layer        build_robot_tools(api) | RobotControlTool(api) | InProcessCodeTool
Skill Layer       SKILL.md docs (visual_pick, visual_place) loaded by SkillUseRail
API Layer         MotionMixin / VisionMixin / SuctionMixin etc. (@robot_tool methods)
Env Layer         BaseRobotEnv — the SINGLE hardware contract (connect/disconnect/observe)
Hardware Layer    XxxDriver — adapter author's main work (serial/CAN/socket)
```

### Key Architectural Patterns

**Capability Gating**: Tools are emitted only for `api.capabilities ∩ env.capabilities`. Env declares what hardware can do (manual `frozenset`); Api derives capabilities from its Mixin MRO (automatic). `build_robot_tools(api, env=env)` enforces the intersection — methods from mixins whose capability string isn't in env simply don't become LLM tools.

**Mixin Default Delegation**: Motion/grasp/get_image methods in mixins have working default implementations that delegate to `self.env.<verb>()`. Api authors only override methods with body-specific geometry (e.g., tip↔flange offset, tilted tool). `get_grasp_info_simple` / `pixel_to_base_xyz` / `analyze_scene` (VisionMixin) have no *mixin-level* default, but `_common/vision.py` provides `default_get_grasp_info_simple` / `default_pixel_to_base_xyz` helpers that factor out the eye-in-hand detect→centroid→project→correct→geometry pipeline — an adapter calls them, supplying only its `seg_fn` and a `pose_to_tf(flange_pose) -> 4x4` callback (the one vendor-specific step); `analyze_scene` still requires a per-adapter implementation.

**@robot_tool Decorator**: Annotates unbound methods with `ToolMeta` (name, desc, input_params JSON Schema auto-generated from type hints, capability, tags). `build_robot_tools` walks the MRO, finds decorated methods, binds them, and wraps them as openjiuwen `LocalFunction` tools. Override methods inherit the decorator metadata; re-decorate to customize descriptions.

**Two Tool Strategies** (can coexist):
- `build_robot_tools(api)` — each `@robot_tool` method becomes a separate LLM tool (good for few tools)
- `RobotControlTool(api)` — single `robot_control` entry point with `action`/`params` dispatch (good for SKILL.md workflows); appended by `build_robot_agent` only when `RobotAgentConfig.enable_skill=True`
- `InProcessCodeTool` — in-process Python execution (available in "code" and "hybrid" modes)

**Safety rails unwrap robot_control**: When RobotControlTool is used, rails transparently unpack `action`/`params` to apply safety checks on the actual motion command.

**RobotSession Lifecycle**: `RobotSession` is a context manager — `__enter__` calls `connect()` (env + sidecars), `__exit__` calls `disconnect()`. Both are idempotent. Sidecars (e.g., detection subprocess) are started/stopped automatically.

**Known Capabilities** (defined in `env/base.py:KNOWN_CAPABILITIES`):
`motion.cartesian`, `motion.joint`, `grasp.suction`, `grasp.parallel`, `vision.camera`, `vision.depth`, `vision.detection`, `sorting.command`, `speech.tts`

### Safety & Auxiliary Rails

1. **SafetyRail** — Pre-motion boundary check: validates Z floor (`z_min_safe`) and XY workspace bounds before `goto_xyzr`/`goto_pose`, and joint soft limits before `move_joint(q)` (`joint_limits`, unit = env's `move_joint` convention). Rejects with `ValueError` (per-failure message: missing q / wrong type / length mismatch / non-finite / out of range) so LLM can self-correct.
2. **RecoveryRail** — On motion/grasp failure, auto-homes + releases end-effector to return to safe state.
3. **VisualFeedbackRail** — Captures camera frame after every motion/grasp, injects into agent context for VLM result verification.
4. **SkillUseRail** — Loads built-in `SKILL.md` docs and appends `RobotControlTool`; attached only when `RobotAgentConfig.enable_skill=True`. (`rails/__init__.py` re-exports only the first three; `SkillUseRail` lives in `agent/builder.py`.)

Note: `TraceRail` (see "Execution Trace & Replay" below) is a fourth parallel rail that lives in `jiuwensymbiosis/agent/trace.py` — **not** under `rails/` — and is gated by `enable_tracing` rather than a safety flag.

Rails are enabled/disabled via `RobotAgentConfig` flags and gated by session capabilities (e.g., VisualFeedbackRail requires `vision.camera`; SafetyRail attaches when **any** of `motion.cartesian` / `motion.joint` is present, so joint-only robots get the `move_joint` soft-limit pre-check too).

### Execution Trace & Replay

`TraceRail` (`jiuwensymbiosis/agent/trace.py`) is an optional parallel rail (enabled via `RobotAgentConfig.enable_tracing`, default **off** for zero overhead) that records each `agent.invoke()` as a structured `ExecutionTrace`:

- Per tool-call step: `tool_name`, `input_params`, `output_summary`, `success`/`error`, `duration_s`, an `observation` snapshot (pose/joints/extra, no raw arrays), and an optional saved `frame_path`.
- Rail events pushed via the `TraceEventSink` interface: SafetyRail rejections, RecoveryRail recovery (with real `home_ok`/`released_ok`), VisualFeedbackRail frame injections.
- `WARNING`+ log lines from `trace_capture_loggers` (default `["jiuwensymbiosis"]`) captured via `TraceLogHandler` — no business-code changes.

The trace JSON is persisted to `<workspace>/traces/{conversation_id}_{timestamp}_{pid}.json` on invoke completion (one write per run); JPEG frames go to `<workspace>/traces/frames/{run_token}/` (one subdir per invoke, so `step_NNN.jpg` never collides across runs) when `trace_save_frames=True`. Override the output dir with `trace_dir` (default `<workspace>/traces`). Cap with `trace_max_entries` / `trace_max_frames`. Full config: `enable_tracing` / `trace_max_entries` / `trace_max_frames` / `trace_save_frames` / `trace_console` / `trace_dir` / `trace_capture_loggers`.

`jiuwensymbiosis-replay <trace.json>` prints a text timeline of steps, rail events, log events, and frame paths. Set `trace_console=True` for a live one-line-per-step dashboard during the run.

### Hardware Adapter Pattern (6 files)

New robot types follow this pattern under `jiuwensymbiosis/adapters/<name>/`:
1. `config.py` — `@dataclass` with `from_yaml()`/`from_dict()`
2. `lowlevel.py` — Driver implementing `RobotDriver` Protocol (motion, gripper/suction, camera)
3. `env.py` — `BaseRobotEnv` subclass: `capabilities` frozenset, `connect`/`disconnect`/`get_observation`, expose `z_min_safe`/`workspace_bounds`/`joint_limits`/`home_pose`/`tool_offset_mm` as properties
4. `api.py` — Multi-inherits Mixins + `BaseRobotApi`; overrides geometry-specific methods, implements vision methods
5. `session.py` — `make_builder(cfg_cls, env_cls, api_cls, ...)` one-liner; `api_kwargs_from_cfg` accepts a declarative list (`["cfg_attr"` or `"cfg_attr:api_kwarg"`, dotted paths OK) so same/near-named cfg→Api field mapping needs no hand-written extractor, and `make_detector_sidecar()` provides the standard detection-server sidecar
6. `config_template.yaml` — YAML template with Chinese annotations

Template at `templates/xxx_adapter/`. Validate statically with `scripts/validate_adapter.py`; smoke-test runtime behavior (every `@robot_tool` callable + JSON-serializable) with `scripts/smoke_test_adapter.py`.

### Visual Perception Pipeline

Detection runs as a subprocess (GroundingDINO + SAM2) via `adapters/_common/detector_sidecar.py`. `RobotSession` manages lifecycle. The `_common` package provides shared utilities: `detector_client.init_detector()`, `vision.detect_and_centroid()`, `vision.apply_xy_correction()`.

### Workspace Resolution

Priority: explicit `workspace` arg > `$JIUWENSYMBIOSIS_WORKSPACE` env var > `~/.jiuwensymbiosis/settings.json` > `~/.jiuwensymbiosis/{session.name}_workspace/`

## Source Tree Layout

```
jiuwensymbiosis/          # Main package
  agent/                  # RobotSession, build_robot_agent, RobotAgentConfig, ModelSpec, MockModel (--mock)
  api/                    # BaseRobotApi, @robot_tool decorator, capability mixins
  env/                    # BaseRobotEnv, MockArmEnv, KNOWN_CAPABILITIES
  tools/                  # build_robot_tools, RobotControlTool, InProcessCodeTool
  rails/                  # SafetyRail, RecoveryRail, VisualFeedbackRail
  skills/                 # Built-in SKILL.md files (visual_pick, visual_place)
  adapters/
    piper/                # Piper 6-DoF reference adapter (6-DoF + gripper + wrist vision)
    _common/              # Shared adapter utilities (builder, detector, vision, calibration, protocol)
  serving/                # Visual perception server subprocess (GroundingDINO + SAM2)
  utils/                  # proxy hygiene (proxy.py), centralised logging (logging.py)
configs/piper/            # YAML configs for piper tasks
templates/xxx_adapter/    # Adapter skeleton for new hardware
tests/
  unit_tests/             # Mirrors package structure
  mocks/                  # MockApi, MockEnv, MockDriver, MockScene
  integration/            # Hardware/GPU-dependent tests
scripts/validate_adapter.py  # Static compatibility checker for new adapters
scripts/smoke_test_adapter.py # Runtime smoke test: drive each @robot_tool with MockEnv
examples/                 # Runnable demos (piper_pick_demo)
docs/                     # Deep-dive manuals: architecture.md, hardware-porting-guide.md, logging.md, trace.md
Makefile                  # check / fix / format / lint / type-check / test targets (conda env "jiuwensymbiosis" by default)
```

## Instruction Priority

- Follow system, tool, and user instructions first, then this file, then
  module-local docs.
- Before changing behavior, inspect the touched module, its exported
  surface in `__init__.py`, and nearby tests/examples.
- Prefer small, targeted diffs. Do not refactor unrelated areas
  opportunistically.

## More Detail

Topic-scoped rules (short, hard, path-gated) live in `.claude/rules/`;
deep reference manuals (longer, on-demand) live in `.claude/skills/`.
Both are listed in `CLAUDE.md` under "Rules & Skills Index".
