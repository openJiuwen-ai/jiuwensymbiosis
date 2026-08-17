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
pip install -e ".[gui]"                                    # + 图形界面 (NiceGUI, 浏览器模式)

# Run tests
pytest                                                     # all unit tests (no hardware needed)
pytest tests/unit_tests/                                   # unit tests only (no hardware/GPU)
pytest -m integration                                      # integration tests (needs hardware/GPU)
pytest tests/unit_tests/agent/test_builder.py              # single test file
pytest -k "test_capabilities"                              # filter by test name

# Validate a hardware adapter
python scripts/validate_adapter.py --module jiuwensymbiosis.adapters.my_robot

# Run a task (generic runner; robot = --config's adapter: field, task = --query).
# mock mode: no hardware, no real LLM (piper only; implies --stepagent).
python examples/run_task.py --config configs/piper/piper.yaml --mock --query "<任务>"
# CLI entry point (after pip install)
jiuwensymbiosis-run --config configs/piper/piper.yaml --mock --query "<任务>"

# Introspection — the machine-readable view a planner / coding agent reads
# instead of adapter source or SKILL.md prose (see docs/autonomous-planning.md)
jiuwensymbiosis-actions --vocabulary [--json]                       # the SHARED action vocabulary (no robot)
jiuwensymbiosis-actions --config configs/cruzr/cruzr.yaml [--json]  # that vocabulary, gated to one body
jiuwensymbiosis-skills  [--json]                                    # skill library + contracts
jiuwensymbiosis-state   --config configs/cruzr/cruzr.yaml [--json]  # live world state (connects!)

# Run the GUI (browser mode; selects a body + real config and runs on hardware)
python -m jiuwensymbiosis.gui        # or the console script: jiuwensymbiosis-gui
#   Opens the default browser at http://127.0.0.1:<port> (NiceGUI, never native=True,
#   so no pywebview/WebKitGTK). No extra system libs needed. The GUI does a startup
#   pre-check (jiuwensymbiosis/gui/preflight.py) and prints the exact package to
#   install (pip install -e ".[gui]") instead of crashing with a raw traceback.

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
Skill Layer       SKILL.md docs (visual_pick, visual_place, transport) — SkillUseRail (tool path) / plan_task (fast path)
API Layer         ActionSpec vocabulary (api/actions.py) + mixins implementing it (@implements)
Env Layer         BaseRobotEnv — the SINGLE hardware contract (connect/disconnect/observe)
Hardware Layer    XxxDriver — adapter author's main work (serial/CAN/socket)
```

### Key Architectural Patterns

**Shared Action Vocabulary**: an action's contract — name, capability gate, params, result shape, pre-conditions and effects — is declared **once** in `api/actions.py` as an `ActionSpec`; a body supplies only an *implementation* via `@implements(SPEC)`. So the same action name means the same thing on every robot, and a plan or SKILL.md written for one body is meaningful on the next. A tool only one body has (bring-up, calibration, vendor demo) keeps the plain `@robot_tool` decorator and is **invisible to the planner** by default. `jiuwensymbiosis-actions --vocabulary` prints the whole vocabulary; `--config <yaml>` prints the subset one body's capabilities admit.

**Capability Gating**: Tools are emitted only for `api.capabilities ∩ env.capabilities`. Env declares what hardware can do (manual `frozenset`); Api derives capabilities from the actions it implements, plus any mixin `capability` attr (automatic). `build_robot_tools(api, env=env)` enforces the intersection — an action whose capability isn't in env simply doesn't become an LLM tool. The capability comes from the action's own `ActionSpec`, never from whichever class declares the method.

**Defaults & Components** (the mixin layer is gone): an action whose implementation is one line of delegation to an Env verb is a plain function in `api/defaults.py` that the adapter calls explicitly — `@implements(GOTO_XYZR)` then `return defaults.goto_xyzr(self, ...)` — so no base class is involved and the MRO stays flat. What is still a class lives in `api/components.py`, because it carries state plus body hooks: `Scene3D` (last detection) and `Approach` (last sensed surface) hold the shared 3D/approach geometry, and a body supplies only the hooks (how to grab a calibrated frame, which detector to run, how to drive the base, whether a wide-angle sensor exists); `Reachability` is not an action provider at all but answers a planning question the planner reads directly. An adapter should *hold* these (`self._scene = Scene3D(self)`, as cruzr does) rather than inherit them. The vendor-specific vision work — `get_grasp_info_simple` / `pixel_to_base_xyz` / `analyze_scene` — has no default and is implemented per adapter on top of `perception/vision.py` (`detect_and_centroid`, `apply_xy_correction`, `build_grasp_result`).

**@robot_tool Decorator**: Annotates unbound methods with `ToolMeta` (name, desc, input_params JSON Schema auto-generated from type hints, capability, tags). `build_robot_tools` walks the MRO, finds decorated methods, binds them, and wraps them as openjiuwen `LocalFunction` tools. Override methods inherit the decorator metadata; re-decorate to customize descriptions.

**Planning Contract** (what makes a body plannable, not just callable): beyond the call schema, every `@robot_tool` carries `returns` (result-field JSON Schema, auto-derived from a `TypedDict` return annotation), `requires` / `provides` / `invalidates` (robot self-state, over the closed vocabulary in `api/state.py:KNOWN_STATE_TOKENS`), and `produces_location` / `consumes_location` / `invalidates_locations` (per-referent location freshness — an action that moves the base stales every prior sensed position). `SkillSpec` + SKILL.md frontmatter carry the same fields, because **a skill is a compound action** and both planning tiers share one reasoning machine. The contract never encodes an order; it states pre-conditions and effects so a planner can *derive* one, and `parse_sequence` accepts any permutation whose pre-conditions hold. `WorldState.snapshot(session)` reports the same vocabulary at runtime (observation overrides belief; an absent token means *unknown*, never *false*). Full manual: `docs/autonomous-planning.md`.

**Two Tool Strategies** (can coexist):
- `build_robot_tools(api)` — each `@robot_tool` method becomes a separate LLM tool (good for few tools)
- `RobotControlTool(api)` — single `robot_control` entry point with `action`/`params` dispatch (good for SKILL.md workflows); appended by `build_robot_agent` only when `RobotAgentConfig.enable_skill=True`
- `InProcessCodeTool` — in-process Python execution (available in "code" and "hybrid" modes)

**Safety rails unwrap robot_control**: When RobotControlTool is used, rails transparently unpack `action`/`params` to apply safety checks on the actual motion command.

**RobotSession Lifecycle**: `RobotSession` is a context manager — `__enter__` calls `connect()` (env + sidecars), `__exit__` calls `disconnect()`. Both are idempotent. Sidecars (e.g., detection subprocess) are started/stopped automatically.

**Known Capabilities** (defined in `env/base.py:KNOWN_CAPABILITIES`):
- arm — `motion.cartesian`, `motion.joint`, `motion.servo`
- mobile manipulator — `motion.base`, `motion.base_servo`, `motion.lift`, `motion.waist`, `motion.goal`
- end effector — `grasp.suction`, `grasp.parallel`, `grasp.dual_arm`
- sensing / planning — `vision.camera`, `vision.depth`, `vision.detection`, `vision.eye_to_hand`, `planning.reachability`
- misc — `sorting.command`, `speech.tts`

### Safety & Auxiliary Rails

1. **SafetyRail** — Pre-motion boundary check, with the watch set **derived from declared capabilities** rather than hard-coded: `motion.cartesian` → Z floor (`z_min_safe`) + XY workspace bounds on `goto_xyzr`/`goto_pose`; `motion.joint` → joint soft limits on `move_joint(q)` (`joint_limits`, unit = env's `move_joint` convention); `motion.base` → per-command translation/turn caps (`base_step_limits`) on `navigate_relative`/`rotate_base`/`drive_arc`; `motion.lift` → `lift_limits` on `set_lift_pose`; `motion.waist` → `waist_step_limit_rad` on `turn_waist`. Every envelope property defaults to `None` = **no range check** (type/finite checks still run), so declaring a capability never invents a limit the hardware never stated. Rejects with `ValueError` (per-failure message: missing q / wrong type / length mismatch / non-finite / out of range) so LLM can self-correct.
2. **RecoveryRail** — On motion/grasp failure, auto-homes + releases end-effector to return to safe state. The release step goes through a generic `release_effector()` hook, and homing consults `env.holding_payload` first — a body still carrying a payload must not be homed blindly (that would drop it).
3. **VisualFeedbackRail** — Captures camera frame after every motion/grasp, injects into agent context for VLM result verification.
4. **DiagnosisRail** — Online failure-feedback (Trace Feedback Loop P1): after a failed step, stages a compact diagnosis (current params + relevant recent history + system state) and flushes it into the next LLM turn via `before_model_call`; gated by `enable_diagnosis` (requires `enable_tracing`). Lives in `jiuwensymbiosis/rails/diagnosis.py`.
5. **SkillUseRail** — Loads built-in `SKILL.md` docs and appends `RobotControlTool`; attached only when `RobotAgentConfig.enable_skill=True`. (`rails/__init__.py` re-exports SafetyRail / RecoveryRail / VisualFeedbackRail / DiagnosisRail; `SkillUseRail` lives in `agent/builder.py`.)

Note: `TraceRail` (see "Execution Trace & Replay" below) is another parallel rail that lives in `jiuwensymbiosis/agent/trace.py` — **not** under `rails/` — and is gated by `enable_tracing` rather than a safety flag.

Rails are enabled/disabled via `RobotAgentConfig` flags and gated by session capabilities (e.g., VisualFeedbackRail requires `vision.camera`; SafetyRail attaches when **any** of `motion.cartesian` / `motion.joint` / `motion.base` / `motion.lift` / `motion.waist` is present, so joint-only arms and gripperless mobile bodies both get a pre-check).

### Two-Tier Autonomous Planning (`exec_mode: fastagent`)

`agent/fast/planner.py:plan_task` turns a task into one flat action sequence, then `run_sequence` executes it with **no per-step LLM call**:

- **Tier 1 — skill composition** (`compile_sequence`): given the world state and the capability-filtered skill library, pick the skills *and* expand their workflows into a flat sequence in **one** inference. The happy path therefore costs exactly one LLM round trip.
- **Tier 2 — action composition** (`compose_actions`): no SKILL.md at all — derive a sequence from the action contracts (`requires`/`provides`/location freshness) alone. It takes over on exactly three **decidable** conditions, never on the model's say-so: (1) no skill survives the capability gate, (2) Tier 1 returns an explicit empty array (it is instructed to when nothing fits), (3) Tier 1 exhausts its correction retries — which subsumes "the chosen skills' pre-conditions cannot hold from the current state", since `parse_sequence` rejects that expansion and feeds the reason back.
- **`parse_sequence`** (`agent/fast/sequence.py`) is the safety net between them: it simulates the state forward, checks `requires ⊆ state`, validates every `<bind>.field` against the producing op's `returns`, and names *which action would produce* a missing pre-condition so the compiler's retry loop can self-correct. It rejects unmet pre-conditions, **not** orderings — any permutation that type-checks is accepted.
- **Runtime re-planning** (`runner.py`): before each step the runner re-measures `WorldState` and re-plans (capped by `max_replans`) when the world *contradicts* the next step's pre-conditions. Contradiction, not absence — a body that cannot report a token is ignorant, not falsified, and treating ignorance as falsehood would re-plan forever.

Distilling a successful Tier 2 sequence back into a SKILL.md is **not implemented**; today a new skill is authored by hand (see `docs/autonomous-planning.md`).

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
2. `lowlevel.py` — Driver implementing the **capability-sliced** Protocols in `env/protocol.py`: `RobotDriver` is only `close()`, and a body implements the slices matching what it declares — `CartesianDriver` / `JointDriver` / `ServoDriver` / `BaseDriver` / `ContinuousBaseDriver` / `LifterDriver` / `WaistDriver` / `DualArmDriver` / `CameraDriver` / `SuctionDriver` / `GripperDriver` / `VisionDriver`. A mobile dual-arm body owes nothing to the single-arm Cartesian contract.
3. `env.py` — `BaseRobotEnv` subclass: `capabilities` frozenset, `connect`/`disconnect`/`get_observation`, expose `home_pose`/`tool_offset_mm` plus whichever SafetyRail envelopes the hardware can actually state (`z_min_safe`/`workspace_bounds`/`joint_limits`/`base_step_limits`/`lift_limits`/`waist_step_limit_rad`; each defaults to `None` = unchecked), and `holding_payload` if the body can carry something
4. `api.py` — Multi-inherits Mixins + `BaseRobotApi`; overrides geometry-specific methods, implements vision methods
5. `session.py` — `make_builder(cfg_cls, env_cls, api_cls, ...)` one-liner; `api_kwargs_from_cfg` accepts a declarative list (`["cfg_attr"` or `"cfg_attr:api_kwarg"`, dotted paths OK) so same/near-named cfg→Api field mapping needs no hand-written extractor, and `make_detector_sidecar()` provides the standard detection-server sidecar
6. `config_template.yaml` — YAML template with Chinese annotations

Template at `templates/xxx_adapter/`. Validate statically with `scripts/validate_adapter.py`; smoke-test runtime behavior (every `@robot_tool` callable + JSON-serializable) with `scripts/smoke_test_adapter.py`.

### Visual Perception Pipeline

Detection runs as a subprocess (GroundingDINO + SAM2) via `perception/detector_sidecar.py`. `RobotSession` manages lifecycle. The body-agnostic `perception/` package provides the shared pipeline: `detector_client.init_detector()`, `vision.detect_and_centroid()`, `vision.apply_xy_correction()`, `object_geometry` (mask → 3D extent), and `scene3d` (the detect → centroid → project → correct → geometry chain behind `Scene3DMixin`).

### Workspace Resolution

Priority: explicit `workspace` arg > `$JIUWENSYMBIOSIS_WORKSPACE` env var > `~/.jiuwensymbiosis/settings.json` > `~/.jiuwensymbiosis/{session.name}_workspace/`

## Source Tree Layout

```
jiuwensymbiosis/          # Main package
  agent/                  # RobotSession, build_robot_agent, RobotAgentConfig, ModelSpec, MockModel (--mock)
    fast/                 # Two-tier planner (plan_task), sequence validator, runner, skill registry
  api/                    # BaseRobotApi, actions.py (shared ActionSpec vocabulary),
                          #   @implements / @robot_tool decorators, capability mixins
                          #   state.py (state vocabulary), memory.py (ExecutionMemory), world_state.py
  env/                    # BaseRobotEnv, MockArmEnv, KNOWN_CAPABILITIES, protocol.py (driver Protocols)
  tools/                  # build_robot_tools, RobotControlTool, InProcessCodeTool
  rails/                  # SafetyRail, RecoveryRail, VisualFeedbackRail, DiagnosisRail
  skills/                 # Built-in SKILL.md files (visual_pick, visual_place, transport)
  perception/             # Body-agnostic vision: frame, calibration, object_geometry, scene3d
  motion/                 # Body-agnostic base motion: base_goal, approach, diff_drive
  adapters/
    piper/                # Piper 6-DoF reference adapter (6-DoF + gripper + wrist vision)
    so101/                # SO-101 5-DoF arm (gripper + eye-to-hand camera)
    cruzr/                # Cruzr mobile dual-arm (base + lifter + waist + paddle grasp)
    _common/              # Shared adapter utilities (builder, detector, vision, calibration)
  serving/                # Visual perception server subprocess (GroundingDINO + SAM2)
  introspect.py           # Machine-readable actions / skills / state views (the CLI's backend)
  utils/                  # proxy hygiene (proxy.py), centralised logging (logging.py)
configs/{piper,so101,cruzr}/  # Per-body YAML; the top-level `adapter:` key picks the session builder
templates/xxx_adapter/    # Adapter skeleton for new hardware
tests/
  unit_tests/             # Mirrors package structure
  mocks/                  # MockApi, MockEnv, MockDriver, MockScene
  integration/            # Hardware/GPU-dependent tests
scripts/validate_adapter.py  # Static compatibility checker for new adapters
scripts/smoke_test_adapter.py # Runtime smoke test: drive each @robot_tool with MockEnv
examples/                 # Runnable demo (run_task — generic runner)
docs/                     # Diátaxis docs: zh/en tutorial, how-to, reference, explanation
design/                   # Internal design and migration records
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
