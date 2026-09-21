# Command-Line Reference

> Category: Reference. Console entry points are defined by `[project.scripts]` in `pyproject.toml`.

## jiuwensymbiosis-run (generic task runner)

```bash
jiuwensymbiosis-run --config configs/cruzr/cruzr.yaml --query "把箱子搬到桌上"
jiuwensymbiosis-run --config configs/piper/piper.yaml   --query "把瓶子放到左边"
```

`--config` is required; the YAML's top-level `adapter:` field selects the robot (`--robot` overrides). Tasks normally come from `--query` (configs do not embed tasks, although the YAML retains an optional `env.cfg.prompt` fallback); startup fails when both are empty. Other common overrides:

| Option | Effect |
|---|---|
| `--robot` | Override the config's `adapter:` field (piper / so101 / cruzr) |
| `--mock` | In-memory dry run for Piper only (MockArmEnv + offline model); implies `--stepagent` |
| `--stepagent` | Force per-step LLM (single-step debugging); default is `fastagent` (compile once, no per-step LLM) |
| `--voice` / `--voice-text` / `--voice-audio-file` / `--voice-once` / `--no-wake` / `--tts` / `--asr-device` | Voice mode |
| `--no-skill` | Disable SkillUseRail + the robot_control dispatcher |
| `--mode` | `tool` / `code` / `hybrid` |
| `--no-visual-feedback` | Disable VisualFeedbackRail |
| `--server-url` / `--model` / `--api-key` | Override LLM endpoint/model/key |
| `--max-iter` / `--workspace` / `--debug` | Iteration cap, workspace, log level |

`--mock` uses an offline model and a Mock environment; `--control-hz`/`--servo-step-mm` tune the fastagent real-time servo.

## piper-pick-demo

```bash
piper-pick-demo --config PATH [--query TEXT | --voice ...] [--mock]
```

Back-compat alias pointing at the same generic `run_task.py` (`jiuwensymbiosis-run`).

## jiuwensymbiosis-replay

```bash
jiuwensymbiosis-replay TRACE_JSON [--open] [--text]
```

Renders a self-contained HTML replay by default and prints its path; `--open` additionally launches the HTML in the
default browser; `--text` outputs a terminal timeline.

## jiuwensymbiosis-gui

```bash
jiuwensymbiosis-gui --list-guis
jiuwensymbiosis-gui --gui workbench --config configs/piper/piper.yaml
python -m jiuwensymbiosis_gui --gui workbench
```

`--gui` selects an installed GUI plugin and defaults to `workbench`; `--list-guis` lists available plugins. `--config`,
`--workspace`, `--host`, `--port`, and `--no-browser` configure the selected app. `--gui-config` passes plugin-specific
settings when that plugin supports them. The launcher loads only the selected plugin's application entry point. Install
that plugin and its documented dependencies in the same Python environment; the `.[gui]` extra supplies the current
workbench's NiceGUI dependencies.

`python -m jiuwensymbiosis.gui` remains as a temporary compatibility shim and forwards to the unified launcher.

## Hand-eye calibration

Installing `.[calib]` provides three entry points:

```bash
jiuwensymbiosis-calibrate-hand-eye --collect-poses OUTPUT --config RUNTIME_YAML [--import-poses INPUT]
jiuwensymbiosis-calibrate-hand-eye --auto WAYPOINT_ARCHIVE --config RUNTIME_YAML --confirm-estop
jiuwensymbiosis-calibrate-hand-eye --replay STATION_ARCHIVE [--config RUNTIME_YAML]
```

`jiuwensymbiosis-calibrate-hand-eye` is the mount-neutral entry point; the camera mount comes from the runtime config or archive and cannot be overridden on the command line. `--collect-poses --import-poses INPUT` connects to no hardware and only normalizes a self-describing waypoint archive into OUTPUT. `jiuwensymbiosis-calibrate-eye-to-hand` runs the same flow but requires `eye_to_hand`. `jiuwensymbiosis-calibrate-eye-in-hand` requires `eye_in_hand` for archive modes (`--collect-poses`, `--replay`, or parameterized `--auto`); other invocations use the legacy Piper wizard and print a compatibility warning.

Exit codes: `0` published (or dry-run passed), `1` execution error, `2` preflight contract failure, `3` only an unloadable REVIEW/candidate report was produced.

The GUI's 「工具 → 手眼标定」 wizard drives these same workflows and additionally generates a printable board PDF and shows live corner detection while teaching; start there for a first calibration.

## Introspection tools (jiuwensymbiosis-actions / -skills / -state)

```bash
jiuwensymbiosis-actions --vocabulary [--json]                       # shared action vocabulary (no robot)
jiuwensymbiosis-actions --config configs/cruzr/cruzr.yaml [--json]  # that vocabulary gated to one body
jiuwensymbiosis-skills  [--json]                                    # skill library + contracts
jiuwensymbiosis-state   --config configs/cruzr/cruzr.yaml [--json]  # live world state (connects!)
```

These three are the machine-readable views a planner / coding agent reads (see [Architecture: two-tier planning](../explanation/architecture.md#two-tier-planning)): what an action is, what a skill's pre-conditions are, and where the current world stands.
