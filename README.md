# JiuwenSymbiosis

English | [中文](README.zh.md)

JiuwenSymbiosis is an embodied-agent framework built on openjiuwen for adapting one safe, auditable Agent workflow to different robot bodies.

## Core Features

- **Body agnostic**: A shared action contract (`ActionSpec`) plus capability gating, with an adapter as a bridge, separate robot geometry and vendor SDKs from Agent workflows.
- **Task composition**: A task is not a hard-coded call sequence — it is dynamically composed from action contracts (`ActionSpec`) and skills. The planner is what organizes a goal into an action sequence.
- **Environment- and body-aware dynamic orchestration**: The planner takes the current body state and ambient sensing as input and dynamically composes the matching action contracts and skills. During execution, if the real state conflicts with the next step's pre-conditions, it re-plans automatically rather than blindly running the established flow.
- **Execution memory**: The planner always knows what is currently known (target locations, body state), maintained automatically by action contracts — sensing books it in, a move invalidates it, no hand-written cache; state established by executed actions settles and is inherited by later steps.
- **Real-time tracking servo**: Perceive and act together — keep tracking a target and stream high-frequency servo commands to the body; used for the critical grasp/place steps and can follow a moving object for real-time grasping.
- **Active search**: When the target is not in view, the body searches in place, reports a bearing, and converges step by step; when unreachable from here it first plans a reachable pose.
- **Reachability reasoning**: When the task describes a target with a spatial relation such as "in the drawer" or "under the crate", the planner judges from that relation whether the target is actually reachable; if the target is occluded or enclosed, it plans first to make the target reachable.
- **Action contracts**: Every action declares pre-conditions, effects, and result shape; a contract never encodes an order — only a permutation whose pre-conditions hold is accepted.
- **Safety loop**: Motion-bounds checks, failure recovery, visual feedback, and execution diagnosis protect physical execution.
- **General visual perception**: An out-of-the-box reusable perception pipeline — frame capture, depth, open-vocabulary detection and segmentation, centroid and depth estimation, pixel-to-base back-projection, calibration, coordinate transforms and correction; eye-in-hand and eye-to-hand differ by one projection function.
- **Skill workflows**: Built-in `visual_pick`, `visual_place`, and `transport` skills standardize common manipulation procedures.
- **Auditable execution**: Structured traces, saved frames, replay, and feedback analysis make runs reproducible and diagnosable.

## Architecture

![JiuwenSymbiosis architecture](docs/images/architecture-layers.en.svg)

The runtime forms a **Perceive → Plan → Execute → Observe → Feedback** loop. Commands flow through Agent, Rails, Tools, API, Env, and Hardware; observations, failures, and trace evidence flow back to the Agent. See the [Architecture explanation](docs/en/explanation/architecture.md) for the full dependency and task sequence diagrams.

## Related Documentation

- [Documentation](docs/en/README.md) — tutorials, how-to guides, API reference, and explanations
- [Examples](examples/README.en.md) — hardware-free Mock and real-hardware examples
- [Feature Matrix](docs/en/reference/feature-matrix.md) — built-in adapter and capability status
- [Contributing](CONTRIBUTING.md) — development, testing, and contribution workflow

## Requirements

| Dependency | Version or requirement |
| --- | --- |
| Operating system | Ubuntu 22.04 (currently verified platform) |
| Python | `>=3.11,<3.14`; the SO-101 adapter requires Python 3.12 |
| Core | `openjiuwen>=0.1.13`; other versions follow `[project.dependencies]` in `pyproject.toml` |
| Vision/GPU | The `[full]` extra uses the CUDA 12.8 build of PyTorch 2.8.0 |
| Real hardware | Prepare the adapter's CAN/serial bus, camera, calibration, vendor SDK, and validated safety bounds |

## Installation

```bash
git clone https://gitcode.com/openJiuwen/jiuwensymbiosis.git
cd jiuwensymbiosis
conda create -n jiuwensymbiosis python=3.12
conda activate jiuwensymbiosis
python -m pip install -e .
```

Install only the optional capabilities you need:

```bash
python -m pip install -e ".[dev]"       # Tests and development tools
python -m pip install -e ".[piper]"     # Piper SDK
python -m pip install -e ".[so101]"     # SO-101 / LeRobot; Python 3.12
python -m pip install -e ".[cruzr]"     # Cruzr dual-arm (pinocchio arm IK; rclpy from the ROS workspace, not a pip dep)
python -m pip install -e ".[voice]"     # ASR and audio capture
python -m pip install -e ".[gui]"       # Existing workbench GUI (NiceGUI browser UI)
python -m pip install -e ".[calib]"     # Hand-eye calibration
python -m pip install -e ".[full]" \
  --extra-index-url https://download.pytorch.org/whl/cu128  # Vision/GPU stack
```

See [Installation and Quick Start](docs/en/tutorial/01-quick-start.md) for combined extras and pinned runtime dependencies.

The unified launcher selects a GUI plugin; this repository currently ships `workbench`:

```bash
jiuwensymbiosis-gui --list-guis
jiuwensymbiosis-gui --gui workbench
# Equivalent module entry point
python -m jiuwensymbiosis_gui --gui workbench
```

The `.[gui]` extra installs the workbench's dependencies. A separately distributed GUI plugin
is installed with its own documented dependencies in the same Python environment, then selected
with `--gui <key>`.

## Built-in Adapters

| Adapter | Status | Main capabilities | Optional dependencies |
| --- | --- | --- | --- |
| Piper | Built-in real adapter | 6-DoF motion, parallel gripper, eye-in-hand RealSense vision | `[piper]`; add `[full]` for vision |
| SO-101 | Built-in real adapter | 5-DoF motion, parallel gripper, eye-to-hand RealSense vision | Python 3.12 + `[so101]`; add `[full]` for vision |
| Cruzr | Built-in real adapter | Mobile dual-arm (base + lift + waist), paddle grasp, waist/head dual cameras | `[cruzr]`; add `[full]` for vision; requires sourcing the ROS workspace at runtime |

`MockArmEnv` remains available as a built-in in-memory simulation Env and powers `--mock` (Piper only), but it is not a hardware adapter.

SCARA and suction are supported extension contracts, but this repository does not currently ship a hardware-accepted built-in adapter for them. See the [Feature Matrix](docs/en/reference/feature-matrix.md) for exact activation conditions.

## Quick Start

Two hardware-verified runs. Fill in your own ports, serials, and endpoints in the config first, and accept the safety bounds.

SO-101 — table-top pick and place:

```bash
python examples/run_task.py \
  --config configs/so101/so101.yaml \
  --query "Put the banana on the plate." \
  --api-key "$OPENJIUWEN_API_KEY"
```

Cruzr — UBTECH's wheeled humanoid, mobile transport (source the ROS 2 + Cruzr workspace first):

```bash
python examples/run_task.py \
  --config configs/cruzr/cruzr.yaml \
  --query "Carry the white box on top of the brown box to the white table with the banana on it." \
  --api-key "$OPENJIUWEN_API_KEY"
```

`examples/run_task.py` is the generic task entry point: `--config` picks the robot (the YAML's top-level `adapter:` field selects it from the registry) and `--query` (or `--voice`) supplies the task, which is not stored in the config. `--mock` provides a hardware-free dry run for Piper only; other robots go through their real session. Full prerequisites for each run are in [Examples](examples/README.en.md).

When writing a Python entry point, call `clear_proxy_env()` before importing `openjiuwen` or modules that import it. The bundled CLI and examples already do this.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

This product serves solely as a workflow orchestration tool and does not embed any AI model capabilities. When users integrate AI models for specific business scenarios, they shall bear full responsibility for compliance obligations under the EU AI Act and other relevant regulatory frameworks.
