# Examples

English | [中文](README.md)

These examples run directly from the repository root. Install dependencies using the English [README](../README.md) first.

## Generic task runner

`examples/run_task.py` is the single entry point for every robot and task: `--config` picks the robot (the YAML's top-level `adapter:` field selects it from the registry), and `--query` (or `--voice`) supplies the task, which is not in the config. The execution mode comes from the YAML's `agent.execution.mode`, defaulting to `fastagent` (compile once, no per-step LLM); add `--stepagent` for single-step debugging.

Both examples below are verified on real hardware. Before running you must complete hardware, calibration, detector-service, and safety-boundary acceptance. Do not run unattended in an unvalidated workspace.

### SO-101 real hardware

SO-101 requires Python 3.12, LeRobot 0.6.x, motor calibration, and a valid eye-to-hand calibration.

`configs/so101/so101.yaml` carries example values from an accepted device. Before touching hardware, set `safety_validated` to `false`, then fill in the serial port, camera serial, calibration paths, and local safety bounds. Only after passing limits, workspace, and E-stop acceptance may you set it back to `true`. Then run:

```bash
python examples/run_task.py \
  --config configs/so101/so101.yaml \
  --query "Put the banana on the plate." \
  --api-key "$OPENJIUWEN_API_KEY"
```

Deployment fields and defaults are in the [SO-101 config template](../jiuwensymbiosis/adapters/so101/config_template.yaml).

### Cruzr real hardware

Cruzr requires a ROS 2 (Jazzy) workspace, waist/head cameras, calibration, and a detector service.

In `configs/cruzr/cruzr.yaml`, keep `adapter: cruzr`. Fill in the ROS workspace path, camera topics, calibration file, URDF path, detector service address, and the orchestration LLM endpoint, then validate the safety bounds. Before running, `source` the ROS + Cruzr workspace and start the detector service. Then run:

```bash
python examples/run_task.py \
  --config configs/cruzr/cruzr.yaml \
  --query "Carry the white box on top of the brown box to the white table with the banana on it." \
  --api-key "$OPENJIUWEN_API_KEY"
```

> To keep serial ports, device serials, and keys out of `git status`, copy to `configs/<robot>/<robot>.local.yaml` and edit that instead — `*.local.yaml` is gitignored — then point `--config` at the copy.

Cruzr declares `motion.base`, so the `transport` skill passes the capability gate; a fixed arm has no mobility, and the same task composes into a different sequence.

## Sample trace

[`sample_trace/`](sample_trace/README.md) holds a sanitized trace JSON, HTML replay, and step images, for understanding trace artifacts. It is not a correctness baseline for the robot.
