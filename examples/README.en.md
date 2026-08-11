# Examples

English | [中文](README.md)

These examples run directly from the repository root. Install dependencies using the English [README](../README.md) first.

## Piper Mock Agent

This command checks the Session, Agent, Skills, Tools, and Rails wiring with an in-memory arm and offline model. It needs no robot, GPU, external service, or API key:

```bash
python examples/piper_pick_demo.py \
  --config configs/piper/piper.yaml \
  --mock \
  --max-iter 1 \
  --no-visual-feedback \
  --workspace /tmp/jiuwensymbiosis-agent-demo \
  --query "Pick up the black box and place it on the white box."
```

The result should contain `"mock: no real model, task skipped"` and the process should exit with status `0`. The fixed offline model does not call robot tools, so this command validates Agent wiring rather than successful physical manipulation.

## Piper Hardware

```bash
python examples/piper_pick_demo.py \
  --config configs/piper/piper.yaml \
  --query "Pick up the black box and place it on the white box." \
  --api-key "$OPENJIUWEN_API_KEY"
```

Before running, validate CAN, the Piper SDK, camera, detector service, workspace, and safety bounds. Do not leave a robot unattended in an unvalidated workspace.

## SO-101 Hardware

SO-101 requires Python 3.12, LeRobot 0.6.x, motor calibration, and a valid eye-to-hand calibration. First copy the supplied configuration to an untracked local file:

```bash
cp configs/so101/so101.yaml configs/so101/so101.local.yaml
```

The supplied configuration contains example values from an accepted device. After copying it, set `safety_validated` to `false`, then enter the local serial port, camera serial number, calibration path, and safety bounds. Set it back to `true` only after validating joint limits, the workspace, and E-stop behavior. Then run:

```bash
python examples/so101_pick_demo.py \
  --config configs/so101/so101.local.yaml \
  --query "Pick up the banana on the table." \
  --fast \
  --no-visual-feedback
```

See the [SO-101 configuration template](../jiuwensymbiosis/adapters/so101/config_template.yaml) for deployment fields and defaults.

## Sample Trace

[`sample_trace/`](sample_trace/README.md) contains a sanitized trace JSON file, HTML replay, and step images. It demonstrates trace artifacts and is not a robot-correctness baseline.
