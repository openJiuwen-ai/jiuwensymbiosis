# Calibrate an SO-101 with a Fixed Camera (Eye-to-Hand)

> Category: How-to. The [Chinese source](../../zh/how-to/calibrate-so101-eye-to-hand.md) is authoritative.

This guide applies only to the following setup:

- Robot arm: SO-101 with five arm joints;
- Camera: fixed to a table, stand, or near the robot base and does not move with the arm;
- Calibration arrangement: eye-to-hand;
- Calibration board: rigidly attached to the gripper and moves with the arm;
- Solved transform: `T_base_cam`, the fixed transform from the camera frame to the robot base frame.

The workflow has three stages:

| Stage | CLI mode | Purpose | Hardware required |
|---|---|---|---|
| 1 | `--collect-poses` | Manually teach a set of joint waypoints | Yes |
| 2 | `--auto` | Replay the waypoints, capture images, solve, and attempt to publish a production calibration | Yes |
| 3 | `--replay` | Solve again offline from a station archive | No |

Stages 1 and 2 are required for an initial calibration. Stage 3 is useful for trying another solver, adjusting
thresholds, or reviewing the result.

## Consider the GUI wizard first

If this is your first calibration, use the graphical wizard instead of the CLI below:

```bash
python -m jiuwensymbiosis_gui --gui workbench    # then open 「工具 → 手眼标定」
```

The wizard folds every step of this guide into four, and adds three things the CLI has no place for:
it **generates a printable board PDF** (with a verification ruler, so print scaling cannot silently
skew your result), it **shows how many board corners are detected while you teach** (the CLI can only
be driven blind, so bad poses surface only after capture finishes), and it **maps each quality-gate
failure to a concrete corrective action**. It drives the same calibration code as this guide and
produces an identical artifact.

Use this guide when you need fine-grained parameter control, scripting, or offline re-solving.

---

## 1. Safety and hardware preparation

Before starting, confirm that:

1. The SO-101 has completed LeRobot motor calibration, and the configured `robot_id` matches the ID used for that
   calibration. See the [official LeRobot calibration instructions](https://huggingface.co/docs/lerobot/so101#calibrate).
2. You have identified the serial port with `lerobot-find-port`.
3. The fixed-camera mount is secure. Do not move, rotate, or refocus the camera during collection.
4. The calibration board is attached rigidly to the gripper and cannot loosen during collection.
5. The camera covers the main SO-101 workspace and its USB connection is stable.
6. An emergency stop or power cutoff is within reach, and the area around the arm is clear before automatic
   collection.
7. Manual teaching releases torque on the five arm joints. Gravity has a noticeable effect on the SO-101 elbow, so
   support the arm with your other hand.

> The calibration configuration uses wider joint limits and relaxed workspace limits. It is suitable only for
> controlled calibration. Do not copy these safety settings directly into a normal runtime configuration.

---

## 2. Install dependencies

The SO-101 adapter requires Python 3.12. Install the calibration and SO-101 dependencies together in the project
environment:

```bash
python -m pip install -e ".[gui,calib,so101]"
```

The `gui` extra installs NiceGUI for the existing workbench. If you only run the headless CLI workflow below,
install `.[calib,so101]` instead.

To run the calibration subsystem tests:

```bash
pytest tests/unit_tests/calibration/
```

---

## 3. Prepare a ChArUco calibration board

A 5×7 ChArUco board is recommended. If you do not already have one, use the shared board generator:

```bash
python -c 'from scripts.calibrate.handeye_board import BoardSpec, generate_board_image; generate_board_image(BoardSpec("charuco", 5, 7, 20.86, 15.2), "tmp/so101_charuco.png")'
```

Printing and mounting requirements:

1. Print at 100% scale with “fit to page” and automatic scaling disabled.
2. After printing, measure the actual square and marker side lengths with a ruler.
3. Use those measurements for `--square-size-mm` and `--marker-size-mm` in every later command.
4. Mount the paper flat on a rigid plate, avoiding bends, curling, and glare.
5. Attach the plate rigidly to the gripper, minimizing occlusion of markers or corners by the gripper.

All examples below assume these measured parameters:

```text
board=charuco
squares-x=5
squares-y=7
square-size-mm=20.86
marker-size-mm=15.2
```

If your measurements differ, update the Stage 2 collection command accordingly. An incorrect square size directly
causes an incorrect translation scale.

---

## 4. Find the SO-101 port and camera serial number

Find the robot-arm serial port. As instructed by LeRobot, you will need to unplug the arm's USB cable during this
procedure:

```bash
lerobot-find-port
```

Find the RealSense serial number:

```bash
rs-enumerate-devices | grep "Serial Number"
```

You can also use Python:

```bash
python -c "import pyrealsense2 as rs; c=rs.context(); print([d.get_info(rs.camera_info.serial_number) for d in c.devices])"
```

---

## 5. Configure the calibration environment

Use `scripts/calibrate/so101_calibrate.yaml` as the template. Copy it to a local configuration so that device paths
and serial numbers are not committed to the repository:

```bash
cp scripts/calibrate/so101_calibrate.yaml scripts/calibrate/so101_calibrate.local.yaml
```

At minimum, review these fields:

```yaml
env:
  cfg:
    low_level:
      name: "so101"
      port: "/dev/your_port"
      robot_id: "so101_left"
      calibration_dir: null

      # Use the actual joint angles at connection time as home during calibration.
      home_use_init_pose: true
      disable_torque_on_disconnect: false

      joint_limits:
        shoulder_pan: [-180.0, 180.0]
        shoulder_lift: [-180.0, 180.0]
        elbow_flex: [-180.0, 180.0]
        wrist_flex: [-180.0, 180.0]
        wrist_roll: [-180.0, 180.0]

      # Mount arrangement for the fixed camera. SO-101 calibration only allows eye_to_hand.
      camera_mount: "eye_to_hand"
      camera_serial: "<your-camera-serial>"
      camera_resolution: [640, 480]
      camera_fps: 30

calibration:
  adapter_module: "jiuwensymbiosis.adapters.so101"
  trajectory:
    space: joint
  output: "tmp/so101_eye_to_hand.json"

  observability:
    min_relative_rotation_deg: 5.0
    min_axis_separation_deg: 15.0
    min_max_rotation_deg: 20.0
    min_translation_baseline_mm: 30.0
    duplicate_rotation_deg: 2.0
    duplicate_translation_mm: 5.0

  # SO-101 servos may have about 3.5 degrees of steady-state error under gravity load.
  capture_gate:
    reach_rotation_deg: 4.0
    reach_translation_mm: 0.5
    exposure_rotation_deg: 1.0
    exposure_translation_mm: 0.5
```

Do not put `camera_mount` in the top-level `calibration:` section. Its single source of truth is
`env.cfg.low_level.camera_mount`.

The remaining commands use this variable:

```bash
export CALIB_CONFIG=scripts/calibrate/so101_calibrate.local.yaml
```

If you prefer not to use a shell variable, replace `$CALIB_CONFIG` in each command with the configuration path.

---

## 6. Stage 1: manually teach waypoints

Before running the following command, support the arm so it cannot fall when torque is released. The command releases
torque on the arm joints. Read the interaction flow and pose requirements below before starting.

Run:

```bash
jiuwensymbiosis-calibrate-eye-to-hand \
  --config "$CALIB_CONFIG" \
  --collect-poses tmp/so101_wp.npz
```

`--collect-poses` records only joint waypoints; it does not capture calibration images, so board parameters do not
participate in this stage.

Interaction flow:

1. The CLI connects to the SO-101.
2. The program disables torque on the five arm joints while leaving gripper torque enabled.
3. Support the elbow and slowly move the arm to the first pose.
4. Confirm that the entire calibration board is visible to the fixed camera, then press Enter to record it.
5. Repeat the movement and recording steps. Prepare 12–20 waypoints.
6. When finished, type `q` and press Enter.
7. The program synchronizes the current joint targets, restores torque, and writes `tmp/so101_wp.npz`.

Teaching-pose requirements:

- Do not let the calibration board contact the table or another object while moving the arm. Movement or deformation
  of the board affects the calibration result.
- Exercise at least two different rotation axes; do not rotate around only one axis.
- Aim for a maximum relative rotation greater than 20°.
- Aim for more than 30 mm of calibration-board translation in the camera frame.
- Cover the center, left, right, front, back, and multiple distances in the field of view. Do not cluster every pose in
  one small region.
- Keep the whole board sharp, correctly exposed, and as unobstructed by the gripper as possible at every pose.
- Do not record nearly identical poses.

If torque restoration fails, the program raises `ManualGuidanceRecoveryError`. Continue supporting the arm, stop the
automatic workflow, check the motor bus, and manually return the robot to a safe pose.

---

## 7. Stage 2: dry run and automatic collection

### 7.1 Run a dry run first

The dry run validates the archive type, adapter, mount, joint order, units, periodicity, and trajectory interpolation.
It does not move the robot or capture images:

```bash
jiuwensymbiosis-calibrate-eye-to-hand \
  --config "$CALIB_CONFIG" \
  --auto tmp/so101_wp.npz \
  --n-stations 20 \
  --dry-run \
  --out tmp/so101_eye_to_hand.json
```

Proceed to live automatic collection only after the dry run succeeds.

### 7.2 Collect, solve, and publish automatically

After confirming that the emergency stop is reachable and the workspace is clear, run:

```bash
jiuwensymbiosis-calibrate-eye-to-hand \
  --config "$CALIB_CONFIG" \
  --board charuco \
  --squares-x 5 --squares-y 7 \
  --square-size-mm 20.86 --marker-size-mm 15.2 \
  --auto tmp/so101_wp.npz \
  --n-stations 20 \
  --confirm-estop \
  --cross-check \
  --min-corners 16 \
  --out tmp/so101_eye_to_hand.json
```

The automatic workflow performs these steps:

1. Validate the waypoint archive's adapter, mount, and joint metadata.
2. Interpolate the trajectory with joint steps no greater than 5° by default.
3. Capture a frame before the first movement to verify the fixed camera, intrinsics, and image data.
4. Move to a sampling position and wait for the robot to settle.
5. Check the difference between the actual and target poses.
6. Capture an image and detect the ChArUco board.
7. Check for robot drift before and after exposure.
8. Record stations that pass the gates and write the station archive.
9. Solve `T_base_cam` and run observability and rigidity-consistency gates.
10. Run a reload smoke test through the SO-101 runtime loader; publish the production calibration only if every check
    passes.

`--n-stations` is both the target sample count and the attempt limit; it does not guarantee that the same number of
stations will be accepted. Stations are skipped when board detection fails, the reach error is too large, or the robot
drifts during exposure.

### 7.3 Output files and exit codes

On success, the workflow creates:

- `tmp/so101_eye_to_hand.json`: production schema-2 calibration with top-level `T_base_cam`;
- `tmp/so101_eye_to_hand.stations.npz`: self-describing station archive for offline re-solving.

If solving completes but a quality gate or reload smoke test fails, it creates:

- `tmp/so101_eye_to_hand.candidate.json`: REVIEW-only artifact whose matrix is at `candidate.T_base_cam`; the runtime
  loader will not load it.

If there are fewer valid stations than the solver minimum, the program may return REVIEW status without creating a
candidate file. Inspect the logs and station archive, then teach new poses or improve imaging before collecting again.

Exit codes:

| Exit code | Meaning |
|---|---|
| `0` | Production publication succeeded, or the dry run succeeded |
| `1` | Execution error |
| `2` | Preflight contract validation failed |
| `3` | REVIEW/candidate only; no production calibration was published |

Use the output as a runtime calibration only when the exit code is `0` and the output JSON contains top-level
`T_base_cam`.

---

## 8. Stage 3: solve again offline (optional)

### 8.1 Without a configuration: candidate only

```bash
jiuwensymbiosis-calibrate-eye-to-hand \
  --replay tmp/so101_eye_to_hand.stations.npz \
  --method HORAUD \
  --cross-check \
  --out tmp/so101_eye_to_hand_horaud.json
```

Without `--config`, the command does not publish a production calibration. It creates only
`tmp/so101_eye_to_hand_horaud.candidate.json`, and the expected exit code is `3`.

### 8.2 With a configuration: production publication allowed

```bash
jiuwensymbiosis-calibrate-eye-to-hand \
  --replay tmp/so101_eye_to_hand.stations.npz \
  --config "$CALIB_CONFIG" \
  --method HORAUD \
  --cross-check \
  --out tmp/so101_eye_to_hand_horaud.json
```

This mode does not connect to the robot or camera, but uses the configuration to confirm the adapter and
`camera_mount`, then runs a reload smoke test through the SO-101 loader. It writes a production JSON only when both the
quality gates and reload smoke test pass.

Available solvers include `PARK`, `TSAI`, `HORAUD`, `ANDREFF`, and `DANIILIDIS`. Prefer `PARK` as the primary method and
use `--cross-check` to identify substantial disagreement between methods. Do not disregard sampling quality merely
because one method passes the gates.

---

## 9. Apply the calibration to the SO-101 runtime configuration

First copy the production calibration file to a stable location, for example:

```bash
mkdir -p configs/so101/calibration
cp tmp/so101_eye_to_hand.json configs/so101/calibration/so101_eye_to_hand.json
```

Then set these fields under `env.cfg.low_level` in the SO-101 runtime configuration:

```yaml
camera_mount: "eye_to_hand"
camera_serial: "<fixed-camera serial used during calibration>"
calib_path: "configs/so101/calibration/so101_eye_to_hand.json"
```

Validate the production JSON before using it:

```bash
python -c 'import json; p=json.load(open("configs/so101/calibration/so101_eye_to_hand.json")); assert p.get("schema_version")==2 and "T_base_cam" in p; print("calibration artifact OK")'
```

A calibration artifact corresponds to one physical installation. Recalibrate after any of these changes:

- The fixed camera or its mount moves;
- The SO-101 base position or orientation changes;
- The board mounting on the gripper loosens or changes before collecting again;
- The camera is replaced, its intrinsics change, or its resolution mode changes.

---

## 10. Troubleshooting

| Symptom | Common cause | Recommended action |
|---|---|---|
| `ManualGuidanceRecoveryError` | Torque restoration failed after teaching | Support the elbow, stop the automatic workflow, check the motor bus, and manually restore a safe pose |
| `camera preflight failed` | Wrong serial number, unstable USB, or unreadable intrinsics | Check `camera_serial`, the USB connection, the RealSense driver, and whether another process is using the camera |
| `board not detected` | Board outside the field of view, occluded corners, glare, blur, or wrong dimensions | Teach new poses, improve lighting, reduce movement speed, and verify the board specification |
| `reach gate failed` | Excessive steady-state servo error or unreachable waypoint | Inspect the actual pose and mechanical load; do not blindly keep increasing `reach_rotation_deg` |
| `exposure drift` | Servos continue drifting during capture | Check mechanical load, power, and joint stability, and improve the settled state |
| `observability_flange_axes` | Poses rotate around only one axis | Teach again with movement around at least one additional rotation axis |
| `observability_camera_axes` | Calibration-board motion relative to the fixed camera is degenerate | Add board tilt, rotation, and in-frame translation; confirm rigid board attachment |
| `observability_*_trans` | Translation coverage is too small | Increase front/back, left/right, or near/far variation within the workspace |
| `observability_duplicates` | Waypoints are too similar | Remove duplicate poses and collect again |
| Only a candidate is created | A quality gate or reload smoke test failed | Inspect the candidate's `reasons`, correct the sampling problem, then collect or solve again |
| Projection has a fixed offset after calibration | Incorrect board size, moved camera/base, or inconsistent coordinate mounting | Measure the board again, confirm that the hardware has not moved, then recalibrate |

Add `--debug` to the end of a command for more detailed logs. Do not manually copy the matrix from a candidate into a
production schema-2 file to bypass quality gates and the reload smoke test.

---

## 11. Command quick reference

```bash
# 1. Teach waypoints
jiuwensymbiosis-calibrate-eye-to-hand \
  --config "$CALIB_CONFIG" \
  --collect-poses tmp/so101_wp.npz

# 2. Preflight without movement
jiuwensymbiosis-calibrate-eye-to-hand \
  --config "$CALIB_CONFIG" \
  --auto tmp/so101_wp.npz --n-stations 20 --dry-run \
  --out tmp/so101_eye_to_hand.json

# 3. Collect, solve, and publish with real hardware
jiuwensymbiosis-calibrate-eye-to-hand \
  --config "$CALIB_CONFIG" \
  --board charuco --squares-x 5 --squares-y 7 \
  --square-size-mm 20.86 --marker-size-mm 15.2 \
  --auto tmp/so101_wp.npz --n-stations 20 \
  --confirm-estop --cross-check \
  --out tmp/so101_eye_to_hand.json

# 4. Solve offline and rerun publication gates
jiuwensymbiosis-calibrate-eye-to-hand \
  --replay tmp/so101_eye_to_hand.stations.npz \
  --config "$CALIB_CONFIG" \
  --method HORAUD --cross-check \
  --out tmp/so101_eye_to_hand_horaud.json
```
