# Piper Coordinates, Safety Bounds, and Migration

This page describes Piper's contract after the tool-offset fix. Positions use mm and angles use degrees. TIP is the tool endpoint and FLANGE is the flange origin; both positions are expressed in the base frame.

## Tool transform

`tool_offset_mm = L` is a non-negative tool extension along **flange-local +Z**. Readback, commands, and driver checks share `adapters/piper/geometry.py:tip_offset_in_base`:

```text
R = Rz(rz) @ Ry(ry) @ Rx(rx)
offset = R @ [0, 0, L]
tip = flange + offset
flange = tip - offset
```

This is extrinsic xyz, or SciPy `Rotation.from_euler("xyz", ..., degrees=True)`. The vendor SDK's matrix-to-RPY formulas agree with this convention; see [Piper SDK FK](https://github.com/agilexrobotics/piper_sdk/blob/master/piper_sdk/kinematics/piper_fk.py) and [SciPy's Euler convention](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html). Agreement between software conventions does not establish hardware acceptance for the installed tool and firmware.

`goto_xyzr` and `servo_to_tip` fix `rx=180, ry=30`, using the requested or current yaw:

```text
offset = (-0.5*L*cos(rz), -0.5*L*sin(rz), -sqrt(3)/2*L)
```

For `L=95` and TIP target `(200, 20, 100)`, yaw=0 gives flange `(247.5, 20, 182.2724...)`; yaw=90 gives flange `(200, 67.5, 182.2724...)`. `get_pose()` instead uses the full live rx/ry/rz.

## Interfaces and home

| Interface | Coordinates |
|---|---|
| `api.get_pose()` / `api.get_home_pose()` | TIP; home uses its stored target orientation |
| `api.goto_xyzr()` / `api.servo_to_tip()` | TIP targets at the fixed tilt above |
| `api.move_direction()` | Translation along base axes, preserving full live orientation; precheck and returned `pose` use TIP |
| `env.get_observation().pose` | TIP, also used by planning state, the state CLI, and trace poses |
| `env.get_observation().extra["flange_pose"]` | Raw FLANGE pose from the same sample |
| `env.get_flange_pose()` / `env.home_pose` / `driver.home_pose` | FLANGE |
| `env.move_to_flange()` / `env.servo_to_flange()` | FLANGE targets; the driver also checks the tool endpoint |

Without a calibration object anchor, `home_pose_xyzrxryrz_mm_deg` is already FLANGE. With a legacy `object.xyz_base_mm` anchor, the physical home target is preserved: flange Z is `anchor_z + home_lift_mm + L`, XY comes from the anchor, and orientation comes from the connection-time pose. `home_use_init_pose=True` overrides this with the full connection-time flange pose. The presence of an anchor determines the configuration source; a camera calibration file alone does not change frame semantics.

`get_home_pose()` reads the target without moving hardware. Before connection or after disconnection it raises a clear `RuntimeError`. `home()` sends the stored flange target through the same checks as other Cartesian commands. A home outside these target limits is rejected.

Initialization diagnoses home against the same target bounds and command quantization: static configuration homes are checked before CAN prechecking; startup snapshots and homes inheriting live orientation are checked after readback. An out-of-bounds home logs the specific reason and warns that ordinary and recovery homing will be rejected, while preserving connection for observation or joint calibration. Malformed or non-finite configuration still rejects initialization. Diagnosis does not modify home or replace checks on each `home()` call. Passing does not guarantee IK reachability or a collision-free return path.

`move_direction()` derives its target from one live flange sample, preserves rx/ry/rz, then converts it to TIP for prechecking. `env.move_to_flange()` and raw flange servo share pose normalization: native `FlangePose`, `Mapping` (including read-only mappings), and attribute objects are accepted. Fields are x/y/z/rx/ry/rz; `r` aliases yaw, with `rz` taking precedence when both are supplied. Blocking motion requires complete position and orientation; servo alone may fill missing orientation from one live sample. Normalizing a complete target never requires readback. Calibration still uses the separate flange read/write interfaces.

`api.servo_to_tip()` requires a mapping with complete, non-None XYZ coordinates. Missing fields raise a descriptive error before readback or dispatch. TIP and FLANGE entry points share yaw-alias resolution, while each caller supplies live orientation only when needed.

The observation's TIP `pose` and `extra["flange_pose"]` are published together from one flange sample. Read or TIP-conversion failure leaves both as `None` and logs a diagnostic; camera and joint observations are collected independently.

Cartesian commands always use MOVE_P and do not promise a straight path. The driver's `joint` parameter is deprecated, retained for compatibility, and does not affect the motion mode.

## Safety bounds

- `z_min_safe` always means the **target TIP Z floor**: `anchor_z + z_safe_margin_mm` with an anchor, otherwise `z_min_safe_mm`, regardless of home storage. It is not a whole-robot collision plane and need not equal an object's top surface. With the tool pointing upward, flange Z below this value alone does not reject a target whose TIP and other target bounds comply.
- The driver checks TIP Z using the **target orientation**. Tilted tools no longer incur an unnecessarily conservative full-length L floor. A `grasp_z` clamped to this floor by `get_grasp_info_simple()` is commandable when the other bounds also hold.
- X/Y bounds apply to both TIP and FLANGE; `z_max_mm` applies to FLANGE. `None` disables one side. When both sides are configured, `min <= max` is required; reversed bounds are rejected before CAN connection. Violations raise `SafetyViolationError` with code `safety_rejected` before `EndPoseCtrl`, replacing silent clamping. The servo controller reports `stopped` with this error code.
- `flange_z_min_safe = z_min_safe + L` is a conservative full-length reference for satisfying the TIP floor, not a clearance derived from flange or gripper geometry. Actual commands use full target orientation and may accept a lower flange target that still satisfies the TIP floor.
- SDK command resolution is 0.001 mm / 0.001°. The original request and encoded target are both checked. Near the TIP floor, Z rounds toward safety using the encoded orientation. If rounding violates the ceiling or XY bounds, the driver rejects the command instead of compensating along other axes.
- Raw flange servo accepts `FlangePose`, `Mapping` (including read-only mappings), and attribute objects. Mappings/attribute objects require complete XYZ; only missing orientation fields come from live readback, whose failure rejects the command. Missing XYZ or unrecognized objects raise an error instead of substituting the current pose. Complete targets do not require readback.
- Calibration loading, effective home parsing, and TIP-floor derivation and finiteness checks occur before CAN prechecking, SDK construction, or enabling hardware. Configuration fallbacks overridden by an anchor are ignored. With `home_use_init_pose=True`, overridden home settings are not parsed, while the effective safety floor is still checked. Initialization requiring live pose runs after connection.

These are Cartesian **target-point** checks. They do not validate the complete MOVE_P trajectory, whole-tool collisions, or the swept volume of joint moves. Full collision checking requires robot/environment geometry, IK-configuration checks, and validation of the actual executed trajectory. Piper currently delegates IK and MOVE_P motion to firmware; this adapter does not provide those guarantees.

## Numeric migration

1. Previously `get_pose()` approximated TIP as `(flange_x, flange_y, flange_z-L)`. It now returns `flange + R@[0,0,L]`. For `rx=180, ry=30, yaw=0, L=95`, the same physical pose changes by `(-47.5, 0, +12.7276)` mm relative to old readback; at yaw=90 the change is `(0, -47.5, +12.7276)` mm. Recheck saved targets according to their source and orientation.
2. Previously `get_home_pose()` exposed driver storage, mixing legacy home coordinates and FLANGE according to configuration. It now consistently reports the physical home's TIP. The physical home target is preserved. Scripts needing flange coordinates should read `env.home_pose`.
3. The yaw=0 `goto_xyzr` offset direction is preserved. Nonzero yaw now rotates the XY tool offset. Check scripts for compensations added for the old behavior to avoid applying them twice. Do not manually translate hand-eye calibration matrices just because API readback changed.
4. Raw FLANGE motion without an anchor now checks the tool endpoint, so previously accepted low targets may be rejected. Edge targets that were silently clamped now fail explicitly; choose targets satisfying both TIP and FLANGE bounds.
5. Poses in `get_observation()`, WorldState, the state CLI, and new traces change from FLANGE to TIP. For `L=135.8, RPY=(180,30,0)`, the change relative to old observations is `(-67.9, 0, -117.60625)` mm, distinct from the old-versus-new `get_pose()` difference above. Old traces are not converted automatically; distinguish versions when analyzing historical data. New observations retain raw flange values in `extra["flange_pose"]`.
6. `move_direction()` now normalizes its target type instead of crashing in the driver and consistently returns a TIP target. It preserves the full live orientation rather than applying `goto_xyzr()`'s fixed tilt.
7. When `servo_to_tip()` receives both `r` and `rz`, a non-None `rz` now takes precedence, matching flange pose normalization. If `rz` is None, `r` is used; live yaw is read only when both are absent or None. Scripts supplying conflicting aliases will command a different yaw. `rz=0` remains a valid target.
8. Fast-path `track_detect` emits yaw targets using the readback's field: `rz` for Piper and `r` for interfaces exposing only `r`. The entry yaw therefore participates in angular slew limiting and completion checks, preventing live `rz` from overriding target `r` when the servo merges poses.

Without an anchor, the old driver applied `z_min_safe_mm` directly to flange Z, while SafetyRail already checked TIP Z. Both now enforce a TIP floor, closing the tool-height gap in raw flange motion. Existing configurations are not converted automatically: verify this value against the physical safety plane and check the TIP height of the configured or startup home.

For `z_min_safe_mm=50`, `L=135.8`, and `RPY=(180,30,0)`, flange Z=100 gives TIP Z≈−17.606 mm and must be rejected. The flange Z needed for that TIP floor is approximately 167.60625 mm, rounded upward to 167.607 mm at SDK resolution, subject to other bounds and reachability. RecoveryRail's automatic home uses the same checks; failure is caught and recorded as `home_ok=False`. Recovery neither bypasses safety bounds nor guarantees success. Do not automatically lower the TIP floor just to preserve the old motion range.

After connecting, inspect both home representations without commanding motion:

```python
flange_home = session.env.home_pose
tip_home = session.api.get_home_pose()
print("FLANGE home:", flange_home.as_tuple())
print("TIP home:", tip_home)
obs = session.env.get_observation()
print("TIP observation:", obs.pose)
print("FLANGE observation:", obs.extra.get("flange_pose"))
```

## Calibration trajectories and workspace bounds

The built-in GUI Piper profile defaults to `trajectory.space: joint`. Teaching records joint angles and execution sends joint commands, bypassing these Cartesian XY/Z checks. Replacing Cartesian clamping with rejection therefore does not break the default joint calibration path. The configured `z_min_safe_mm` does not validate table clearance along joint trajectories either.

For Cartesian trajectories, including imported archives, execution dispatches according to the archive's space. The driver rejects out-of-box flange targets and the exception aborts capture. If calibration has been confirmed to require a larger workspace, explicitly adjust Piper's `x_min_mm/x_max_mm/y_min_mm/y_max_mm` and the necessary height limits; SO-101's `workspace_bounds: null` is not a Piper configuration field. Legacy silent clamping did not guarantee returning to the taught position.

## Hardware acceptance record

Automated tests use fake CAN and cover independent numerical expectations, command quantization, rejection, and servo error propagation. They do not establish the physical tool axis or firmware's interpretation of orientation. Record SDK/firmware versions, tool length, and local axis direction. In a verified reachable area with sufficient clearance, incrementally verify yaw=0 and positive/negative yaw, including reachable ±90° poses. Position evidence must come from independent measurement or an accepted camera calibration, rather than command echoes agreeing with `get_pose()`. Initial hand-eye parameters in the repository and yaw=0-only touch measurements are not yaw-sweep acceptance evidence.
