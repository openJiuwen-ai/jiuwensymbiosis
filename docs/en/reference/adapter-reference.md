# Robot Adapter Reference

> Category: Reference. This page is the stable contract, parameters, and implementation locations for an adapter. The [Chinese source](../../zh/reference/adapter-reference.md) is authoritative.

For your first adapter read [Build Your First Robot Adapter](../tutorial/02-build-first-adapter.md); for bringing a vendor SDK and real hardware in from a Mock, read [Port a Robot Hardware Adapter](../how-to/port-hardware-adapter.md). This page is not a sequential procedure.

## 1. Adapter files

```text
jiuwensymbiosis/adapters/<name>/
├── __init__.py
├── config.py
├── lowlevel.py
├── env.py
├── api.py
├── session.py
└── config_template.yaml

# Optional hand-eye calibration wrapper (copy template calibration.py here)
jiuwensymbiosis/calibration/adapters/<name>.py
```

| File | Stable responsibility | Must not contain |
|---|---|---|
| `__init__.py` | Export the renamed Session builder as the adapter package entry point | Hardware or business implementations |
| `config.py` | Config dataclass, `from_dict()`, `from_yaml()` | Hardware connection and task text |
| `lowlevel.py` | Vendor SDK, CAN, serial, socket, camera and actuator I/O | Agent, Rail, `@implements` |
| `env.py` | Capabilities, lifecycle, observation, safety properties, driver wrapping | Prompts and vendor workflow orchestration |
| `api.py` | `@implements(SPEC)` bindings, body geometry, camera/calibration/detector hooks | Duplicated detection/correction pipelines |
| `session.py` | Config/Env/Api, HTTP clients, optional local processes and extra-object wiring | Large business implementations |
| `config_template.yaml` | Deployable starting point with annotated fields | User tasks and secrets |
| `calibration.py` (optional) | Hand-eye wrapper exposing `CALIBRATION_ADAPTER_SPEC` | Calibration solving/quality logic (owned by the calibration subsystem) |

Template lives in `templates/xxx_adapter/`.

## 2. Action vocabulary, capabilities, and tools

The `ActionSpec` contract class is defined in `api/decorators.py`, while shared vocabulary instances live in `jiuwensymbiosis/api/actions.py`; the capability vocabulary is `jiuwensymbiosis.env.base.KNOWN_CAPABILITIES`. Each `ActionSpec` declares: name, description, capability gate, parameter names, result shape, pre-conditions and effects (including `opens_access`/`closes_access`), location freshness, and whether it is visible to the planner.

| Capability | Actions (`ActionSpec`) |
|---|---|
| `motion.cartesian` | `goto_xyzr`, `goto_pose`, `move_direction`, `get_pose`, `get_home_pose` |
| `motion.joint` | `move_joint`, `move_named_joint`, `get_joint_positions` |
| `motion.servo` | (no standalone public action; fast-path `track_detect`/`track_grasp` ops drive `api.servo_to_tip`/`env.servo_to_flange`) |
| `motion.base` | `navigate_relative`, `rotate_base`, `drive_arc` |
| `motion.base_servo` | (continuous base-drive primitive) |
| `motion.lift` | `set_lift_pose`, `lift_to_clearance` |
| `motion.waist` | `turn_waist` |
| `motion.goal` | `approach_for_grasp`, `approach_for_place` |
| `motion.dual_arm` | `dual_arm_grasp`, `dual_arm_place` |
| `grasp.parallel` | `open_gripper`, `close_gripper` |
| `grasp.suction` | `activate_suction`, `deactivate_suction` |
| `grasp.paddle` | (implemented via `dual_arm_grasp`) |
| `vision.camera` | `get_image`, `pixel_to_base_xyz` |
| `vision.detection` | `get_grasp_info_simple`, `locate_for_grasp`, `locate_for_place`, `analyze_scene` |
| `vision.search` | `search_target` |

The Env manually declares hardware capabilities; the Api **derives** its own from the specs of the actions it implements (plus declared marker-capability class attributes); the effective tool set is their intersection. An Env may declare marker capabilities with no corresponding action; an Api capability absent from the Env never becomes a tool.

`adapters/_common/capability_spec.py` provides capability→action and capability→driver-member maps (`CAPABILITY_ACTIONS`, `CAPABILITY_DRIVER_MEMBERS`), shared by the validator and generator, mirroring `api/defaults.py` and `env/protocol.py`.

## 3. Driver Protocol contract

Protocols are defined in `jiuwensymbiosis/env/protocol.py`, **sliced by capability** — one protocol per capability. A mobile dual-arm body has no flange pose; a bench arm has no wheels; a single god-protocol would force one of them into `NotImplementedError` stubs. `RobotDriver` therefore holds only what every driver has (`close()`), and each capability adds its own sibling protocol. Drivers use structural typing (`typing.Protocol`), not a base class.

| Protocol | Capability | Required members |
|---|---|---|
| `RobotDriver` | Every driver | `close()` (idempotent) |
| `CartesianDriver` | `motion.cartesian` | `home_pose`, `z_min_safe`, `flange_z_min_safe`, `tool_offset_mm`, `home()`, `get_pose()`, `move_to_pose_blocking(pose, ...)` |
| `JointDriver` | `motion.joint` (indexed, whole vector) | `get_angles()`, `move_joint_blocking(q, timeout_s=...)` |
| `NamedJointDriver` | `motion.joint` (named, partial command) | `get_joint_positions()`, `move_joints_blocking(targets, timeout_s=...)` |
| `ServoDriver` | `motion.servo` | `servo_to_pose(pose)` (non-blocking) |
| `BaseDriver` | `motion.base`/`motion.goal` | `navigate_relative(dx_m, dy_m=0, dyaw_rad=0)`, `navigate_arc(radius_m, dyaw_rad)` |
| `ContinuousBaseDriver` | `motion.base_servo` | `start_base_drive()`, `base_drive_running(handle)`, `steer_base_drive(handle, bearing_rad)`, `hold_base_drive(handle)`, `stop_base_drive(handle)` |
| `LifterDriver` | `motion.lift` | `set_lifter(q_lifter)` |
| `WaistDriver` | `motion.waist` | `turn_waist(delta_rad)` |
| `DualArmDriver` | `motion.dual_arm` | `home()` (both arms home; the coordination itself is shared, taking a body hook) |
| `CameraDriver` | `vision.camera` | `intrinsics`, `grab_frames()` |
| `SuctionDriver` | `grasp.suction` | `suction_state`, `suction_di_last`, `set_suction(on)` |
| `GripperDriver` | `grasp.parallel` | `set_gripper(on)`, `gripper_state` |
| `VisionDriver` | eye-in-hand `vision.detection` | `tf_flange_cam`, `calibration` |
| `HandGuidingDriver` (optional) | Manual pose teaching | `hand_guiding(*, include_end_effector=False)` context manager; synchronize targets before restoring torque, raising `HandGuidingRecoveryError` on recovery failure |

Key semantics:

- `get_pose()`/`home_pose` return the vendor Pose — `(x,y,z,r)` for SCARA or `(x,y,z,rx,ry,rz)` for six-axis;
- `move_to_pose_blocking()` takes a FLANGE target and must block until completion or failure; TIP↔FLANGE conversion belongs to the Api layer (`goto_xyzr`);
- `move_joint_blocking()` units are the adapter's convention but must match `joint_limits`;
- `servo_to_pose()` is non-blocking; explicit `False` means the controller did not advance this tick, `True` or legacy `None` means accepted;
- base verbs use **metres** (detections are millimetres) and return `{ok, reason, ...}`;
- `grab_frames()` returns aligned `(rgb_uint8, depth_m_float32)` or `None`;
- `close()` must be idempotent.

The named `VisionDriver` Protocol currently covers only eye-in-hand calibration. Eye-to-hand adapters such as SO-101 expose `tf_base_cam` and `calibration` through body-specific structured interfaces and declare the `vision.eye_to_hand` marker; there is no standalone named eye-to-hand Protocol yet.

A multi-capability driver may define a composite Protocol to tighten `low_level`'s type; `PiperFullDriver` is `CartesianDriver + JointDriver + CameraDriver + GripperDriver + VisionDriver`.

## 4. BaseRobotEnv contract

Abstract methods on `BaseRobotEnv`:

| Method | Semantics |
|---|---|
| `connect()` | Open the hardware connection; must be idempotent |
| `disconnect()` | Release hardware; safe to call in any state |
| `get_observation()` | Best-effort snapshot; transient sensor gaps must not fail the whole call |
| `home()` | Return the body to its safe posture; `home` is the one unconditional action, so a Cartesian default is not allowed — it is abstract |

`RobotObservation` fields:

| Field | Type | Convention |
|---|---|---|
| `pose` | `dict | None` | SCARA commonly `x,y,z,r`; six-axis `x,y,z,rx,ry,rz` |
| `joints` | `list[float] | None` | Units per robot |
| `rgb` | `np.ndarray | None` | H×W×3 `uint8` |
| `depth` | `np.ndarray | None` | H×W `float32` metres, aligned to RGB |
| `extra` | `dict` | Lightweight gripper/force/status flags |

Env properties:

| Property | Default | Consumer |
|---|---|---|
| `low_level` | `None` | Env default verbs and controlled vision access |
| `z_min_safe` | `None` | SafetyRail TIP Z floor |
| `workspace_bounds` | `None` | SafetyRail, `(xmin,ymin,xmax,ymax)` |
| `joint_limits` | `None` | SafetyRail, key order matches joint index |
| `home_pose` | `None` | Api `get_home_pose()` |
| `tool_offset_mm` | `0.0` | TIP↔FLANGE geometry |
| `joint_units` | `None` | `move_joint`/observed joint unit (`"deg"`/`"rad"`; unstated means unknown) |
| `default_orientation_policy` | `None` | Default tilt `goto_xyzr` applies when omitted |
| `base_step_limits` | `None` | SafetyRail, `(max|translation|m, max|turn|rad)` per base command |
| `lift_limits` | `None` | SafetyRail, `set_lifter` soft limits |
| `waist_step_limit_rad` | `None` | SafetyRail, `turn_waist` per-command cap |
| `holding_payload` | Not provided | RecoveryRail checks before homing so a carried payload is not dropped blindly |
| `cameras` | `(None,)` | Perceivable cameras (best-first); `grab_calibrated_frame(camera)` |
| `urdf_path`/`arm_chains`/`arm_joints` | `None` | Derive `planning.reachability`; joints each arm actuates |

Default delegations:

| Env verb | Driver target |
|---|---|
| `get_flange_pose()` | `driver.get_pose()` |
| `move_to_flange(pose)` | `driver.move_to_pose_blocking(pose)` |
| `move_joint(targets)` | Named: straight to `move_joints_blocking`; indexed: read current config, rewrite named entries, then `move_joint_blocking(q)` |
| `set_end_effector(True/False)` | `set_gripper()` or `set_suction()` by capability |
| `grab_rgb()` | default `get_observation().rgb` |

`reset()` and software `emergency_stop()` default to no-ops. Physical E-stop always belongs to the hardware safety system.

## 5. Api, `@implements`, and override rules

The Api subclasses `BaseRobotApi` and binds each action with `@implements(SPEC)`. Built-in default behavior lives in `api/defaults`, covering common motion, joints, grasp, image, and visual procedures.

| Situation | Handling |
|---|---|
| Behaviour matches common semantics | `@implements(SPEC)` then `return defaults.<action>(self, ...)` one-line forward |
| TIP/FLANGE, pose, or field-name difference | Override the method body |
| Body-specific semantics needed | Still `@implements(SPEC)`; the contract comes from the spec, the implementation only says "how" |
| A capability outside the vocabulary | Complete the capability-extension contract first, then add the action (`ActionSpec` in `api/actions.py` + `@implements` on the Api) |

`@implements` attaches a `ToolMeta` (spec + `input_params`, the call schema derived from this body's signature); a signature that cannot accept a parameter the spec promises raises `ContractViolation` at import time. Bring-up, calibration and debug views are **not** actions: leave them undecorated and drive them from `scripts/`.

A visual adapter explicitly binds `pixel_to_base_xyz(u, v, depth_m)` with `@implements(PIXEL_TO_BASE_XYZ)`: eye-in-hand bodies may forward to `perception/vision.default_pixel_to_base_xyz` with a `pose_to_tf` callback (combining live `T_base_flange @ T_flange_cam` and applying calibration XY correction), while eye-to-hand bodies implement it with `T_base_cam`. `locate_for_grasp`/`locate_for_place`/`analyze_scene` are shared by `perception/scene3d` and consume a calibrated `CameraFrame` plus detector hook directly; `search_target`/`approach_for_grasp`/`approach_for_place` are shared by `motion/approach`. An adapter supplies camera/calibration/detector hooks and overrides only body-specific geometry.

## 6. Config and Session builder

Declare source-relative mapping keys in Config `path_fields: ClassVar[tuple[str, ...]]` (omit for pathless configs). `from_yaml` uses `adapters._common.config.load_yaml_config`, sharing `parse_config` with Runtime and smoke. Flat and legacy nested layouts use the same rules; missing targets never revert to the working directory.

A Config provides at least `from_dict(data)` and `from_yaml(path)`. Common fields by capability:

| Group | Common fields |
|---|---|
| Basic & connection | `name`, CAN/serial/network address, speed, timeout |
| Kinematics | Home pose, `tool_offset_mm`, orientation convention |
| Safety | `z_min_safe_mm`, XY bounds, `joint_limits` |
| End effector | travel, force, suction I/O |
| Camera | serial, resolution, FPS, intrinsics/calibration path |
| Detection | `detector.mode`; remote `endpoint`; local model, device and thresholds |
| Grasp/place geometry | `z_correction_mm`, `grasp_z_offset_mm`, `place_z_offset_mm` |

`make_builder()` signature:

```python
def make_builder(
    cfg_cls,
    env_cls,
    api_cls,
    *,
    config_factory=None,
    resource_keys=None,
    api_kwargs_from_cfg=None,
    sidecar_builders=None,
    managed_detector=False,
    decorate=None,
): ...
```

| Parameter | Role |
|---|---|
| `cfg_cls` | Config class with `from_yaml`/`from_dict` |
| `config_factory` | Optional loader replacing cfg_cls loading methods; exposed as builder `.config_factory` |
| `resource_keys` | Effective cfg → device resource keys; exposed as `.resource_keys(cfg)` for Runtime/official admission |
| `env_cls` | Env class built from `cfg` |
| `api_cls` | Api class built from `env` and optional kwargs |
| `api_kwargs_from_cfg` | `list[str]` declarative mapping or compatible `cfg -> dict` callback |
| `sidecar_builders` | Each receives cfg, returns a context manager, zero-arg factory, or `None` |
| `managed_detector` | Defaults to `False`; set `True` for a visual adapter so Session creates, injects and closes `detector_client` |
| `decorate` | Final Session decoration; normally unused |

Declarative field mapping:

| Form | Result |
|---|---|
| `"z_correction_mm"` | `cfg.z_correction_mm` to the same-named Api param |
| `"camera_calib_path:calibration_path"` | Rename a configuration field; the target must match an Api constructor parameter |

The returned Builder supports `build(cfg)`, `.from_yaml(path)`, and `.from_dict(data)`.
Config uses `DetectorConfig` for its `detector` field and passes the full runtime
mapping to `parse_detector_config(data)`; the default mode is `disabled`.
With `managed_detector=True`, the Api constructor must accept the `detector_client`
keyword argument and reuse that callable. Passing only a URL is a legacy compatibility path.

GroundingDINO/SAM2 provides an HTTP inference service.
`make_detector_sidecar(cfg_attr="detector")` starts an owned local subprocess (sidecar)
only for `detector.mode: local`, and Session stops it on exit. `remote` connects through
`detector.endpoint` to an externally managed HTTP service, including one on `localhost`.
Session closes only its own client, never that external service. `disabled` sends no
inference requests. All three construction forms accept `include_sidecars=False`:
maintenance skips local process startup and inference credential resolution, retains
client cleanup, and opens the client lazily only for an explicit inference request.
See the [remote inference guide](../how-to/remote-inference.md) for configuration.

Resource keys and actual connections must use the same effective configuration. Cruzr accepts an explicit
`ros_domain_id` in YAML; otherwise Config captures `ROS_DOMAIN_ID` when constructed (default 0). ROS
initialization and command workers use that value, and a pre-existing ROS context in another domain is rejected.
Owned local subprocesses must propagate shutdown failures. If startup fails after successful rollback, they may expose
`cleanup_report() -> CleanupReport` to confirm release; otherwise Session retains an unknown cleanup state.

## 7. Shared perception and geometry modules

| Module | Main interface | Purpose |
|---|---|---|
| `adapters/_common/builder.py` | `make_builder()`, `make_detector_sidecar()` | Session client ownership and optional local subprocess wiring |
| `adapters/_common/capability_spec.py` | `CAPABILITY_ACTIONS`, `CAPABILITY_DRIVER_MEMBERS` | capability→action/driver-member maps (validator & generator) |
| `adapters/_common/safety.py` | `WorkspaceBounds`, `check_flange_z()` | TIP/FLANGE Z defence |
| `perception/config.py` | `DetectorConfig`, `parse_detector_config()` | remote / local / disabled configuration and legacy normalization |
| `perception/detector_client.py` | `create_detector_client()`, `segment_image()`; legacy `init_detector()` | Closable HTTP inference client |
| `perception/detector_sidecar.py` | `detector_subprocess()` | Lifecycle of an owned local detector subprocess only |
| `perception/scene3d.py` | `locate_for_grasp()`, `locate_for_place()`, `analyze_scene()` | calibrated frame→detection→masked point cloud→object/surface geometry |
| `perception/vision.py` | `detect_and_centroid()`, `apply_xy_correction()`, `default_pixel_to_base_xyz()`, `default_get_grasp_info_simple()` | shared detection/correction/eye-in-hand projection functions |
| `perception/calibration.py` | `load_calibration()` | versioned hand-eye calibration loading |
| `motion/approach.py` | `search_target()`, `approach_target_for_grasp()`, `approach_target_for_place()` | search → face the target → converge to a work pose (base approach) |
| `motion/dual_arm.py` | `dual_arm_grasp()`, `dual_arm_place()` | two-arm coordinated grasp/place (with force confirmation) |
| `contracts.py` | `GraspResult`, `ObjectGeometryResult`, `SPATIAL_RELATIONS`, … | action result types + spatial-relation set (owned by no layer) |

### `create_detector_client`

```python
from jiuwensymbiosis.perception.config import parse_detector_config
from jiuwensymbiosis.perception.detector_client import create_detector_client

detector_cfg = parse_detector_config({
    "detector": {
        "mode": "remote",
        "endpoint": {"url": "http://127.0.0.1:8114"},
    },
})
# Standalone scripts own their client; adapters use the Session-injected client.
with create_detector_client(detector_cfg) as seg_fn:
    results = seg_fn(image_ndarray, text_prompt="blue box")
```

Result items carry boolean `mask`, `box=[x1,y1,x2,y2]`, score, and label. An empty list
means inference succeeded without detections. Unavailable services, timeouts,
invalid responses and expired results raise `InferenceServiceError`
with the corresponding `inference_*` code. Failures never enable local models automatically.

`init_detector(url)` preserves the original Python call form and uses the current `/v1` interface.
It also returns a client that must be closed and raises explicit service failures.
New adapters use Session injection with `managed_detector=True`; new scripts use
the context manager above.

### `detect_and_centroid`

```python
with create_detector_client(detector_cfg) as seg_fn:
    result = detect_and_centroid(
        rgb=rgb_ndarray,
        depth_img_m=depth_ndarray,
        seg_fn=seg_fn,
        object_name="red block",
        tcp_at_grab=pose_at_grab,
        captured_monotonic_s=capture_started,  # time.monotonic() recorded before capture
    )
```

Successful results include `u`, `v`, `depth_m`, best, mask shape, and image shape.
Observation failures include `no_detection`, `empty_mask` and `no_valid_depth`.
HTTP service failures return `ok=False`, `reason="detector_unavailable"` and retain
the `inference_*` value in `error_code`; cancellation propagates to the caller.

### `apply_xy_correction`

```python
xyz_final, description = apply_xy_correction(
    xyz_raw=raw_base_xyz,
    xy_transform=calibration.get("xy_transform"),
    xy_correction_mm=calibration.get("xy_correction_mm"),
)
```

`xy_transform` takes priority over a legacy translation. Regular adapters never call it directly; the shared flow applies it once.

## 8. Capability extension contract

Extend the vocabulary only when no existing string describes the hardware's real capability. One extension must update together:

1. Add the string to `env/base.py:KNOWN_CAPABILITIES`;
2. If it needs an action, add an `ActionSpec` to `api/actions.py` and an `@implements` on the Api (forward to `api/defaults` when the delegation is generic);
3. Add an Env common verb when the delegation is uniform, else implement in the adapter;
4. Env declares the capability, Api implements the action;
5. Update `capability_spec.py`, the validator, and the tool-generation tests;
6. Add the corresponding Rail check when the action carries motion risk, or state explicitly that the controller is responsible.

Do not add strings casually to dodge an unknown-capability error. A marker capability may have no action, but it should have a clear consumer.

## 9. Validator and test entry points

| Entry | Purpose |
|---|---|
| `scripts/validate_adapter.py --module ...` | Static check of directory, signatures, capability alignment, and Driver members |
| `scripts/smoke_test_adapter.py --module ...` | Inject a stub driver into the Env (no real hardware connection), invoke every generated tool, check serializable results |
| `tests/unit_tests/env/` | Env, capability, and safety-property reference tests |
| `tests/unit_tests/api/` | action vocabulary, `@implements`, and capability derivation |
| `tests/unit_tests/agent/` | Session, Builder, Rails, and tool assembly |
| `tests/mocks/` | Mock Driver, Env, Api, and scenes |

Common validation outcomes (the validator prints numbered Chinese messages; these are semantic summaries):

| Symptom | Meaning |
|---|---|
| Unknown capability (for example E-04) | Env string not in the vocabulary |
| Api capability missing from `Env.capabilities` (A-08) | Tool filtered by the capability intersection |
| Driver missing members required by a capability (for example D-14) | Declared capability disagrees with a Driver Protocol |

## 10. Built-in adapter implementation locations

| Concern | Piper | SO-101 | Cruzr |
|---|---|---|---|
| Capabilities & Env | `adapters/piper/env.py` | `adapters/so101/env.py` | `adapters/cruzr/env.py` |
| Vendor driver | `adapters/piper/lowlevel.py` | `adapters/so101/lowlevel.py` | `adapters/cruzr/lowlevel.py` |
| Geometry | `adapters/piper/geometry.py` | `adapters/so101/geometry.py` | `adapters/cruzr/geometry.py` |
| Calibration | `adapters/piper/_calibration.py` + `calibration/adapters/piper.py` | `adapters/so101/_calibration.py` + `calibration/adapters/so101.py` | `adapters/cruzr/_calibration.py` (camera calibration loading only) |
| Api | `adapters/piper/api.py` | `adapters/so101/api.py` | `adapters/cruzr/api.py` |
| Config | `adapters/piper/config.py` | `adapters/so101/config.py` | `adapters/cruzr/config.py` |
| Session | `adapters/piper/session.py` | `adapters/so101/session.py` | `adapters/cruzr/session.py` |

Piper's 30° tilted tool, historical nested YAML, and temporary Z correction are body-specific; a new adapter should reuse them only when the hardware genuinely has the same constraint. Cruzr's dual-arm + lift reachability (`Reachability` override), head/waist dual cameras, and ROS 2 driver are also body-specific — the generic mechanisms (`scene3d`/`approach`/`dual_arm` shared implementations) must not be copied elsewhere.
