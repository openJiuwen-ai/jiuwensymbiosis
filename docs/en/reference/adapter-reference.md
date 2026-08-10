# Robot Adapter Reference

> Category: Reference. This page is the lookup source for stable adapter contracts, parameters, and implementation locations.

For a first implementation, use [Build Your First Robot Adapter](../tutorial/02-build-first-adapter.md). To replace a
mock with a vendor SDK and real hardware, use [Port a Robot Hardware Adapter](../how-to/port-hardware-adapter.md). This
page is not a sequential procedure.

## 1. The six adapter files

```text
jiuwensymbiosis/adapters/<name>/
├── config.py
├── lowlevel.py
├── env.py
├── api.py
├── session.py
└── config_template.yaml
```

| File | Stable responsibility | Must not contain |
|---|---|---|
| `config.py` | Configuration dataclass, `from_dict()`, and `from_yaml()` | Hardware connections or task text |
| `lowlevel.py` | Vendor SDK, CAN, serial, socket, camera, and actuator I/O | Agent, Rail, or `@robot_tool` logic |
| `env.py` | Capabilities, lifecycle, observations, safety properties, Driver wrapper | Prompts or vendor workflow orchestration |
| `api.py` | Mixin composition, tool semantics, body geometry, raw visual projection | Duplicate detection/correction pipelines |
| `session.py` | Config/Env/Api, sidecars, and extra-object wiring | Large business implementations |
| `config_template.yaml` | Annotated deployment starting point | User tasks or secret credentials |

The skeleton is in `templates/xxx_adapter/`.

## 2. Capability, Mixin, and tool mapping

`jiuwensymbiosis.env.base.KNOWN_CAPABILITIES` defines the vocabulary:

| Capability | Mixin or role | Main inherited tools or behavior |
|---|---|---|
| `motion.cartesian` | `MotionMixin` | `home()`, `get_pose()`, `get_home_pose()`, `goto_xyzr()` |
| `motion.joint` | `JointMotionMixin` | `move_joint()` |
| `motion.servo` | Adapter-specific real-time tool | Non-blocking streaming pose command |
| `grasp.suction` | `SuctionMixin` | `activate_suction()`, `deactivate_suction()` |
| `grasp.parallel` | `ParallelGripperMixin` | `open_gripper()`, `close_gripper()` |
| `vision.detection` | `VisionMixin` | `get_image()`, `get_grasp_info_simple()`, `pixel_to_base_xyz()` |
| `vision.camera` | Marker capability | RGB is available; no tool by itself |
| `vision.depth` | Marker capability | Depth is available; no tool by itself |
| `vision.eye_to_hand` | Marker capability | Camera is fixed in the base/world frame |
| `sorting.command` | Adapter or dedicated tool | Opaque sorting protocol |
| `speech.tts` | Adapter or service | Text-to-speech |

Env declares physical capabilities manually. Api derives software capabilities from its Mixin MRO. Effective tool
capabilities are the intersection. Marker capabilities may exist without a Mixin; an Api capability missing from Env
does not generate tools.

## 3. Driver Protocol contract

Protocols live in `jiuwensymbiosis/env/protocol.py`. Concrete Drivers use structural typing and do not inherit them.

| Protocol | Capability | Required members |
|---|---|---|
| `RobotDriver` | `motion.cartesian` | `home_pose`, `z_min_safe`, `flange_z_min_safe`, `tool_offset_mm`, `close()`, `home()`, `get_pose()`, `move_to_pose_blocking(pose, ...)` |
| `JointDriver` | `motion.joint` | `get_angles()`, `move_joint_blocking(q, timeout_s=...)` |
| `ServoDriver` | `motion.servo` | `servo_to_pose(pose)` |
| `GripperDriver` | `grasp.parallel` | `set_gripper(on)`, `gripper_state` |
| `SuctionDriver` | `grasp.suction` | `set_suction(on)`, `suction_state`, `suction_di_last` |
| `CameraDriver` | `vision.camera` | `intrinsics`, `grab_frames()` |
| `VisionDriver` | eye-in-hand `vision.detection` | `tf_flange_cam`, `calibration` |

Key semantics:

- `get_pose()` and `home_pose` return a vendor Pose, either SCARA `(x,y,z,r)` or six-axis `(x,y,z,rx,ry,rz)`.
- `move_to_pose_blocking()` accepts a FLANGE target and blocks until completion or failure.
- `move_joint_blocking()` units are adapter-defined but match `joint_limits`.
- `servo_to_pose()` is non-blocking; explicit `False` means no progress this cycle, while `True` or legacy `None` means accepted.
- `grab_frames()` returns aligned `(rgb_uint8, depth_m_float32)` or `None`.
- `close()` is idempotent.

The current named `VisionDriver` Protocol covers eye-in-hand calibration. Eye-to-hand adapters such as SO-101 expose
`tf_base_cam` and `calibration` as an adapter-specific structural surface and advertise the `vision.eye_to_hand` marker
capability; there is not yet a separate named eye-to-hand Protocol.

Multi-capability Drivers may define a composite Protocol to narrow `low_level`. `PiperFullDriver` combines
`RobotDriver + JointDriver + CameraDriver + GripperDriver + VisionDriver`.

## 4. BaseRobotEnv contract

Abstract methods:

| Method | Semantics |
|---|---|
| `connect()` | Open hardware; idempotent |
| `disconnect()` | Release hardware; safe in every state |
| `get_observation()` | Best-effort snapshot; transient sensor gaps do not fail the whole call |

`RobotObservation` fields:

| Field | Type | Convention |
|---|---|---|
| `pose` | `dict | None` | SCARA commonly uses `x,y,z,r`; six-axis uses `x,y,z,rx,ry,rz` |
| `joints` | `list[float] | None` | Units follow the robot convention |
| `rgb` | `np.ndarray | None` | H×W×3 `uint8` |
| `depth` | `np.ndarray | None` | H×W `float32` meters, aligned to RGB |
| `extra` | `dict` | Lightweight gripper, force, or status fields |

Env properties:

| Property | Default | Consumer |
|---|---|---|
| `low_level` | `None` | Env delegation and controlled visual access |
| `z_min_safe` | `None` | SafetyRail TIP Z floor |
| `workspace_bounds` | `None` | SafetyRail, ordered `(xmin,ymin,xmax,ymax)` |
| `joint_limits` | `None` | SafetyRail, key order matches joint index |
| `home_pose` | `None` | `MotionMixin.get_home_pose()` |
| `tool_offset_mm` | `0.0` | TIP-to-FLANGE geometry |

Inherited delegation:

| Env verb | Driver target |
|---|---|
| `home()` | `driver.home()` |
| `get_flange_pose()` | `driver.get_pose()` |
| `move_to_flange(pose)` | `driver.move_to_pose_blocking(pose)` |
| `move_joint(q)` | `driver.move_joint_blocking(q)` |
| `set_end_effector(True/False)` | Capability-based `set_gripper()` or `set_suction()` |
| `grab_rgb()` | `get_observation().rgb` by default |

`reset()` and software `emergency_stop()` default to no-op. Physical E-stop remains a hardware safety function.

## 5. Api, Mixin, and override rules

Capability Mixins precede `BaseRobotApi` in the inheritance list. Built-in defaults cover ordinary motion, joints,
grasp, image capture, and the visual grasp/place pipeline.

| Situation | Action |
|---|---|
| Behavior matches public semantics | Inherit the Mixin implementation |
| TIP/FLANGE, orientation, or field names differ | Override the method body |
| Keep the inherited tool description | Do not redecorate the override |
| Supply a body-specific description | Redecorate and explicitly supply `tags` |
| Hardware needs a new vocabulary capability | Complete the capability extension contract first |

`@robot_tool` metadata includes name, description, input JSON Schema, capability, and tags. Tool construction walks the
MRO, so an override may inherit decorator metadata.

`VisionMixin` owns this sequence:

```text
frames → detection → mask centroid/median depth → raw projection → XY/Z correction → grasp/place height → result
```

A visual adapter implements `_project_pixel_to_base_raw(u, v, depth_m)`: eye-in-hand composes live
`T_base_flange @ T_flange_cam`; eye-to-hand uses fixed `T_base_cam`. The raw method does not apply correction.
`analyze_scene()` has no universal semantics and is implemented only when needed.

## 6. Config and Session Builder

Config supplies `from_dict(data)` and `from_yaml(path)`. Choose fields by capability:

| Group | Common fields |
|---|---|
| Identity and connection | `name`, CAN/serial/network address, speed, timeout |
| Kinematics | Home pose, `tool_offset_mm`, orientation convention |
| Safety | `z_min_safe_mm`, XY bounds, `joint_limits` |
| End effector | Opening, effort, suction I/O |
| Camera | Serial, resolution, FPS, intrinsics/calibration path |
| Detection | URL, spawn, models, thresholds |
| Grasp/place geometry | `z_correction_mm`, `grasp_z_offset_mm`, `place_z_offset_mm` |

`make_builder()` signature:

```python
make_builder(
    cfg_cls,
    env_cls,
    api_cls,
    *,
    api_kwargs_from_cfg=None,
    sidecar_builders=None,
    decorate=None,
)
```

| Parameter | Purpose |
|---|---|
| `cfg_cls` | Config class with `from_yaml` and `from_dict` |
| `env_cls` | Env constructed with `cfg` |
| `api_cls` | Api constructed with `env` and optional kwargs |
| `api_kwargs_from_cfg` | Declarative `list[str]` mapping or compatible `cfg -> dict` callback |
| `sidecar_builders` | Each receives cfg and returns a context manager, zero-argument factory, or `None` |
| `decorate` | Final Session callback; normally unused |

Declarative field mapping:

| Form | Result |
|---|---|
| `"z_correction_mm"` | Pass `cfg.z_correction_mm` under the same Api name |
| `"detector.url:detector_service_url"` | Read a nested field and rename it |

The returned Builder supports `build(cfg)`, `.from_yaml(path)`, and `.from_dict(data)`.
`make_detector_sidecar(cfg_attr="detector")` reads detector configuration and starts GroundingDINO/SAM2 with the
Session when `spawn` is true.

## 7. Shared perception and geometry modules

| Module | Main interface | Purpose |
|---|---|---|
| `adapters/_common/builder.py` | `make_builder()`, `make_detector_sidecar()` | Session factory and detector sidecar |
| `adapters/_common/safety.py` | `WorkspaceBounds`, `check_flange_z()` | TIP/FLANGE Z guard |
| `adapters/_common/capability_spec.py` | capability specifications | Static adapter validation |
| `perception/detector_client.py` | `init_detector()` | HTTP detector client |
| `perception/detector_sidecar.py` | `detector_subprocess()` | Detector service lifecycle |
| `perception/vision.py` | `detect_and_centroid()` | Detection, centroid, and median depth |
| `perception/vision.py` | `build_grasp_result()` | Correction and grasp/place geometry |
| `perception/vision.py` | `apply_xy_correction()` | Multi-point or legacy translation correction |
| `perception/vision.py` | `dump_grasp_debug()` | Optional debug artifacts |
| `perception/calibration.py` | `load_calibration()` | Versioned hand-eye calibration loading |
| `utils/geometry.py` | pinhole and SE(3) helpers | Projection and transform composition |

### `init_detector`

```python
seg_fn = init_detector("http://127.0.0.1:8114")
results = seg_fn(image_ndarray, text_prompt="blue box")
```

Each result contains a boolean `mask`, `[x1,y1,x2,y2]`, score, and label. An unreachable service returns an empty result.

### `detect_and_centroid`

```python
result = detect_and_centroid(
    rgb=rgb_ndarray,
    depth_img_m=depth_ndarray,
    seg_fn=seg_fn,
    object_name="red block",
    tcp_at_grab=pose_at_grab,
)
```

Success includes `u`, `v`, `depth_m`, best, mask shape, and image shape. Failure reasons include `no_detection`,
`empty_mask`, and `no_valid_depth`.

### `apply_xy_correction`

```python
xyz_final, description = apply_xy_correction(
    xyz_raw=raw_base_xyz,
    xy_transform=calibration.get("xy_transform"),
    xy_correction_mm=calibration.get("xy_correction_mm"),
)
```

`xy_transform` takes priority over legacy translation. Normal adapters do not call this directly; `VisionMixin` owns it
inside the shared projection path.

## 8. Capability extension contract

Extend the vocabulary only when no current string describes real hardware. One extension updates all of:

1. Add the string to `env/base.py:KNOWN_CAPABILITIES`.
2. If it generates tools, add a Mixin with the exact capability in `api/mixins.py`.
3. Add a public Env verb when delegation is universal; otherwise implement it in the adapter.
4. Declare the Env capability and inherit the Api Mixin.
5. Update `capability_spec.py`, validator coverage, and tool-generation tests.
6. Add a Rail check for safety-relevant actions or document controller ownership.

Do not add arbitrary strings to bypass an unknown-capability error. A marker capability may have no Mixin, but it must
have a defined consumer.

## 9. Validator and test entry points

| Entry point | Purpose |
|---|---|
| `scripts/validate_adapter.py --module ...` | Static directory, signature, capability, and Driver-member checks |
| `scripts/smoke_test_adapter.py --module ...` | Connect Mock Env, call generated tools, verify serializable results |
| `tests/unit_tests/env/` | Env, capability, and safety-property examples |
| `tests/unit_tests/api/` | Mixin, decorator, and capability derivation |
| `tests/unit_tests/agent/` | Session, Builder, Rails, and tool assembly |
| `tests/mocks/` | Mock Driver, Env, Api, and scene patterns |

Common validation results:

| Symptom | Meaning |
|---|---|
| unknown capability | Env value is outside the vocabulary |
| Api capability missing from Env | Capability intersection filters the tool |
| Driver member missing | Declared capability and Driver Protocol disagree |
| tool result not serializable | Tool returned ndarray, Pose, or another native object |

## 10. Piper implementation map

| Concern | Piper file | Notes |
|---|---|---|
| Capability and Env | `adapters/piper/env.py` | Capabilities, lifecycle, observations, boundaries |
| Vendor Driver | `adapters/piper/lowlevel.py` | CAN/SDK, motion, gripper, camera |
| Geometry | `adapters/piper/geometry.py` | Pose, TIP/FLANGE, projection helpers |
| Calibration | `adapters/piper/_calibration.py` | Hand-eye transform loading |
| Api | `adapters/piper/api.py` | Tilted-tool geometry and raw projection |
| Config | `adapters/piper/config.py` | PiperConfig, DetectorServerConfig, historical YAML compatibility |
| Session | `adapters/piper/session.py` | `make_builder()`, mappings, detector sidecar |

Piper's 30-degree tilted tool, historical nested YAML, and temporary Z correction are body-specific. Reuse them only
when new hardware has the same constraints.
