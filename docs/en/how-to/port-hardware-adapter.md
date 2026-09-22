# Port a Robot Hardware Adapter

> Category: How-to. This guide turns an already working mock adapter into production hardware integration.

If this is your first adapter, complete [Build Your First Robot Adapter](../tutorial/02-build-first-adapter.md) first.
This guide does not repeat the complete "6 required Python files + 1 YAML, plus an optional calibration wrapper" example. Use the
[Robot Adapter Reference](../reference/adapter-reference.md) for exact Capability, Protocol, Env, `@implements`, and
`make_builder()` contracts.

## Scope and completion criteria

The three adapter documents have distinct jobs:

| Goal | Start here |
|---|---|
| Build a runnable first adapter without hardware | [Build Your First Robot Adapter](../tutorial/02-build-first-adapter.md) |
| Integrate a vendor SDK, camera, and end effector | This guide |
| Look up contracts, fields, parameters, or Piper locations | [Robot Adapter Reference](../reference/adapter-reference.md) |

A production adapter is complete when:

- Env and Api capabilities align and generated tools represent real hardware;
- Driver connection, motion, end-effector, and sensor calls repeat safely and fail clearly;
- TIP, FLANGE, camera, and base frames are documented and verified;
- Config, Session, and optional detector sidecars start and stop cleanly;
- software boundaries, controller limits, and the physical E-stop have been accepted;
- static validation, mock smoke tests, unit tests, and low-speed hardware acceptance pass.

Start from the Tutorial's mock adapter and replace one boundary at a time:

```text
capabilities and frames → Driver → Env/Api → vision → Session → safety and acceptance
```

## 1. Confirm hardware capabilities and coordinate conventions

Record the differences between the vendor manual, actual controller capabilities, and the mock adapter. Do not declare
a hardware capability merely because an Api `@implements` exists.

| Decision | Confirm |
|---|---|
| Motion | Cartesian, joint, or streaming servo; completion semantics; position and angle units |
| End effector | Suction or parallel gripper; open/close meaning; whether feedback is reliable |
| Camera | RGB, depth, frame alignment; eye-in-hand or eye-to-hand |
| Frames | Whether vendor poses describe TIP or FLANGE; Euler order, direction, and unit |
| Safety | TIP Z floor, XY workspace, joint soft limits, and controller hard limits |

Env declares only physical support:

```python
class MyEnv(BaseRobotEnv):
    capabilities = frozenset({
        "motion.cartesian",
        "grasp.parallel",
        "vision.camera",
        "vision.depth",
        "vision.detection",
    })
```

Api binds the actions the body supports from the shared vocabulary with `@implements(SPEC)`:

```python
class MyApi(BaseRobotApi):
    @implements(GOTO_XYZR)
    def goto_xyzr(self, x: float, y: float, z: float, r: float | None = None,
                  *, orientation_policy: str = "top_down") -> None:
        return defaults.goto_xyzr(self, x, y, z, r)

    @implements(CLOSE_GRIPPER)
    def close_gripper(self, force_n: float | None = None) -> dict:
        return defaults.close_gripper(self, force_n)
```

Tools are generated for `api.capabilities ∩ env.capabilities`. Inspect the result after assembly:

```python
with session:
    print(session.describe()["effective_capabilities"])
```

Define public semantics before implementing the Driver: XYZ in `goto_xyzr(x, y, z, r)` identifies the tool TIP in the
base frame, while Driver `move_to_pose_blocking(pose)` accepts a FLANGE target. Tool offsets and tilted mounts are
body-specific Api geometry and must not be hidden in model prompts.

## 2. Replace the Mock Driver with the vendor SDK

`lowlevel.py` translates stable framework verbs into serial, CAN, socket, or vendor SDK operations. Agent prompts,
`@implements`, and Rail logic do not belong in the Driver.

```text
LLM tool → Api @implements/override → public Env verb → Driver → controller
evidence and failures ←──────── through the same boundaries ←────────
```

Motion and end-effector access goes through public Env verbs. Vision calibration, intrinsics, and raw frames may use
the Protocol-typed `env.low_level` penetration point.

### Production Driver requirements

- Make Driver `connect()`/`close()` idempotent and reconnectable after cleanup; Env exposes `disconnect()` and delegates
  teardown to Driver `close()`.
- Make `move_to_pose_blocking()` wait for completion or a clear failure, not only command acceptance.
- Document every frame and unit; depth uses meters at the acquisition boundary.
- Raise actionable errors for timeout, unreachable target, communication loss, and malformed state.
- Serialize motion, camera, and state access when the SDK is not thread-safe.
- Keep a final workspace guard in the Driver or controller; SafetyRail does not replace hard limits.
- Begin real motion at low speed with the physical E-stop reachable.

The minimal shape of a Cartesian Driver is below. See the
[Driver Protocol contract](../reference/adapter-reference.md#3-driver-protocol-contract) for all capability-specific
members.

```python
class MyDriver:
    def connect(self) -> None: ...
    def close(self) -> None: ...
    def get_pose(self): ...
    def home(self) -> None: ...
    def move_to_pose_blocking(self, pose) -> None: ...

    # Supply only for declared capabilities
    def move_joint_blocking(self, q: list[float], *, timeout_s=30.0) -> None: ...
    def set_gripper(self, on: bool) -> None: ...
    def grab_frames(self): ...  # (rgb_uint8, depth_m_float32) or None
```

The current `templates/xxx_adapter` and `scripts/new_adapter` legacy scaffold still names the low-level teardown method
`disconnect()`. A generated adapter is internally consistent because its Env calls that method, but it does not satisfy
the formal `RobotDriver.close()` lifecycle member. Before treating generated code as a production `RobotDriver`, rename
the method or add an idempotent `close()` alias and make the Env delegate to it.

Do not replace every hardware feature at once. Validate connection and state, Home, one safe pose, end-effector open and
close, and camera frames in that order before enabling the Agent.

## 3. Complete the production Env and body-specific Api

Env is the only hardware contract used by Agent, Tools, and Rails. It owns lifecycle, observations, and safety
properties without reimplementing vendor business logic.

### Env lifecycle and observations

Keep the driver handle before connecting and track connection state separately. Clear the handle only after a successful close; propagate cleanup errors and retain failed handles for retry. If a constructor acquired hardware and cannot roll back, raise `HardwareCleanupError` from `agent.lifecycle` so Session and admission retain the uncertainty.

```python
class MyEnv(BaseRobotEnv):
    def __init__(self, cfg):
        self.cfg = cfg
        self.low_level = None
        self._connected = False

    def connect(self) -> None:
        if self._connected:
            return
        if self.low_level is not None:
            raise RuntimeError("previous cleanup is not confirmed")
        self.low_level = MyDriver(self.cfg.can_port)
        self.low_level.connect()
        self._connected = True

    def disconnect(self) -> None:
        if self.low_level is not None:
            self.low_level.close()
            self.low_level = None
        self._connected = False

    def get_observation(self) -> RobotObservation:
        if self.low_level is None:
            return RobotObservation()
        pose = self.low_level.get_pose()
        return RobotObservation(
            pose={"x": pose.x, "y": pose.y, "z": pose.z,
                  "rx": pose.rx, "ry": pose.ry, "rz": pose.rz},
            extra={"connected": True},
        )
```

Expose `z_min_safe`, `workspace_bounds`, and `joint_limits` from Config, and `home_pose` plus `tool_offset_mm` from the
Driver. See the [Env contract](../reference/adapter-reference.md#4-baserobotenv-contract) for field types and inherited
delegation.

### Api geometry and visual projection

Inherit ordinary motion, joint, grasp, and image operations. Override only public semantics that differ by body. For a
vertical tool, TIP-to-FLANGE conversion can look like:

```python
@implements(GOTO_XYZR)   # the contract — description, gate, params — comes from the spec
def goto_xyzr(self, x: float, y: float, z: float, r: float | None = None) -> None:
    current = self.env.get_flange_pose()
    flange = MyPose(x, y, z + self.env.tool_offset_mm,
                    current.rx, current.ry, current.rz if r is None else r)
    self.env.move_to_flange(flange)
```

This scalar example applies only when the tool extends along base Z. Use a full transform for a tilted mount.

A visual adapter binds the `pixel_to_base_xyz()` projection action with `@implements(PIXEL_TO_BASE_XYZ)`. Eye-to-hand example:

```python
@implements(PIXEL_TO_BASE_XYZ)
def pixel_to_base_xyz(self, u: float, v: float, depth_m: float) -> dict:
    driver = self.env.low_level
    if driver is None:
        raise RuntimeError("env not connected")
    calibration = driver.calibration or {}
    intrinsics = calibration.get("intrinsics")
    if intrinsics is None:
        intrinsics = driver.intrinsics
    tf_base_cam = driver.tf_base_cam
    if intrinsics is None or tf_base_cam is None:
        raise RuntimeError("eye-to-hand calibration unavailable")
    p_cam = pixel_and_depth_to_camera_xyz((u, v), depth_m, intrinsics)
    xyz = apply_transform(tf_base_cam, p_cam)
    return {"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])}
```

Eye-in-hand bodies may forward this action to `perception/vision.default_pixel_to_base_xyz()` with a `pose_to_tf`
callback; the shared implementation composes live `T_base_flange @ T_flange_cam` and applies calibration XY correction.
`get_image()` may forward through `api.defaults`, while `pixel_to_base_xyz()` and `get_grasp_info_simple()` remain
explicit `@implements` actions whose eye-in-hand implementations can reuse `perception/vision`. Higher-level
`locate_for_grasp()`, `locate_for_place()`, and `analyze_scene()` consume calibrated frames and detector hooks through
`perception/scene3d`; override them only for body-specific semantics.

## 4. Integrate detection, calibration, and correction

This section applies only to adapters declaring `vision.detection`. Validate the pipeline without an Agent first:

1. `grab_frames()` returns aligned RGB `uint8` and depth-in-meters `float32`.
2. Intrinsics exist and match the image dimensions.
3. The detector returns mask, box, score, and label for a text target.
4. The hand-eye transform direction and mounting model are correct.
5. Raw projection is checked at several workspace locations.
6. Only then configure XY correction, `z_correction_mm`, `grasp_z_offset_mm`, and `place_z_offset_mm`.

Do not hide depth-unit, transform-direction, or loose-camera errors behind large runtime offsets. Recalibrate first and
then add small residual correction. See [Calibrate Hand-Eye Geometry](calibrate-hand-eye.md) and the
[shared perception modules](../reference/adapter-reference.md#7-shared-perception-and-geometry-modules).

Manage a local GroundingDINO/SAM2 server as a Session sidecar. For an external service, disable `detector.spawn` but
preserve the same URL and failure semantics. An unreachable detector produces an empty result, which `api/defaults`
converts to `{"ok": false, "reason": "no_detection"}`.

## 5. Configure deployment YAML and Session

Hardware configuration describes the robot, services, calibration, and Agent switches; it does not store the user's
task. Prefer flat, explicit fields for a new adapter and support historical nested YAML only when migration requires it.

```yaml
adapter: my_robot
name: my_robot
can_port: can0
move_speed: 20
tool_offset_mm: 95.0
z_min_safe_mm: 50.0
x_min_mm: 0.0
x_max_mm: 600.0
y_min_mm: -400.0
y_max_mm: 400.0
detector:
  spawn: true
  host: 127.0.0.1
  port: 8114
```

Mark required connection fields, units, and dangerous defaults in `config_template.yaml`. Resolve relative calibration
paths against the YAML file rather than the caller's working directory.

Declare adapter-owned path keys as `path_fields: ClassVar[tuple[str, ...]]` on Config, such as `("calib_path", "urdf_path")`, and delegate `from_yaml` to `adapters._common.config.load_yaml_config(cls, path)`. Runtime snapshots and smoke use the same `parse_config(factory, data, source_dir=...)`, which returns a normalized copy and parses the effective config once. Paths expand environment variables and `~` and resolve against the source directory even when absent. Source-less `from_dict(data)` retains its existing semantics. New hardware path fields belong in the adapter declaration, not Runtime or validation scripts.

### Wire the common Builder

```python
from jiuwensymbiosis.adapters._common.builder import make_builder, make_detector_sidecar

build_my_session = make_builder(
    MyConfig,
    MyEnv,
    MyApi,
    api_kwargs_from_cfg=[
        "detector.url:detector_service_url",
        "z_correction_mm",
        "grasp_z_offset_mm",
        "place_z_offset_mm",
    ],
    sidecar_builders=[make_detector_sidecar()],
    resource_keys=lambda cfg: (f"can:{cfg.can_port}",),
)
```

A bare `api_kwargs_from_cfg` field passes under the same name; `cfg.path:api_name` renames a dotted nested path. Use the
legacy callback only when declarative mapping cannot express the transformation. Ordinary adapters do not need
`decorate`; it is reserved for Session objects that Config, Api arguments, and sidecars cannot express.

`resource_keys(cfg)` declares device identities from the effective config: the example uses a CAN interface; serial devices need canonical device paths, and ROS endpoints need their domain and command endpoint. Aliases for one device must produce identical keys. Common admission adds cameras, spawned detector services, and optional `physical_device_id`; explicit IDs cannot replace derived keys. Builders expose `.config_factory` and `.resource_keys` to Runtime without taking locks themselves. A bare `with session` does not provide cross-process admission: applications submit tasks through Runtime, while official maintenance entries use admitted_session.

Both template wiring examples include `resource_keys`. The validator's S-17 check verifies the factory and resource callback interfaces. Before constructing a stub session, `smoke_test_adapter.py` validates keys derived from the actual config and rejects empty device keys without `physical_device_id`; it never connects hardware. Interface checks cannot prove physical identity: verify that aliases for the same device produce identical keys.

The interactive `new_adapter` generator also emits `adapter:` and resource callbacks. CAN/serial/TCP keys use the configured endpoint; USB requires a serial number. ROS/custom connections remain placeholders: implement actual resource keys or supply top-level `physical_device_id` before starting through Runtime. The generator does not guess a ROS domain.

Validate all construction forms and always use context-managed lifecycle:

```python
# Build from Config, YAML, or a dictionary
session = build_my_session(MyConfig())
session = build_my_session.from_yaml("configs/my_robot/default.yaml")
session = build_my_session.from_dict({"name": "test", "can_port": "can0"})

with session:
    print(session.describe())
```

## 6. Configure software and hardware safety boundaries

SafetyRail checks Cartesian Z, XY workspace, and joint soft limits before dispatch and transparently unwraps
`RobotControlTool` actions. A rejected call raises a corrective `ValueError` and must not reach hardware.

### Boundary configuration and ownership

```yaml
z_min_safe_mm: 50.0
x_min_mm: 0.0
x_max_mm: 600.0
y_min_mm: -400.0
y_max_mm: 400.0
joint_limits:
  J1: [-150.0, 150.0]
  J2: [-100.0, 100.0]
```

- `z_min_safe` is a TIP-frame floor; the Driver's FLANGE floor includes the tool offset.
- `workspace_bounds` order is `(xmin, ymin, xmax, ymax)`.
- `joint_limits` key order and units match `move_joint(q)`.
- Test just below, exactly at, and just above every boundary.
- SafetyRail is a software precheck; Driver and controller hard limits remain required.
- Software `emergency_stop()` does not replace a physical E-stop, guarding, or operating procedure.

## 7. Run automated validation and hardware acceptance

Run static and generated-tool checks first:

```bash
python scripts/validate_adapter.py --module jiuwensymbiosis.adapters.my_robot
python scripts/smoke_test_adapter.py --module jiuwensymbiosis.adapters.my_robot
```

### Tests and staged acceptance

Unit tests cover at least:

- capability intersection and generated tool metadata;
- repeated connect/disconnect and failed-connect cleanup;
- observation fields, units, and unavailable sensors;
- TIP/FLANGE, camera/base, and tilted-tool transforms;
- Z, XY, joint boundaries, and non-finite inputs;
- every suction or gripper capability branch;
- absent, malformed, and valid calibration;
- sidecar startup and abnormal Session teardown;
- JSON-serializable results from every tool.

Accept real hardware in increasing risk order:

1. Inspect configuration, device enumeration, and camera without motor power.
2. Connect at low speed, read state, and disconnect.
3. Verify Home and one central workspace pose.
4. Verify boundary rejection and confirm no rejected command reaches the controller.
5. Verify release, recovery, and abnormal disconnect.
6. Check visual projection at several workspace positions.
7. Run a complete Agent task last.

Keep an operator and reachable physical E-stop present. Record robot model, firmware, SDK, tool offset, calibration file,
and acceptance results with the deployment configuration.

## 8. Troubleshoot common failures

| Symptom | Check first | Action |
|---|---|---|
| unknown capability | Spelling and `KNOWN_CAPABILITIES` | Use an existing value or update vocabulary, action, Env, validator, and tests together |
| Expected tool missing | Env capability and Api actions | Inspect `effective_capabilities` and the validator |
| Repeated connect fails | `connect()` idempotence | Publish `low_level` only after success and fully clean failure paths |
| `no_detection` | Service, models, frames, prompt, thresholds | Test the detector separately; do not duplicate the shared pipeline |
| Projection has a fixed or directional error | Depth units, intrinsics, transform direction, mounting | Recalibrate and ensure the raw seam does not apply correction twice |
| TIP/FLANGE Z is confused | Public tool semantics and tool offset | Compare Api target to Driver command; use a full transform for a tilted tool |
| SKILL.md is not loaded | Agent `enable_skill` and resource path | Confirm `RobotControlTool` assembly; this is not a hardware capability |

For interface and parameter questions, return to the [Robot Adapter Reference](../reference/adapter-reference.md). If
the mock example itself does not pass, return to [Build Your First Robot Adapter](../tutorial/02-build-first-adapter.md)
before debugging real hardware.
