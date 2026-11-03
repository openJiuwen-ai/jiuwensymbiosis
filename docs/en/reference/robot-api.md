# Robot Environment, Capability, and Tool API Reference

> Category: Reference. The [Chinese source](../../zh/reference/robot-api.md) is authoritative. This page is based on the
> public code of `jiuwensymbiosis.env`, `jiuwensymbiosis.api`, and `jiuwensymbiosis.tools`.

## `RobotObservation`

```python
RobotObservation(
    pose: dict | None = None,
    joints: list[float] | None = None,
    rgb: numpy.ndarray | None = None,
    depth: numpy.ndarray | None = None,
    extra: dict = {},
)
```

Pose fields follow the adapter convention: SCARA commonly uses `x/y/z/r`, while a six-axis body uses `x/y/z/rx/ry/rz`. Joint units must match the Env's convention (`env.joint_units`; unstated means unknown). Camera depth is normally represented in metres at acquisition boundaries.

## `BaseRobotEnv`

Subclasses implement:

```python
connect() -> None
disconnect() -> None
get_observation() -> RobotObservation
home() -> None
```

`home` is abstract because it backs the one unconditional action (`ActionSpec(capability=None)`): a Cartesian default would smuggle `motion.cartesian` into bodies the capability gate cannot stop. Optional lifecycle hooks are `reset()` and `emergency_stop()`. Common contract properties include `capabilities`, `low_level`, `z_min_safe`, `workspace_bounds`, `joint_limits`, `home_pose`, `tool_offset_mm`, `joint_units`, `default_orientation_policy`, `base_step_limits`, `lift_limits`, `waist_step_limit_rad`, `cameras`, and `urdf_path`/`arm_chains`/`arm_joints`.

`has(capability)` checks the declaration. An unknown capability raises `ValueError` when the subclass is created.

## `BaseRobotApi` and the action vocabulary

`BaseRobotApi(env)` retains the Env reference and carries an `@implements` in two places:

- `home` (the one unconditional action; overrides to delegate `env.home()`)
- each body's additional actions

The `capabilities` property is **derived from the specs of the actions a body implements** (each `@implements` contributes its `ActionSpec`'s capability), plus any declared marker-capability class attribute (`motion.servo`, `planning.reachability` — which have no corresponding action). **Implementing an action automatically grants its capability**, and never advertises a capability the body has not got.

The action vocabulary is in `jiuwensymbiosis/api/actions.py`. Each action declares its description, capability gate, parameter names, result shape, `requires`/`provides`/`invalidates`, `opens_access`/`closes_access`, `produces_location`/`consumes_location`/`invalidates_locations`, and `planner_visible`. Generic implementations (one line of delegation to an Env verb) live in `api/defaults.py`, which an adapter takes as needed.

## Known capabilities

- `motion.cartesian`
- `motion.joint`
- `motion.servo`
- `motion.base`
- `motion.base_servo`
- `motion.lift`
- `motion.waist`
- `motion.goal`
- `motion.dual_arm`
- `grasp.suction`
- `grasp.parallel`
- `grasp.paddle`
- `vision.camera`
- `vision.depth`
- `vision.detection`
- `vision.eye_to_hand`
- `vision.search`
- `planning.reachability`
- `sorting.command`
- `speech.tts`
- `simulation.truth`

The source of truth is `jiuwensymbiosis.env.base.KNOWN_CAPABILITIES`.

## RGB-D target location and approach

`locate_target(object_name, reference=None, relation="on")` is gated by `vision.detection`.
It returns `source="rgbd_perception"`, a `point_base_mm` in forward/left/up base coordinates,
horizontal `distance_m`, `bearing_rad` (+left), frame ID and capture time. The point is the
visible surface's median (`point_kind="visible_point_median"`), not the full object centre.
`object` echoes the request; `detection_label`, `detection_score` and `box_px` are model
predictions. Inspect the same-frame RGB/overlay before choosing to approach. Missing,
ambiguous or insufficient-depth candidates fail; an unmeasured candidate cannot be
discarded just to select a different same-class object.

`approach_target(object_name, target_distance_m=1.2, reference=None, relation="on")` is
gated by `motion.goal`. It uses bounded turn/forward steps with two fresh associated
observations per decision. Success requires arrival **and measured stopping**; exhausting
the budget is failure. It provides no global routing or grasping. A single capability
tag does not validate all prerequisites: the adapter binding must admit this action only
when calibrated RGB-D, segmentation, measured base pose, guarded motion and measured
stopping are available. Existing `motion.goal` adapters do not automatically gain it.

### Binding extensions

| Extension | Contract and missing behavior |
|---|---|
| `env.navigate_relative(dx_m, dy_m=0, dyaw_rad=0)` or `api.nav_relative` | `env.protocol.MeasuredBaseDriver`; returns `contracts.MeasuredBaseMotionResult`. Turn then translate, block and measure stopping. Even `(0, 0, 0)` requests a stop confirmation. Plain `ok=True`, missing fields or unknown stopping prevent sensing/another step (`stop_unconfirmed`). |
| `env.check_target_step(forward, turn, frame_id, tuning) -> str \| None` | `motion.target_approach.TargetStepGuard`; required before any nonzero step, including pure rotation. Validate the identified frame's freshness/epoch and swept full-body envelope. `None` permits the step; a nonempty reason refuses it. Missing hook returns `clearance_unavailable`; missing/occluded evidence must refuse. |
| `api._grab_target_frame(camera) -> CameraFrame \| None` | Optional target capture override; otherwise `_grab_calibrated_frame(camera)`, then `env.grab_calibrated_frame(camera)`. Return actual aligned sensor data. |
| `env.validate_frame(frame) -> None` | Optional additional epoch/freshness validation after perception, before publishing caches. Raise `FrameValidationError(reason)` on invalid evidence. Applies to shared scene sensing too. |
| `env.validate_target_frame(frame) -> None` | Optional target-specific validation at the same publication boundary. Raise `FrameValidationError(reason)` on invalid evidence. Absent validators add no adapter checks; the binding must provide them when its reset/lifecycle semantics require them. |

The shared layer retains geometry, frame-order and association checks. DetectorClient also
checks capture age. Optional hooks do not replace these checks; custom capture/detector
bindings must enforce their own acquisition freshness. No legacy driver may synthesize
`settled=True` or `stop_confirmation="measured"` from command acknowledgement or elapsed time.

`CameraFrame` carries aligned RGB/depth, intrinsics and `tf_base_cam` (camera→base,
translation in **mm**; depth in **m**). Supply a unique `frame_id` and capture time on the
host monotonic clock; automatic IDs/timestamps cannot establish buffered-frame freshness.
Target association additionally consumes capture-aligned `metadata`:

| Field | Meaning |
|---|---|
| `base_sample.position_m` | Finite 3-vector of measured base position in a stable reference, metres. |
| `base_sample.body_yaw_rad` | Finite real yaw in the same reference, radians. |
| `base_sample.odom_frame` | Stable odometry reference/epoch ID; change it when odometry resets. |
| `scene_instance_id` | Optional scene/reset epoch ID, independent of odometry. |

At least one reference ID is required. A scene-only ID is sufficient only when base pose
uses that stable scene frame and the ID changes on every coordinate reset. When both are
provided, both must match: change, appearance or disappearance invalidates association.
Object ground truth must never replace the sensor evidence or measured base pose.

The default minimum target distance (0.8 m) is a distance to the visible-point median,
not guaranteed surface clearance. Before advancing, the nearer of the two observed near
surfaces must leave `min(forward, distance_tolerance_m)` after subtracting body radius and
clearance. Defaults satisfy `0.8 >= 0.55 + 0.15 + 0.05`; protruding geometry or larger bodies
can still yield `insufficient_target_clearance`. Preserve that refusal.

`simulation.truth` gates `sim_list_entities`, `sim_get_robot_state`, and `sim_get_entity_state`;
these return explicit `simulation_ground_truth` snapshots and are not marker capabilities
or sensor substitutes. `navigate_to_pose` and `navigate_to_entity` are additional
`motion.goal` contracts. This layer provides shared contracts/algorithms; production
Genesis bindings follow separately. The detector factory forwards `required_backend`,
but backend equality is enforced only when the caller supplies it. GroundingDINO accepts
decoded `text_labels` or string `labels` aligned with boxes; missing/numeric/misaligned
labels are service failures, not empty detections or request-caption predictions.

## `@implements(SPEC)`

```python
implements(spec: ActionSpec) -> Callable   # api/actions.py
```

The one way a method becomes a tool. It attaches a `ToolMeta` holding `spec` plus `input_params`, the call schema derived from this body's signature (filtered to `spec.params`, refined by `spec.param_schema`). Every contract field — description, capability, tags, requires / provides / invalidates, location freshness — is read back off the spec, so an implementation has no channel for saying something the vocabulary does not.

A signature that cannot accept a parameter the spec promises raises `ContractViolation` at import time.

Bring-up, calibration and debug views are **not** actions: leave them undecorated and drive them from `scripts/`.

## The action contract (callable, and plannable)

Each `ActionSpec` additionally carries a planning contract that the two-tier planner uses to derive a legal order:

- `result` — JSON Schema of result fields, auto-derived from a `TypedDict` (or a union); the authoritative source is `jiuwensymbiosis/contracts.py` (owned by no layer), with success/failure shapes merged
- `requires`/`provides`/`invalidates` — robot self-state, over `api/state.py:KNOWN_STATE_TOKENS` (`payload.held`/`payload.clear`/`payload.stowed`/`body.home`)
- `opens_access`/`closes_access` — barrier open/close effects, consumed by `parse_sequence`
- `produces_location`/`consumes_location`/`invalidates_locations` — location freshness (an action that senses where something is *produces*; one that moves the base *invalidates* every prior location)

A contract never encodes an order; `parse_sequence` rejects only permutations whose pre-conditions do not hold. `WorldState.snapshot(session)` reports the same vocabulary at runtime (observation overrides belief; an absent token is unknown, never false).

## Tool generation

```python
build_robot_tools(api, *, env=None, allow=None, deny=None, planner_only=False) -> list[Any]
list_tool_meta(api, *, env=None) -> list[dict]
```

When an Env is supplied, generated tools are gated by `api.capabilities ∩ env.capabilities`. `allow` and `deny` filter by tool name. `planner_only=True` emits only shared-vocabulary actions whose specs are `planner_visible`; both agent paths use this setting. The capability comes from the action's own `ActionSpec`, never from whichever class declares the method.

## Aggregated and code tools

```python
RobotControlTool(
    api,
    *,
    env=None,
    name="robot_control",
    description=None,
    agent_id=None,
)
```

`available_actions` lists dispatchable actions. `invoke({"action": ..., "params": {...}})` returns a `ToolOutput`. Safety rails transparently unwrap `action`/`params`.

```python
InProcessCodeTool(globals_provider)
```

`run(code)` executes Python in process and returns a result dictionary. `as_openjiuwen_tool(**kwargs)` creates a `LocalFunction`. `globals_provider` must return a fresh globals mapping for each execution.
