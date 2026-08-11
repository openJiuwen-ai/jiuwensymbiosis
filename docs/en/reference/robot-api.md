# Robot Environment, Capability, and Tool API Reference

> Category: Reference. The [Chinese source](../../zh/reference/robot-api.md) is authoritative.

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

Pose fields follow the adapter convention: SCARA commonly uses `x/y/z/r`, while a six-axis body uses
`x/y/z/rx/ry/rz`. Joint units must match the Env's `move_joint()` contract. Camera depth is normally represented in
meters at acquisition boundaries.

## `BaseRobotEnv`

Subclasses implement:

```python
connect() -> None
disconnect() -> None
get_observation() -> RobotObservation
```

Optional lifecycle hooks are `reset()` and `emergency_stop()`. The base class supplies default delegations for
`home`, `get_flange_pose`, `move_to_flange`, `move_joint`, `set_end_effector`, and `grab_rgb` when the low-level driver
satisfies the corresponding Protocol.

Common contract properties include:

- `capabilities` and `low_level`;
- `z_min_safe`, `workspace_bounds`, and `joint_limits`;
- `home_pose` and `tool_offset_mm`.

`has(capability)` checks the declaration. An unknown capability raises `ValueError` when the subclass is created.

## `BaseRobotApi` and capability Mixins

`BaseRobotApi(env)` retains the Env reference. Its `capabilities` property is the union of `capability` declarations in
the Mixin MRO.

| Mixin | Capability | Main tools |
| --- | --- | --- |
| `MotionMixin` | `motion.cartesian` | home, pose query, cartesian and directional motion |
| `JointMotionMixin` | `motion.joint` | joint-space movement |
| `SuctionMixin` | `grasp.suction` | suction engage/release |
| `ParallelGripperMixin` | `grasp.parallel` | gripper open/close |
| `VisionMixin` | `vision.detection` | image, detection, projection, and scene analysis |

Motion, joints, grasp, and `get_image()` delegate to the Env by default. `VisionMixin` implements the shared
`get_grasp_info_simple()` and `pixel_to_base_xyz()` pipeline; an adapter provides `_project_pixel_to_base_raw()` and
implements `analyze_scene()` only when it needs that optional higher-level tool.

## Known capabilities

- `motion.cartesian`
- `motion.joint`
- `motion.servo`
- `grasp.suction`
- `grasp.parallel`
- `vision.camera`
- `vision.depth`
- `vision.detection`
- `vision.eye_to_hand`
- `sorting.command`
- `speech.tts`

The source of truth is `jiuwensymbiosis.env.base.KNOWN_CAPABILITIES`.

## `@robot_tool`

```python
robot_tool(
    _func=None,
    *,
    name=None,
    desc=None,
    capability=None,
    input_params=None,
    tags=None,
)
```

The decorator attaches `ToolMeta` to an unbound method. When omitted, the name comes from the function name, the
description from the first docstring line, and the JSON input schema from type annotations and defaults. An adapter
override inherits the Mixin metadata unless it redecorates the method.

## Tool generation

```python
build_robot_tools(api, *, env=None, allow=None, deny=None) -> list[Any]
list_tool_meta(api, *, env=None) -> list[dict]
```

When an Env is supplied, generated tools are gated by `api.capabilities ∩ env.capabilities`. `allow` and `deny` filter
by tool name.

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

`available_actions` lists dispatchable actions. `invoke({"action": ..., "params": {...}})` returns a `ToolOutput`.
Safety and trace Rails transparently unwrap this aggregate call to the real action.

```python
InProcessCodeTool(globals_provider)
```

`run(code)` executes Python in process and returns a result dictionary. `as_openjiuwen_tool(**kwargs)` creates a
`LocalFunction`. `globals_provider` must return a fresh globals mapping for each execution.
