# 机器人环境、能力与工具 API 参考

> 类别：Reference。本页以 `jiuwensymbiosis.env`、`jiuwensymbiosis.api` 和 `jiuwensymbiosis.tools` 的公开代码为基线。

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

位姿字段由机器人约定；SCARA 通常使用 `x/y/z/r`，六轴通常使用 `x/y/z/rx/ry/rz`。关节角单位必须遵循对应 Env 的约定（`env.joint_units`；未声明视为未知），深度默认以米为单位。

## `BaseRobotEnv`

子类必须实现：

```python
connect() -> None
disconnect() -> None
get_observation() -> RobotObservation
home() -> None
```

`home` 之所以是抽象方法：它背后是唯一的无条件动作（`ActionSpec(capability=None)`），若给出笛卡尔默认实现，就会把 `motion.cartesian`
悄悄塞进能力门拦不住的本体里。可选生命周期方法为 `reset()` 和 `emergency_stop()`。常用属性包括 `capabilities`、`low_level`、`z_min_safe`、`workspace_bounds`、`joint_limits`、`home_pose`、`tool_offset_mm`、`joint_units`、`default_orientation_policy`、`base_step_limits`、`lift_limits`、`waist_step_limit_rad`、`cameras`、`urdf_path`/`arm_chains`/`arm_joints`。

`has(capability)` 检查能力声明。子类创建时，未知能力会立即触发 `ValueError`。

## `BaseRobotApi` 与动作词表

`BaseRobotApi(env)` 保存 Env 引用，并在两个地方提供 `@implements`：

- `home`（唯一无条件动作，重写它委托 `env.home()`）
- 各本体加的其余动作

`capabilities` 属性**反推自本体实现的动作的 spec**（每个 `@implements` 贡献自己 `ActionSpec` 的能力），再加上声明的 marker 能力类属性（如 `motion.servo`、`planning.reachability`——它们没有对应动作，只能靠属性声明）。**实现哪个动作就自动具备哪个能力**，不会广告本体没有的能力。

动作词表在 `jiuwensymbiosis/api/actions.py`。每条动作声明：描述、能力门、参数名、结果形状、`requires`/`provides`/`invalidates`、`opens_access`/`closes_access`、`produces_location`/`consumes_location`/`invalidates_locations`、`planner_visible`。通用实现（转发 Env 动词就能完成）在 `api/defaults.py`，适配器按需取用。

## 已知 Capability

- `motion.cartesian`
- `motion.joint`
- `motion.servo`
- `motion.base`
- `motion.base_servo`
- `motion.lift`
- `motion.waist`
- `motion.goal`
- `motion.dual_arm`
- `motion.jump`
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

实际词表以 `jiuwensymbiosis.env.base.KNOWN_CAPABILITIES` 为准。

## RGB-D 目标定位与接近

`locate_target(object_name, reference=None, relation="on")` 由 `vision.detection` 门控。
返回 `source="rgbd_perception"`、底座前/左/上坐标系下的 `point_base_mm`、水平距离
`distance_m`、方位角 `bearing_rad`（向左为正）、帧 ID 和采集时间。测量点是可见表面的
中位点（`point_kind="visible_point_median"`），不代表完整物体中心。
`object` 回显查询；`detection_label`、`detection_score` 和 `box_px` 是模型预测。
决定接近前应检查同帧 RGB/分割叠图。缺失、歧义或深度不足均失败；不能简单丢弃无法
测量的候选，再选择另一个同类对象。

`approach_target(object_name, target_distance_m=1.2, reference=None, relation="on")`
由 `motion.goal` 门控，以有限的转向/前进步和每次决策前两帧新鲜关联观测闭环接近。
成功需要到达目标范围且**实测停车**；预算耗尽是失败，不提供全局寻路或抓取。
单个能力标签不验证全部前提：适配器绑定只有在标定 RGB-D、分割、实测底座位姿、
运动净空检查和实测停车均可用时才能提供该动作。旧 `motion.goal` 适配器不会自动获得它。

### 绑定扩展契约

| 扩展点 | 契约与缺失行为 |
|---|---|
| `env.navigate_relative(dx_m, dy_m=0, dyaw_rad=0)` 或 `api.nav_relative` | 遵循 `env.protocol.MeasuredBaseDriver`，返回 `contracts.MeasuredBaseMotionResult`。先转后平移，阻塞并实测停车；`(0, 0, 0)` 也必须请求停车确认。只有 `ok=True`、缺少字段或停车未知均阻止感知/下一步，返回 `stop_unconfirmed`。 |
| `env.check_target_step(forward, turn, frame_id, tuning) -> str \| None` | 遵循 `motion.target_approach.TargetStepGuard`；任何非零动作（含纯转向）前必需。核对指定帧的新鲜度/参考系代次与完整本体扫掠空间；`None` 放行，非空原因拒绝。缺少钩子返回 `clearance_unavailable`，缺失/遮挡证据必须拒绝。 |
| `api._grab_target_frame(camera) -> CameraFrame \| None` | 可选目标采集覆盖；否则依次回退到 `_grab_calibrated_frame(camera)`、`env.grab_calibrated_frame(camera)`。必须返回实际对齐传感数据。 |
| `env.validate_frame(frame) -> None` | 可选附加参考系代次/新鲜度校验，在感知结束、发布缓存前调用；无效时抛出 `FrameValidationError(reason)`。也用于共享场景感知。 |
| `env.validate_target_frame(frame) -> None` | 同一缓存发布边界上的可选目标专用校验；无效时抛出 `FrameValidationError(reason)`。缺少校验器不会增加适配器检查；本体重置/生命周期需要附加检查时由绑定层提供。 |

共享层保留几何、帧顺序和关联检查，DetectorClient 还检查采集年龄。可选钩子不替代
这些检查；自定义采集/检测器绑定必须确保实际采集新鲜度。旧驱动不能用命令确认或
时间流逝伪造 `settled=True`、`stop_confirmation="measured"`。

`CameraFrame` 保存对齐 RGB/深度、内参和 `tf_base_cam`（相机→底座，平移为**毫米**；
深度为**米**）。应提供唯一 `frame_id` 和主机单调时钟上的采集时间，自动生成的
ID/时间不能证明缓冲帧新鲜。目标关联额外消费与采集对齐的 `metadata`：

| 字段 | 含义 |
|---|---|
| `base_sample.position_m` | 稳定参考系下的实测底座位置，有限三维向量，米。 |
| `base_sample.body_yaw_rad` | 同一参考系下的有限实数偏航角，弧度。 |
| `base_sample.odom_frame` | 稳定里程计参考系/代次 ID；里程计重置时必须更新。 |
| `scene_instance_id` | 可选场景/重置代次 ID，与里程计独立。 |

至少需要一个参考系 ID。仅有场景 ID 时，底座位姿必须使用该稳定场景坐标系，且每次
坐标重置均更新 ID。两者同时提供时必须分别一致；变化、出现或消失都会使关联失效。
对象真值不能替代传感证据或实测底座位姿。

默认最小目标距离 0.8 米表示到可见中位点的距离，不保证表面净空。前进前，两帧中
更近的可见表面减去本体半径及净空后，须至少剩余 `min(forward, distance_tolerance_m)`。
默认值满足 `0.8 >= 0.55 + 0.15 + 0.05`；突出几何或较大本体仍可能触发
`insufficient_target_clearance`，应保留该拒绝行为。

`simulation.truth` 门控 `sim_list_entities`、`sim_get_robot_state`、`sim_get_entity_state`，
返回明确标注 `simulation_ground_truth` 的快照，不是 marker 能力或传感替代品。
`navigate_to_pose`、`navigate_to_entity` 是另外两项 `motion.goal` 契约。
本层提供共享契约/算法，Genesis 生产绑定后续接入。检测器工厂会转发 `required_backend`，
但只有调用方显式传值才校验后端一致性。GroundingDINO 接受与框对齐的已解码
`text_labels` 或字符串 `labels`；缺失、数值或长度不符的标签视为服务失败，不能降级为
空检测或将请求文本当作预测标签。

## `@implements(SPEC)`

```python
implements(spec: ActionSpec) -> Callable   # api/actions.py
```

方法变成工具的唯一入口。它把一个 `ToolMeta` 挂到方法上，内含 `spec` 与 `input_params`——后者由本体自己的签名推导（按 `spec.params` 过滤，再由 `spec.param_schema` 细化）。描述、capability、tags、requires / provides / invalidates、位置新鲜度等契约字段全部回读自 spec，所以实现方没有渠道对规划器说词表之外的话。

若签名接不下 spec 承诺的参数，导入时即抛 `ContractViolation`。

bring-up、标定与调试视图**不是动作**：不加装饰器，用 `scripts/` 下的脚本驱动。

## 动作契约（可调用，也可规划）

每个 `ActionSpec` 还携带规划契约，供两级规划器推导合法顺序：

- `result` —— 结果字段 JSON Schema，自动派生自 `TypedDict`（或并集）；权威源在 `jiuwensymbiosis/contracts.py`（不归属任何层），失败/成功形状取并集
- `requires`/`provides`/`invalidates` —— 本体自身状态，基于 `api/state.py:KNOWN_STATE_TOKENS`（`payload.held`/`payload.clear`/`payload.stowed`/`body.home`）
- `opens_access`/`closes_access` —— 屏障开/闭语义（先拉开抽屉才能拿里面的东西；`parse_sequence` 消费它）
- `produces_location`/`consumes_location`/`invalidates_locations` —— 位置新鲜度（感知目标在哪的动作→产生；移动底盘的动作→作废所有从旧视角测得的位置）

契约不编码顺序；`parse_sequence` 只拒绝前置条件不满足的排列。`WorldState.snapshot(session)` 在运行时用同一词表汇报当前状态（观测覆盖推想；缺失即未知，从不 false）。

## 工具构建

```python
build_robot_tools(api, *, env=None, allow=None, deny=None, planner_only=False) -> list[Any]
list_tool_meta(api, *, env=None) -> list[dict]
```

传入 Env 时，有效工具按 `api.capabilities ∩ env.capabilities` 门控；`allow` 和 `deny` 使用工具名过滤；`planner_only=True` 只发射 `planner_visible` 的共享词表动作（两条 agent 路径均以此调用）。能力来自动作自身的 `ActionSpec`，从不来自哪个类声明了方法。

## 聚合工具与代码工具

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

`available_actions` 返回可派发动作；`invoke({"action": ..., "params": {...}})` 返回 `ToolOutput`。安全 Rail 透明解包 `action`/`params`。

```python
InProcessCodeTool(globals_provider)
```

`run(code)` 在进程内执行代码并返回结果字典；`as_openjiuwen_tool(**kwargs)` 构造对应的 `LocalFunction`。调用方必须提供每次执行时返回全局变量字典的 `globals_provider`。


## jump

`jump()` 使用本体已验证的固定轨迹，返回 `ok`、`settled`、`airborne` 和 `reason`。
要求 `payload.clear`，使 `body.home` 和已有感知位置失效；成功必须实测腾空、落地与停稳。
Jumper 必须先显式张开当前夹爪。取消会等待落地，不能在半空中停止。
