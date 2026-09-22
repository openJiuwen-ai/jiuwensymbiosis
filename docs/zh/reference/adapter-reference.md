# 机器人适配器参考

> 类别：Reference。本页集中查询适配器的稳定契约、参数和实现位置。

第一次实现适配器请阅读[构建第一个机器人适配器](../tutorial/02-build-first-adapter.md)；从 Mock 接入厂商 SDK 和真机请阅读[移植机器人硬件适配器](../how-to/port-hardware-adapter.md)。本页不提供顺序操作流程。

## 1. 适配器文件

```text
jiuwensymbiosis/adapters/<name>/
├── __init__.py
├── config.py
├── lowlevel.py
├── env.py
├── api.py
├── session.py
└── config_template.yaml

# 可选：手眼标定 wrapper（模板 calibration.py 拷贝到独立目录）
jiuwensymbiosis/calibration/adapters/<name>.py
```

| 文件 | 稳定职责 | 不应包含 |
|---|---|---|
| `__init__.py` | 导出命名后的 Session builder，作为适配器包入口 | 硬件或业务实现 |
| `config.py` | 配置 dataclass、`from_dict()`、`from_yaml()` | 硬件连接和任务文本 |
| `lowlevel.py` | 厂商 SDK、CAN、串口、Socket、相机和执行器 I/O | Agent、Rail、`@implements` |
| `env.py` | 能力、生命周期、观测、安全属性、Driver 包装 | 提示词和厂商流程编排 |
| `api.py` | `@implements(SPEC)` 绑定、机型几何、相机/标定/检测钩子 | 重复的检测/校正流水线 |
| `session.py` | Config/Env/Api、sidecar 和附加对象接线 | 大段业务实现 |
| `config_template.yaml` | 可部署起点及字段注释 | 用户任务和秘密凭据 |
| `calibration.py`（可选） | 手眼标定 wrapper，暴露 `CALIBRATION_ADAPTER_SPEC` | 标定求解/质检逻辑（归 calibration 子系统） |

模板位于 `templates/xxx_adapter/`。

## 2. 动作词表、能力与工具

动作契约由 `api/decorators.py:ActionSpec` 声明，共享词表实例集中在 `jiuwensymbiosis/api/actions.py`；能力词表由 `jiuwensymbiosis/env.base.KNOWN_CAPABILITIES` 定义。每个 `ActionSpec` 声明：名字、描述、能力门、参数名、结果形状、前置条件与效果（含屏障开/闭 `opens_access`/`closes_access`）、位置新鲜度、是否对规划器可见。

| Capability | 动作（`ActionSpec`） |
|---|---|
| `motion.cartesian` | `goto_xyzr`、`goto_pose`、`move_direction`、`get_pose`、`get_home_pose` |
| `motion.joint` | `move_joint`、`move_named_joint`、`get_joint_positions` |
| `motion.servo` | （无独立公共动作；由 fast path 实时伺服 op `track_detect`/`track_grasp` 经 `api.servo_to_tip`/`env.servo_to_flange` 驱动） |
| `motion.base` | `navigate_relative`、`rotate_base`、`drive_arc` |
| `motion.base_servo` | （连续底盘驱动原语） |
| `motion.lift` | `set_lift_pose`、`lift_to_clearance` |
| `motion.waist` | `turn_waist` |
| `motion.goal` | `approach_for_grasp`、`approach_for_place` |
| `motion.dual_arm` | `dual_arm_grasp`、`dual_arm_place` |
| `grasp.parallel` | `open_gripper`、`close_gripper` |
| `grasp.suction` | `activate_suction`、`deactivate_suction` |
| `grasp.paddle` | （经 `dual_arm_grasp` 实现） |
| `vision.camera` | `get_image`、`pixel_to_base_xyz` |
| `vision.detection` | `get_grasp_info_simple`、`locate_for_grasp`、`locate_for_place`、`analyze_scene` |
| `vision.search` | `search_target` |

Env 手动声明硬件能力；Api 的能力**反推自它实现的动作的 spec**（加上声明的 marker 能力类属性）；有效工具能力为二者交集。Env 可声明无对应动作的标记能力，Api 有而 Env 无的能力不会生成工具。

`adapters/_common/capability_spec.py` 提供能力→动作、能力→Driver 成员映射（`CAPABILITY_ACTIONS`、`CAPABILITY_DRIVER_MEMBERS`），供验证器与生成器共用，与 `api/defaults.py`、`env/protocol.py` 保持一致。

## 3. Driver Protocol 契约

Protocol 定义在 `jiuwensymbiosis/env/protocol.py`，**按能力切片**——一个能力一个协议，移动双臂本体没有 flange pose，台架臂没有轮子，单一"上帝协议"会把其中一方逼成 `NotImplementedError` 桩；`RobotDriver` 因此只有每个驱动都有的 `close()`，各能力加自己的兄弟协议。具体 Driver 使用结构化类型（`typing.Protocol`），不要求继承这些类。

| Protocol | 对应能力 | 必需成员 |
|---|---|---|
| `RobotDriver` | 所有驱动 | `close()`（幂等） |
| `CartesianDriver` | `motion.cartesian` | `home_pose`、`z_min_safe`、`flange_z_min_safe`、`tool_offset_mm`、`home()`、`get_pose()`、`move_to_pose_blocking(pose, ...)` |
| `JointDriver` | `motion.joint`（索引式，整向量） | `get_angles()`、`move_joint_blocking(q, timeout_s=...)` |
| `NamedJointDriver` | `motion.joint`（命名式，可部分命令） | `get_joint_positions()`、`move_joints_blocking(targets, timeout_s=...)` |
| `ServoDriver` | `motion.servo` | `servo_to_pose(pose)`（非阻塞） |
| `BaseDriver` | `motion.base`/`motion.goal` | `navigate_relative(dx_m, dy_m=0, dyaw_rad=0)`、`navigate_arc(radius_m, dyaw_rad)` |
| `ContinuousBaseDriver` | `motion.base_servo` | `start_base_drive()`、`base_drive_running(handle)`、`steer_base_drive(handle, bearing_rad)`、`hold_base_drive(handle)`、`stop_base_drive(handle)` |
| `LifterDriver` | `motion.lift` | `set_lifter(q_lifter)` |
| `WaistDriver` | `motion.waist` | `turn_waist(delta_rad)` |
| `DualArmDriver` | `motion.dual_arm` | `home()`（双臂归位；具体协同由共享实现 + 本体钩子完成） |
| `CameraDriver` | `vision.camera` | `intrinsics`、`grab_frames()` |
| `SuctionDriver` | `grasp.suction` | `suction_state`、`suction_di_last`、`set_suction(on)` |
| `GripperDriver` | `grasp.parallel` | `set_gripper(on)`、`gripper_state` |
| `VisionDriver` | eye-in-hand `vision.detection` | `tf_flange_cam`、`calibration` |
| `HandGuidingDriver`（可选） | 人工拖动示教 | `hand_guiding(*, include_end_effector=False)` 上下文管理器（释放力矩→人工摆位→恢复前先同步目标，失败抛 `HandGuidingRecoveryError`） |

关键语义：

- `get_pose()`/`home_pose` 返回厂商 Pose，可为 SCARA 的 `(x,y,z,r)` 或六轴的 `(x,y,z,rx,ry,rz)`；
- `move_to_pose_blocking()` 接收 FLANGE 目标，必须阻塞至完成或失败；TIP↔FLANGE 转换属于 Api 层（`goto_xyzr`）；
- `move_joint_blocking()` 的单位由适配器约定，但必须与 `joint_limits` 一致；
- `servo_to_pose()` 非阻塞，显式 `False` 表示本周期未推进，`True` 或兼容的 `None` 表示已接受；
- 底盘动词用**米**（检测用毫米），返回 `{ok, reason, ...}`；
- `grab_frames()` 返回对齐的 `(rgb_uint8, depth_m_float32)` 或 `None`；
- `close()` 必须幂等。

当前具名的 `VisionDriver` Protocol 只覆盖 eye-in-hand 标定。SO-101 等 eye-to-hand 适配器以机型专属的结构化接口暴露 `tf_base_cam` 和 `calibration`，并声明 `vision.eye_to_hand` 标记能力；目前还没有独立命名的 eye-to-hand Protocol。

多能力驱动可定义组合 Protocol 收紧 `low_level` 类型；`PiperFullDriver` 是
`CartesianDriver + JointDriver + CameraDriver + GripperDriver + VisionDriver` 的现有示例。

## 4. BaseRobotEnv 契约

`BaseRobotEnv` 的抽象方法：

| 方法 | 语义 |
|---|---|
| `connect()` | 打开硬件连接，必须幂等 |
| `disconnect()` | 释放硬件，任何状态调用都应安全 |
| `get_observation()` | 最佳努力快照，瞬时传感器缺口不应导致整次调用失败 |
| `home()` | 回到本体的安全姿态；`home` 是唯一无条件动作，不允许由笛卡尔默认代劳，故声明为抽象方法 |

`RobotObservation` 字段：

| 字段 | 类型 | 约定 |
|---|---|---|
| `pose` | `dict | None` | SCARA 常用 `x,y,z,r`；六轴常用 `x,y,z,rx,ry,rz` |
| `joints` | `list[float] | None` | 单位按机器人约定 |
| `rgb` | `np.ndarray | None` | H×W×3 `uint8` |
| `depth` | `np.ndarray | None` | H×W `float32` 米，与 RGB 对齐 |
| `extra` | `dict` | 夹爪、力矩或状态标志等轻量信息 |

Env 属性：

| 属性 | 默认 | 消费者 |
|---|---|---|
| `low_level` | `None` | Env 默认动词和视觉受控穿透 |
| `z_min_safe` | `None` | SafetyRail 的 TIP Z 下限 |
| `workspace_bounds` | `None` | SafetyRail，顺序 `(xmin,ymin,xmax,ymax)` |
| `joint_limits` | `None` | SafetyRail，键顺序对应关节索引 |
| `home_pose` | `None` | `Api` 的 `get_home_pose()` |
| `tool_offset_mm` | `0.0` | TIP↔FLANGE 几何 |
| `joint_units` | `None` | `move_joint`/观测关节的单位（`"deg"`/`"rad"`；未声明视为未知） |
| `default_orientation_policy` | `None` | `goto_xyzr` 省略时的默认倾角 |
| `base_step_limits` | `None` | SafetyRail，底盘单命令 `(max|translation|m, max|turn|rad)` |
| `lift_limits` | `None` | SafetyRail，`set_lifter` 的软限位 |
| `waist_step_limit_rad` | `None` | SafetyRail，`turn_waist` 单命令上限 |
| `cameras` | `(None,)` | 可感知相机列表（最佳优先）；`grab_calibrated_frame(camera)` |
| `holding_payload` | 无 | RecoveryRail 归位前查它——还抱着东西的本体不能盲目归位 |
| `urdf_path`/`arm_chains`/`arm_joints` | `None` | 派生 `planning.reachability`；双臂各自驱动的关节 |

已有默认委托：

| Env 动词 | Driver 目标 |
|---|---|
| `get_flange_pose()` | `driver.get_pose()` |
| `move_to_flange(pose)` | `driver.move_to_pose_blocking(pose)` |
| `move_joint(targets)` | 命名式直达 `move_joints_blocking`；索引式读当前配置后改写命名条目再 `move_joint_blocking(q)` |
| `set_end_effector(True/False)` | 按能力调用 `set_gripper()` 或 `set_suction()` |
| `grab_rgb()` | 默认返回 `get_observation().rgb` |

`reset()` 和软件 `emergency_stop()` 默认无操作。物理急停始终属于硬件安全系统。

## 5. Api、`@implements` 与覆写规则

Api 子类化 `BaseRobotApi`，用 `@implements(SPEC)` 绑定每条动作。内置默认行为由 `api/defaults` 提供，覆盖常见运动、关节、抓取、取图和视觉流程。

| 情况 | 处理 |
|---|---|
| 行为与公共语义一致 | `@implements(SPEC)` 后 `return defaults.<action>(self, ...)` 一行转发 |
| TIP/FLANGE、姿态或字段名不同 | 覆写方法体 |
| 需要机型专属语义 | 仍是 `@implements(SPEC)`，契约来自 spec；实现只写"怎么做" |
| 硬件有词表外的新能力 | 先完成能力扩展契约，再新增动作（在 `api/actions.py` 加 `ActionSpec` + 在 Api 上加 `@implements`） |

`@implements` 将 `ToolMeta`（spec + 由这个本体签名推导的 `input_params`）挂到方法上；签名接不下 spec 承诺的参数时，导入即抛 `ContractViolation`。bring-up、标定与调试视图**不是动作**：不加装饰器，用 `scripts/` 下的脚本驱动。

视觉适配器实现 `@implements(PIXEL_TO_BASE_XYZ)` 的 `pixel_to_base_xyz(u, v, depth_m)` 投影动作：eye-in-hand 可转发 `perception/vision.default_pixel_to_base_xyz`（提供 `pose_to_tf` 回调，组合实时 `T_base_flange @ T_flange_cam`，并按标定应用 XY 校正），eye-to-hand 使用 `T_base_cam`。`locate_for_grasp`/`locate_for_place`/`analyze_scene` 已由 `perception/scene3d` 共享实现（`api/defaults` 转发），直接消费标定 `CameraFrame` 与检测器钩子；`search_target`/`approach_for_grasp`/`approach_for_place` 已由 `motion/approach` 共享实现——适配器只提供相机/标定/检测钩子，并在需要机型专属几何时覆写。

## 6. Config 与 Session Builder

路径字段由 Config 的 `path_fields: ClassVar[tuple[str, ...]]` 声明（无路径可省略）。`from_yaml` 使用 `adapters._common.config.load_yaml_config`，与 Runtime、smoke 共用 `parse_config`；解析支持扁平及历史嵌套配置，不因文件暂不存在而退回当前工作目录。

Config 至少提供 `from_dict(data)` 和 `from_yaml(path)`。常用字段按能力选择：

| 分组 | 常用字段 |
|---|---|
| 基本与连接 | `name`、CAN/串口/网络地址、速度、超时 |
| 运动学 | Home 位姿、`tool_offset_mm`、姿态约定 |
| 安全 | `z_min_safe_mm`、XY 边界、`joint_limits` |
| 末端 | 开口、力、吸盘 I/O |
| 相机 | serial、分辨率、FPS、内参/标定路径 |
| 检测 | URL、spawn、模型、阈值 |
| 抓放几何 | `z_correction_mm`、`grasp_z_offset_mm`、`place_z_offset_mm` |

`make_builder()` 签名：

```python
make_builder(
    cfg_cls,
    env_cls,
    api_cls,
    *,
    config_factory=None,
    resource_keys=None,
    api_kwargs_from_cfg=None,
    sidecar_builders=None,
    decorate=None,
)
```

| 参数 | 作用 |
|---|---|
| `cfg_cls` | 带 `from_yaml`/`from_dict` 的 Config 类 |
| `config_factory` | 可选配置工厂，覆盖 cfg_cls 的加载方法；builder 以 `.config_factory` 公开实际工厂 |
| `resource_keys` | 有效 cfg → 设备资源键；builder 公开 `.resource_keys(cfg)`，供 Runtime/官方维护准入使用 |
| `env_cls` | 以 `cfg` 构造的 Env 类 |
| `api_cls` | 以 `env` 和可选 kwargs 构造的 Api 类 |
| `api_kwargs_from_cfg` | `list[str]` 声明式映射或兼容的 `cfg -> dict` 回调 |
| `sidecar_builders` | 每项接收 cfg，返回 context manager、零参工厂或 `None` |
| `decorate` | 最终 Session 装饰回调，普通适配器通常不用 |

声明式字段映射：

| 写法 | 结果 |
|---|---|
| `"z_correction_mm"` | `cfg.z_correction_mm` 传给同名 Api 参数 |
| `"detector.url:detector_service_url"` | 嵌套字段重命名后传入 |

返回的 Builder 支持 `build(cfg)`、`.from_yaml(path)` 和 `.from_dict(data)`。`make_detector_sidecar(cfg_attr="detector")` 读取检测子配置，在 `spawn` 为真时随 Session 启停 GroundingDINO/SAM2 服务。

资源键与实际连接必须使用同一份有效配置。Cruzr 的 `ros_domain_id` 可在 YAML 中显式指定；
省略时在 Config 构建时捕获 `ROS_DOMAIN_ID`（默认 0）。ROS 初始化和命令 worker 使用该值，
已有 ROS context 的域不一致时拒绝连接。sidecar 的退出失败必须传播；若启动失败但已完成回滚，
可提供 `cleanup_report() -> CleanupReport` 确认释放，否则 Session 保留未知清理状态。

## 7. 共享感知与几何模块

| 模块 | 主要接口 | 用途 |
|---|---|---|
| `adapters/_common/builder.py` | `make_builder()`、`make_detector_sidecar()` | Session 工厂和检测 sidecar |
| `adapters/_common/capability_spec.py` | `CAPABILITY_ACTIONS`、`CAPABILITY_DRIVER_MEMBERS` | 能力→动作/Driver 成员映射（验证器与生成器共用） |
| `adapters/_common/safety.py` | `WorkspaceBounds`、`check_flange_z()` | TIP/FLANGE Z 防御 |
| `perception/detector_client.py` | `init_detector()` | HTTP 检测客户端 |
| `perception/detector_sidecar.py` | `detector_subprocess()` | 检测服务生命周期 |
| `perception/scene3d.py` | `locate_for_grasp()`、`locate_for_place()`、`analyze_scene()` | 标定帧→检测→掩膜点云→物体/表面几何（3-D 场景感知） |
| `perception/vision.py` | `detect_and_centroid()`、`apply_xy_correction()`、`default_pixel_to_base_xyz()`、`default_get_grasp_info_simple()` | 检测/校正/eye-in-hand 投影共享函数 |
| `perception/calibration.py` | `load_calibration()` | 版本化手眼标定加载 |
| `motion/approach.py` | `search_target()`、`approach_target_for_grasp()`、`approach_target_for_place()` | 寻靶→对准目标面→收敛到工作位姿（基础接近） |
| `motion/dual_arm.py` | `dual_arm_grasp()`、`dual_arm_place()` | 双臂协同抓/放（含接触力确认） |
| `contracts.py` | `GraspResult`、`ObjectGeometryResult`、`SPATIAL_RELATIONS` 等 | 动作结果类型 + 空间关系集（不归属任何层） |

### `init_detector`

```python
seg_fn = init_detector("http://127.0.0.1:8114")
results = seg_fn(image_ndarray, text_prompt="blue box")
```

结果项包含布尔 `mask`、`[x1,y1,x2,y2]`、score 和 label；服务不可达时返回空结果。

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

成功结果包含 `u`、`v`、`depth_m`、best、mask shape 和 image shape；失败原因包括 `no_detection`、`empty_mask`、`no_valid_depth`。

### `apply_xy_correction`

```python
xyz_final, description = apply_xy_correction(
    xyz_raw=raw_base_xyz,
    xy_transform=calibration.get("xy_transform"),
    xy_correction_mm=calibration.get("xy_correction_mm"),
)
```

`xy_transform` 优先于旧式平移量。常规适配器不直接调用；共享流程统一调用。

## 8. Capability 扩展契约

只有现有字符串无法描述真实硬件功能时才扩展词表。一次扩展必须同步：

1. 在 `env/base.py:KNOWN_CAPABILITIES` 添加字符串；
2. 需要动作时在 `api/actions.py` 添加 `ActionSpec`，在 Api 加 `@implements`（能通用委托的转发 `api/defaults`）；
3. 能统一委托时增加 Env 公共动词，否则由适配器实现；
4. Env 声明能力，Api 实现动作；
5. 更新 `capability_spec.py`、验证器和工具生成测试；
6. 涉及动作安全时增加对应 Rail 检查或明确由控制器负责。

不要为了绕过未知能力错误而随意增加字符串。标记能力可以没有动作，但应有明确消费者。

## 9. 验证器与测试入口

| 入口 | 作用 |
|---|---|
| `scripts/validate_adapter.py --module ...` | 静态检查目录、签名、能力对齐和 Driver 成员 |
| `scripts/smoke_test_adapter.py --module ...` | 向 Env 注入桩驱动（不真实连接硬件），调用所有生成工具并检查可序列化结果 |
| `tests/unit_tests/env/` | Env、能力和安全属性参考测试 |
| `tests/unit_tests/api/` | 动作词表、`@implements` 和 capability 推导 |
| `tests/unit_tests/agent/` | Session、Builder、Rails 和工具装配 |
| `tests/mocks/` | Mock Driver、Env、Api 和场景 |

常见验证结果（验证器实际输出为中文+编号，此处按含义归纳）：

| 现象 | 含义 |
|---|---|
| 未知能力（如 E-04） | Env 字符串不在词表 |
| Api 能力不在 Env.capabilities 中（A-08） | 工具被能力交集过滤 |
| 驱动缺少能力所需方法（如 D-14） | 声明的能力与 Driver Protocol 不一致 |

## 10. 内置适配器实现位置对照

| 关注点 | Piper | SO-101 | Cruzr |
|---|---|---|---|
| 能力与 Env | `adapters/piper/env.py` | `adapters/so101/env.py` | `adapters/cruzr/env.py` |
| 厂商驱动 | `adapters/piper/lowlevel.py` | `adapters/so101/lowlevel.py` | `adapters/cruzr/lowlevel.py` |
| 几何 | `adapters/piper/geometry.py` | `adapters/so101/geometry.py` | `adapters/cruzr/geometry.py` |
| 标定 | `adapters/piper/_calibration.py` + `calibration/adapters/piper.py` | `adapters/so101/_calibration.py` + `calibration/adapters/so101.py` | `adapters/cruzr/_calibration.py`（仅相机标定加载） |
| Api | `adapters/piper/api.py` | `adapters/so101/api.py` | `adapters/cruzr/api.py` |
| Config | `adapters/piper/config.py` | `adapters/so101/config.py` | `adapters/cruzr/config.py` |
| Session | `adapters/piper/session.py` | `adapters/so101/session.py` | `adapters/cruzr/session.py` |

Piper 的 30° 倾斜工具、历史嵌套 YAML 和临时 Z 校正都属于机型特例；新适配器只在硬件确有相同约束时复用。Cruzr 的双臂 + 升降可达性（`Reachability` 覆写）、头/腰双路相机、ROS 2 驱动也属于机型特例，通用机制（`scene3d`/`approach`/`dual_arm` 共享实现）不要复制到别处。
