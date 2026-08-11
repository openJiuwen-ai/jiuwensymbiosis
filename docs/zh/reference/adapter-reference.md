# 机器人适配器参考

> 类别：Reference。本页集中查询适配器的稳定契约、参数和实现位置。

第一次实现适配器请阅读[构建第一个机器人适配器](../tutorial/02-build-first-adapter.md)；从 Mock 接入厂商 SDK 和真机请阅读[移植机器人硬件适配器](../how-to/port-hardware-adapter.md)。本页不提供顺序操作流程。

## 1. 六个适配器文件

```text
jiuwensymbiosis/adapters/<name>/
├── config.py
├── lowlevel.py
├── env.py
├── api.py
├── session.py
└── config_template.yaml
```

| 文件 | 稳定职责 | 不应包含 |
|---|---|---|
| `config.py` | 配置 dataclass、`from_dict()`、`from_yaml()` | 硬件连接和任务文本 |
| `lowlevel.py` | 厂商 SDK、CAN、串口、Socket、相机和执行器 I/O | Agent、Rail、`@robot_tool` |
| `env.py` | 能力、生命周期、观测、安全属性、Driver 包装 | 提示词和厂商流程编排 |
| `api.py` | Mixin 组合、工具语义、机型几何、RAW 视觉投影 | 重复的检测/校正流水线 |
| `session.py` | Config/Env/Api、sidecar 和附加对象接线 | 大段业务实现 |
| `config_template.yaml` | 可部署起点及字段注释 | 用户任务和秘密凭据 |

模板位于 `templates/xxx_adapter/`。

## 2. Capability、Mixin 与工具

能力词表由 `jiuwensymbiosis.env.base.KNOWN_CAPABILITIES` 定义：

| Capability | Mixin 或角色 | 继承的主要工具或行为 |
|---|---|---|
| `motion.cartesian` | `MotionMixin` | `home()`、`get_pose()`、`get_home_pose()`、`goto_xyzr()` |
| `motion.joint` | `JointMotionMixin` | `move_joint()` |
| `motion.servo` | 适配器专属实时工具 | 非阻塞伺服位姿指令 |
| `grasp.suction` | `SuctionMixin` | `activate_suction()`、`deactivate_suction()` |
| `grasp.parallel` | `ParallelGripperMixin` | `open_gripper()`、`close_gripper()` |
| `vision.detection` | `VisionMixin` | `get_image()`、`get_grasp_info_simple()`、`pixel_to_base_xyz()` |
| `vision.camera` | 标记能力 | 表示可获取 RGB，不直接生成工具 |
| `vision.depth` | 标记能力 | 表示可获取深度，不直接生成工具 |
| `vision.eye_to_hand` | 标记能力 | 表示相机固定在基座或世界坐标系 |
| `sorting.command` | 适配器或专用工具 | 不透明分拣协议 |
| `speech.tts` | 适配器或服务 | 文本转语音 |

Env 手动声明硬件能力；Api 从 Mixin MRO 自动推导软件能力；有效工具能力为二者交集。Env 可声明无 Mixin 的标记能力，Api Mixin 对应能力缺失于 Env 时不会生成工具。

## 3. Driver Protocol 契约

Protocol 定义在 `jiuwensymbiosis/env/protocol.py`。具体 Driver 使用结构化类型，不要求继承这些类。

| Protocol | 对应能力 | 必需成员 |
|---|---|---|
| `RobotDriver` | `motion.cartesian` | `home_pose`、`z_min_safe`、`flange_z_min_safe`、`tool_offset_mm`、`close()`、`home()`、`get_pose()`、`move_to_pose_blocking(pose, ...)` |
| `JointDriver` | `motion.joint` | `get_angles()`、`move_joint_blocking(q, timeout_s=...)` |
| `ServoDriver` | `motion.servo` | `servo_to_pose(pose)` |
| `GripperDriver` | `grasp.parallel` | `set_gripper(on)`、`gripper_state` |
| `SuctionDriver` | `grasp.suction` | `set_suction(on)`、`suction_state`、`suction_di_last` |
| `CameraDriver` | `vision.camera` | `intrinsics`、`grab_frames()` |
| `VisionDriver` | eye-in-hand `vision.detection` | `tf_flange_cam`、`calibration` |

关键语义：

- `get_pose()`/`home_pose` 返回厂商 Pose，可为 SCARA 的 `(x,y,z,r)` 或六轴的 `(x,y,z,rx,ry,rz)`；
- `move_to_pose_blocking()` 接收 FLANGE 目标，必须阻塞至完成或失败；
- `move_joint_blocking()` 的单位由适配器约定，但必须与 `joint_limits` 一致；
- `servo_to_pose()` 非阻塞，显式 `False` 表示本周期未推进，`True` 或兼容的 `None` 表示已接受；
- `grab_frames()` 返回对齐的 `(rgb_uint8, depth_m_float32)` 或 `None`；
- `close()` 必须幂等。

当前具名的 `VisionDriver` Protocol 只覆盖 eye-in-hand 标定。SO-101 等 eye-to-hand 适配器以机型专属的
结构化接口暴露 `tf_base_cam` 和 `calibration`，并声明 `vision.eye_to_hand` 标记能力；目前还没有独立命名的
eye-to-hand Protocol。

多能力驱动可定义组合 Protocol 收紧 `low_level` 类型；`PiperFullDriver` 是
`RobotDriver + JointDriver + CameraDriver + GripperDriver + VisionDriver` 的现有示例。

## 4. BaseRobotEnv 契约

`BaseRobotEnv` 的抽象方法：

| 方法 | 语义 |
|---|---|
| `connect()` | 打开硬件连接，必须幂等 |
| `disconnect()` | 释放硬件，任何状态调用都应安全 |
| `get_observation()` | 最佳努力快照，瞬时传感器缺口不应导致整次调用失败 |

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
| `home_pose` | `None` | `MotionMixin.get_home_pose()` |
| `tool_offset_mm` | `0.0` | TIP↔FLANGE 几何 |

已有默认委托：

| Env 动词 | Driver 目标 |
|---|---|
| `home()` | `driver.home()` |
| `get_flange_pose()` | `driver.get_pose()` |
| `move_to_flange(pose)` | `driver.move_to_pose_blocking(pose)` |
| `move_joint(q)` | `driver.move_joint_blocking(q)` |
| `set_end_effector(True/False)` | 按能力调用 `set_gripper()` 或 `set_suction()` |
| `grab_rgb()` | 默认返回 `get_observation().rgb` |

`reset()` 和软件 `emergency_stop()` 默认无操作。物理急停始终属于硬件安全系统。

## 5. Api、Mixin 与覆写规则

Api 多继承顺序为能力 Mixins 在前、`BaseRobotApi` 在后。内置默认行为覆盖常见运动、关节、抓取、取图和视觉抓放流程。

| 情况 | 处理 |
|---|---|
| 行为与公共语义一致 | 直接继承 Mixin |
| TIP/FLANGE、姿态或字段名不同 | 覆写对应方法体 |
| 只想使用继承的工具描述 | 覆写时无需重新加 `@robot_tool` |
| 需要机型专属描述 | 重新装饰，并显式给出 `tags` |
| 硬件有词表外的新能力 | 先完成能力扩展契约，再新增工具 |

`@robot_tool` 元数据包含名称、描述、输入 JSON Schema、capability 和 tags。工具构建器遍历 MRO，覆写方法可以继承装饰器元数据。

`VisionMixin` 拥有以下共享顺序：

```text
抓帧 → 检测 → mask 质心/中值深度 → RAW 投影 → XY/Z 校正 → 抓放高度 → 结构化结果
```

视觉适配器必须实现 `_project_pixel_to_base_raw(u, v, depth_m)`：eye-in-hand 组合实时 `T_base_flange @ T_flange_cam`，eye-to-hand 使用固定 `T_base_cam`。RAW 方法不得应用校正。`analyze_scene()` 没有统一语义，仅在需要该工具时实现。

## 6. Config 与 Session Builder

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
    api_kwargs_from_cfg=None,
    sidecar_builders=None,
    decorate=None,
)
```

| 参数 | 作用 |
|---|---|
| `cfg_cls` | 带 `from_yaml`/`from_dict` 的 Config 类 |
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

## 7. 共享感知与几何模块

| 模块 | 主要接口 | 用途 |
|---|---|---|
| `adapters/_common/builder.py` | `make_builder()`、`make_detector_sidecar()` | Session 工厂和检测 sidecar |
| `adapters/_common/safety.py` | `WorkspaceBounds`、`check_flange_z()` | TIP/FLANGE Z 防御 |
| `adapters/_common/capability_spec.py` | capability specifications | 静态适配器验证 |
| `perception/detector_client.py` | `init_detector()` | HTTP 检测客户端 |
| `perception/detector_sidecar.py` | `detector_subprocess()` | 检测服务生命周期 |
| `perception/vision.py` | `detect_and_centroid()` | 检测、质心和中值深度 |
| `perception/vision.py` | `build_grasp_result()` | 校正和抓放几何 |
| `perception/vision.py` | `apply_xy_correction()` | 多点或旧式平移校正 |
| `perception/vision.py` | `dump_grasp_debug()` | 可选调试产物 |
| `perception/calibration.py` | `load_calibration()` | 版本化手眼标定加载 |
| `utils/geometry.py` | 针孔与 SE(3) 工具 | 投影和变换组合 |

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

`xy_transform` 优先于旧式平移量。常规适配器不直接调用；`VisionMixin` 在共享投影流程中统一调用。

## 8. Capability 扩展契约

只有现有字符串无法描述真实硬件功能时才扩展词表。一次扩展必须同步：

1. 在 `env/base.py:KNOWN_CAPABILITIES` 添加字符串；
2. 能生成工具时，在 `api/mixins.py` 添加同名 capability 的 Mixin；
3. 能统一委托时增加 Env 公共动词，否则由适配器实现；
4. Env 声明能力，Api 继承 Mixin；
5. 更新 `capability_spec.py`、验证器和工具生成测试；
6. 涉及动作安全时增加对应 Rail 检查或明确由控制器负责。

不要为了绕过未知能力错误而随意增加字符串。标记能力可以没有 Mixin，但应有明确消费者。

## 9. 验证器与测试入口

| 入口 | 作用 |
|---|---|
| `scripts/validate_adapter.py --module ...` | 静态检查目录、签名、能力对齐和 Driver 成员 |
| `scripts/smoke_test_adapter.py --module ...` | 连接 Mock Env，调用所有生成工具并检查可序列化结果 |
| `tests/unit_tests/env/` | Env、能力和安全属性参考测试 |
| `tests/unit_tests/api/` | Mixin、装饰器和 capability 推导 |
| `tests/unit_tests/agent/` | Session、Builder、Rails 和工具装配 |
| `tests/mocks/` | Mock Driver、Env、Api 和场景 |

常见验证结果：

| 现象 | 含义 |
|---|---|
| unknown capability | Env 字符串不在词表 |
| Api capability missing from Env | 工具被能力交集过滤 |
| Driver member missing | 声明的能力与 Driver Protocol 不一致 |
| tool result not serializable | 工具返回了 ndarray、Pose 或其他原生对象 |

## 10. Piper 实现位置对照

| 关注点 | Piper 文件 | 说明 |
|---|---|---|
| 能力与 Env | `adapters/piper/env.py` | capabilities、生命周期、观测和边界 |
| 厂商驱动 | `adapters/piper/lowlevel.py` | CAN/SDK、运动、夹爪和相机 |
| 几何 | `adapters/piper/geometry.py` | Pose、TIP/FLANGE 和投影辅助 |
| 标定 | `adapters/piper/_calibration.py` | 手眼标定矩阵加载 |
| Api | `adapters/piper/api.py` | 倾斜工具几何和 RAW 投影 |
| Config | `adapters/piper/config.py` | PiperConfig、DetectorServerConfig、历史 YAML 兼容 |
| Session | `adapters/piper/session.py` | `make_builder()`、字段映射和检测 sidecar |

Piper 的 30° 倾斜工具、历史嵌套 YAML 和临时 Z 校正都属于机型特例；新适配器只在硬件确有相同约束时复用。
