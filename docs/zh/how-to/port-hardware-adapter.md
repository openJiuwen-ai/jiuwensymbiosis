# 移植机器人硬件适配器

> 类别：How-to。本文说明如何把已经跑通的 Mock 适配器替换为可投产的真实硬件适配器。

第一次接触适配器时，请先完成[构建第一个机器人适配器](../tutorial/02-build-first-adapter.md)。本文不重复“6 个必写 Python 文件 + 1 份 YAML（外加可选标定 wrapper）”的完整示例；Capability、Protocol、Env 属性、`@implements` 和 `make_builder()` 的精确契约统一查阅[机器人适配器参考](../reference/adapter-reference.md)。

## 使用范围与完成标准

这三篇文档的分工如下：

| 你的目标 | 阅读入口 |
|---|---|
| 无硬件完成第一个可运行适配器 | [构建第一个机器人适配器](../tutorial/02-build-first-adapter.md) |
| 把厂商 SDK、相机和末端执行器接入真机 | 本文 |
| 查询接口、字段、参数和 Piper 实现位置 | [机器人适配器参考](../reference/adapter-reference.md) |

生产适配器完成后应满足：

- Env 与 Api 能力对齐，生成工具只覆盖硬件真实能力；
- Driver 的连接、运动、末端和传感接口可重复调用并正确失败；
- TIP、FLANGE、相机和基座坐标系有书面约定并通过实测；
- Config、Session、HTTP 客户端和可选本地检测进程能完整初始化与清理；
- 软件边界、控制器限位和物理急停均已验收；
- 静态验证、Mock 冒烟、单元测试和低速真机验收通过。

建议先复制 Tutorial 中的 Mock 适配器，在每一步只替换一层：

```text
确认能力与坐标 → 替换 Driver → 补齐 Env/Api → 接入视觉 → 配置 Session → 安全与验收
```

## 1. 确认硬件能力与坐标约定

先把厂商手册、控制器实际能力和当前 Mock 适配器做成差异记录。不要因为 Api 有某个动作的 `@implements` 就声明硬件支持该能力。

| 决策 | 必须确认的内容 |
|---|---|
| 运动 | 笛卡尔、关节或实时伺服；阻塞完成语义；位置与角度单位 |
| 末端 | 吸盘或平行夹爪；开闭语义；状态反馈是否可信 |
| 相机 | RGB、深度、帧对齐；eye-in-hand 或 eye-to-hand |
| 坐标 | 厂商返回的是 TIP 还是 FLANGE；姿态角顺序、方向和单位 |
| 安全 | TIP Z 下限、XY 工作空间、关节软限位、控制器硬限位 |

Env 只声明真实存在的能力：

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

Api 用 `@implements(SPEC)` 绑定共享词表里本体支持的动作，公开软件能力：

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

工具按 `api.capabilities ∩ env.capabilities` 生成。完成接线后必须检查：

```python
with session:
    print(session.describe()["effective_capabilities"])
```

在写驱动前明确公共工具语义：`goto_xyzr(x, y, z, r)` 的 XYZ 表示基座坐标系中的工具 TIP；Driver 的 `move_to_pose_blocking(pose)` 接收 FLANGE 目标。带工具偏移或倾斜安装时，转换属于 Api 的机型专属几何，不能藏在模型提示词里。

## 2. 用厂商 SDK 替换 Mock Driver

`lowlevel.py` 只负责把稳定的框架动词翻译为串口、CAN、Socket 或厂商 SDK 调用。Agent 提示词、`@implements` 和 Rail 逻辑都不应进入驱动。

```text
LLM 工具 → Api @implements/覆写 → Env 公开动词 → Driver → 控制器
观测与异常 ←────────────── 同一边界逐层返回 ←──────────
```

运动和末端操作必须经 Env 公开动词访问 Driver；视觉标定、内参和原始帧允许经带 Protocol 类型的 `env.low_level` 受控访问。

### 生产 Driver 要求

- Driver 的 `connect()`/`close()` 必须幂等，错误清理后可再次连接；Env 对外提供 `disconnect()` 并委托
  Driver 的 `close()` 完成底层清理；
- `move_to_pose_blocking()` 必须等到到位或明确失败，不能只等控制器“已接收”；
- 所有坐标系和单位写在方法注释中，深度在采集边界使用米；
- 超时、不可达、通信中断和畸形状态应抛出可定位的异常；
- SDK 非线程安全时，对运动、相机和状态读取做串行化；
- Driver 或控制器保留最后一道工作空间防线，SafetyRail 不能替代硬限位；
- 真实运动从低速开始，并始终保证操作者可触达物理急停。

一个笛卡尔驱动的最小形状如下，能力对应的完整 Protocol 成员见[适配器参考](../reference/adapter-reference.md#3-driver-protocol-契约)：

```python
class MyDriver:
    def connect(self) -> None: ...
    def close(self) -> None: ...
    def get_pose(self): ...
    def home(self) -> None: ...
    def move_to_pose_blocking(self, pose) -> None: ...

    # 按声明能力提供
    def move_joint_blocking(self, q: list[float], *, timeout_s=30.0) -> None: ...
    def set_gripper(self, on: bool) -> None: ...
    def grab_frames(self): ...  # (rgb_uint8, depth_m_float32) 或 None
```

当前 `templates/xxx_adapter` 和 `scripts/new_adapter` 的兼容脚手架仍把底层清理方法命名为 `disconnect()`；生成的
Env 会调用该方法，因此脚手架内部可以运行，但尚不满足正式 `RobotDriver.close()` 生命周期成员。将生成代码用于
生产适配器前，应重命名该方法或补一个幂等的 `close()` 别名，并让 Env 委托给它。

不要一次同时替换所有硬件功能。推荐依次验证连接和状态、Home、单个安全位姿、末端开闭、相机帧，最后才启用 Agent。

## 3. 补齐生产 Env 与机型专属 Api

Env 是 Agent、Tools 和 Rails 唯一依赖的硬件契约。它管理生命周期、观测和安全属性，不重新实现厂商业务逻辑。

### Env 生命周期与观测

连接前保留 Driver 句柄，连接状态独立记录；只有关闭成功后才清空句柄。关闭失败必须传播并保留可重试句柄，不能只记日志后假装释放。构造函数中已经获取设备却无法回滚时，抛出 `HardwareCleanupError`（`agent.lifecycle`），让 Session 和资源准入保留不确定状态。

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

同时从 Config 暴露 `z_min_safe`、`workspace_bounds` 和 `joint_limits`，从 Driver 暴露 `home_pose` 与 `tool_offset_mm`。字段类型和默认委托方法见[Env 契约](../reference/adapter-reference.md#4-baserobotenv-契约)。

### Api 几何与视觉投影

运动、关节、抓取和取图通常直接转发 `api.defaults`。只覆写公共语义与机型不一致的部分。例如垂直工具的 TIP→FLANGE 转换：

```python
@implements(GOTO_XYZR)   # 描述、能力门、参数都来自词表里的那条 spec
def goto_xyzr(self, x: float, y: float, z: float, r: float | None = None) -> None:
    current = self.env.get_flange_pose()
    flange = MyPose(x, y, z + self.env.tool_offset_mm,
                    current.rx, current.ry, current.rz if r is None else r)
    self.env.move_to_flange(flange)
```

这个示例只适用于沿基座 Z 轴的标量偏移。倾斜工具必须使用完整变换，不能照搬 `z + offset`。

视觉适配器用 `@implements(PIXEL_TO_BASE_XYZ)` 实现 `pixel_to_base_xyz()` 投影动作。以 eye-to-hand 为例：

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

eye-in-hand 可把该动作转发给 `perception/vision.default_pixel_to_base_xyz()`，并提供把实时法兰位姿转换成 `T_base_flange` 的 `pose_to_tf` 回调；共享实现再组合 `T_base_cam = T_base_flange(live) @ T_flange_cam`。`get_image()` 可由 `api.defaults` 转发；`pixel_to_base_xyz()` 和 `get_grasp_info_simple()` 则是适配器显式绑定的 `@implements` 动作，eye-in-hand 本体可复用 `perception/vision` 中的对应共享实现。面向场景的 `locate_for_grasp()`、`locate_for_place()` 和 `analyze_scene()` 由 `perception/scene3d` 处理标定帧、检测与三维几何，适配器只在确有机型专属语义时覆写。

## 4. 接入检测、标定与校正

只有声明 `vision.detection` 的适配器才需要本节。先在不启动 Agent 的情况下依次验证：

1. `grab_frames()` 返回对齐的 RGB `uint8` 和深度米 `float32`；
2. 相机内参可用，图像尺寸与内参匹配；
3. 检测服务对目标文本返回 mask、box、score 和 label；
4. 手眼变换方向正确，已区分 eye-in-hand 与 eye-to-hand；
5. RAW 投影在多个工作区位置误差方向一致；
6. 最后才配置 XY 校正、`z_correction_mm`、`grasp_z_offset_mm` 和 `place_z_offset_mm`。

不要用大幅运行时偏移掩盖错误的深度单位、变换方向或相机松动。应先重新标定，再增加小范围校正。手眼标定操作见[手眼标定指南](calibrate-hand-eye.md)，共享检测和校正函数见[适配器参考](../reference/adapter-reference.md#7-共享感知与几何模块)。

GroundingDINO/SAM2 对外提供 **HTTP 推理服务**。`detector.mode: local` 由 Session
启动并管理本地托管子进程（sidecar）；`remote` 连接外部管理的服务，地址也可以是
`localhost`。两种模式使用同一 HTTP 客户端和结果契约。Session 退出时关闭自己的客户端，
仅停止它启动的本地子进程，不停止外部管理的服务。未配置检测时默认 `disabled`。

客户端的连接、超时、协议和结果过期错误通过 `InferenceServiceError` 的
`inference_*` 错误码区分；共享感知动作将它们报告为
`{"ok": false, "reason": "detector_unavailable", "error_code": "inference_timeout"}` 等失败结果。
只有推理成功但没有目标时才属于 `no_detection`，不能把服务异常转换为空结果。
部署与迁移操作见[远程 HTTP 视觉与语音服务](remote-inference.md)。

## 5. 配置部署 YAML 与 Session

硬件配置只描述机器人、服务、标定和 Agent 开关，不存放用户任务。新适配器优先使用平铺、语义明确的字段；历史嵌套 YAML 兼容只在确有迁移需求时实现。

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
  mode: remote
  endpoint:
    url: http://gpu-server:8114
    request_timeout_s: 30
```

必填连接字段、单位和危险默认值必须在 `config_template.yaml` 中标注。相对标定路径应相对于 YAML 文件解析，而不是依赖调用者当前目录。

Config 的 `detector` 字段使用 `perception.config.DetectorConfig`，默认工厂为
`DetectorConfig`；`from_dict(data)` 调用 `parse_detector_config(data)` 解析完整运行配置。
需要本地托管进程时改用 `detector: {mode: local, local: {device: cuda, port: 8114}}`，
只关闭推理时使用 `detector: {mode: disabled}`。

Config 用 `path_fields: ClassVar[tuple[str, ...]]` 声明本体的路径字段（例如 `("calib_path", "urdf_path")`），`from_yaml` 委托给 `adapters._common.config.load_yaml_config(cls, path)`。Runtime 配置快照和 smoke 复用同一`parse_config(factory, data, source_dir=...)` 入口，返回规范化副本和仅解析一次的有效配置。路径会展开环境变量与 `~`，即使目标尚不存在也相对来源目录解析；无来源的 `from_dict(data)` 保持原语义。新增硬件路径字段只改适配器声明，不要在 Runtime 或验证脚本里加字段名单。

### 使用公共 Builder 接线

```python
from jiuwensymbiosis.adapters._common.builder import make_builder, make_detector_sidecar

build_my_session = make_builder(
    MyConfig,
    MyEnv,
    MyApi,
    api_kwargs_from_cfg=[
        "z_correction_mm",
        "grasp_z_offset_mm",
        "place_z_offset_mm",
    ],
    sidecar_builders=[make_detector_sidecar()],
    managed_detector=True,
    resource_keys=lambda cfg: (f"can:{cfg.can_port}",),
)
```

`api_kwargs_from_cfg` 的裸字段同名传递，`cfg.path:api_name` 支持重命名，配置路径可用点号访问嵌套对象。只有声明式映射无法表达转换时才使用旧式回调。普通适配器不需要 `decorate`；它只用于向 Session 注入无法由 Config、Api 参数或已有资源管理机制表达的附加对象。

`managed_detector=True` 为每个 Session 创建一个客户端，并通过 `detector_client` 关键字
传入 `MyApi.__init__`；Api 接收并保存这个可调用对象，感知钩子复用它，不自行按 URL
另建客户端或关闭它。客户端生命周期归 Session。`make_detector_sidecar()` 只在
`mode: local` 时启动进程；`include_sidecars=False` 的维护会话只注册客户端清理，
不启动模型、不解析推理凭据、不做网络检查。只有显式发起推理时才按需打开客户端。

`resource_keys(cfg)` 从实际有效配置声明设备身份；示例为 CAN 接口，串口应规范化到真实设备路径，ROS 应包含域和命令端点。同一设备的配置别名必须产生相同键。公共准入另补相机、本地托管检测进程和可选 `physical_device_id`，后者不能覆盖派生键。外部 HTTP 推理服务可共享，不占独占设备锁；本体的 HTTP 控制端点仍按物理设备准入。builder 公开 `.config_factory` 与 `.resource_keys` 供 Runtime 使用，不在 builder 内重复加锁；`with session` 本身不负责跨进程准入，应用任务应使用 Runtime，官方维护入口使用 admitted_session。

模板的基础和进阶 wiring 都包含 `resource_keys`。`validate_adapter.py` 的 S-17 检查工厂和设备键回调接口；`smoke_test_adapter.py` 在构建 stub session 前用实际配置验证键，设备键为空且未提供 `physical_device_id` 时直接报错，不连接硬件。接口检查不验证物理身份，仍需确认同一设备的别名生成相同键。

交互式 `new_adapter` 生成器也写入 `adapter:` 和设备键回调。CAN/串口/TCP 使用生成配置中的端点，USB 需要序列号；ROS/custom 仍为待补充的连接模板，必须实现实际资源键或提供顶层 `physical_device_id` 后才能经 Runtime 启动，不自动猜测 ROS 域。

验证三种构造方式，并始终使用上下文管理生命周期：

```python
# 从 Config、YAML 或字典构建
session = build_my_session(MyConfig())
session = build_my_session.from_yaml("configs/my_robot/default.yaml")
session = build_my_session.from_dict({"name": "test", "can_port": "can0"})

with session:
    print(session.describe())
```

## 6. 配置软件与硬件安全边界

SafetyRail 在下发前检查笛卡尔 Z、XY 工作空间和关节软限位，并自动解包 `RobotControlTool` 的 `action/params`。被拒绝的调用抛出带修正信息的 `ValueError`，且不得触发任何硬件命令。

### 边界配置与责任分层

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

- `z_min_safe` 使用 TIP 坐标；Driver 的 FLANGE 下限应包含工具偏移；
- `workspace_bounds` 顺序为 `(xmin, ymin, xmax, ymax)`；
- `joint_limits` 的键顺序必须与 `move_joint(q)` 索引顺序一致，单位也必须一致；
- 对每个边界测试“略低、等于、略高”三个值；
- SafetyRail 是软件预检，Driver/控制器仍需硬限位；
- 软件 `emergency_stop()` 不能替代物理急停、安全围栏和操作规程。

## 7. 自动验证与真机验收

先运行静态结构检查和生成工具冒烟测试：

```bash
python scripts/validate_adapter.py --module jiuwensymbiosis.adapters.my_robot
python scripts/smoke_test_adapter.py --module jiuwensymbiosis.adapters.my_robot
```

### 测试与分阶段验收

单元测试至少覆盖：

- 能力交集和生成工具元数据；
- 重复连接、重复断开和连接失败清理；
- `RobotObservation` 字段、单位和缺失传感器；
- TIP/FLANGE、相机/基座以及倾斜工具变换；
- Z、XY、关节边界和非有限输入；
- 吸盘或夹爪每个能力分支；
- 标定缺失、畸形和有效三种情况；
- HTTP 客户端清理、可选本地进程启停与 Session 异常退出；
- 每个工具返回值可 JSON 序列化。

真机按风险递增验收：

1. 不使能动力时检查配置、设备枚举和相机；
2. 低速连接、读取状态并断开；
3. 验证 Home 和单个工作区中心位姿；
4. 验证边界拒绝，确认被拒绝时控制器未收到命令；
5. 验证末端释放、恢复和异常断开；
6. 在多个工作区位置验证视觉投影；
7. 最后才运行完整 Agent 任务。

验收期间必须有操作者和可触达的物理急停。记录机器人型号、固件、SDK、工具偏移、标定文件和测试结果，作为部署配置的一部分。

## 8. 常见故障处理

| 现象 | 优先检查 | 处理 |
|---|---|---|
| `unknown capabilities` | 拼写和 `KNOWN_CAPABILITIES` | 使用现有词汇；确需扩展时同步词表、动作、Env、验证器和测试 |
| 预期工具未生成 | Env 能力与 Api 动作 | 查看 `effective_capabilities` 和验证器 |
| 重复连接失败 | `connect()` 幂等性 | Driver 成功连接后再发布 `low_level`，失败路径彻底清理 |
| `no_detection` | 成功推理后的图像、提示词和阈值 | 核对目标是否可见，不在适配器中复制共享流程 |
| `detector_unavailable` / `inference_*` | 检测模式、HTTP 地址、服务状态、超时和结果时效 | 按错误码排查；不转为无目标结果，不自动启用本地模型 |
| 投影存在固定或方向性偏差 | 深度单位、内参、变换方向、安装方式 | 先重新标定；确认 RAW 接缝未重复应用校正 |
| TIP/FLANGE Z 混淆 | 公共工具语义与工具偏移 | 对比 Api 目标和 Driver 最终命令；倾斜工具使用完整变换 |
| SKILL.md 未加载 | Agent 的 `enable_skill` 和资源路径 | 确认 `RobotControlTool` 已装配；这不是硬件能力问题 |

若问题属于接口字段或参数含义，转到[机器人适配器参考](../reference/adapter-reference.md)；若 Mock 示例本身尚未跑通，返回[构建第一个机器人适配器](../tutorial/02-build-first-adapter.md)，不要带着框架集成问题进入真机调试。
