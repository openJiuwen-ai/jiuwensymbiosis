# JiuwenSymbiosis 硬件移植指南

将任意机械臂、末端执行器或传感器接入 JiuwenSymbiosis 框架的完整指南。

---

## 0. 前置知识

### 框架分层架构

```
┌──────────────────────────────────────────────────┐
│  Agent Layer      │  build_robot_agent()         │  一键构建 LLM 智能体
│                   │  RobotSession                │  管理硬件生命周期 + 子进程
│                   │  RobotAgentConfig            │  模型、模式、安全开关
├──────────────────────────────────────────────────┤
│  Safety Rails     │  SafetyRail                  │  运动前边界拦截
│                   │  RecoveryRail                │  异常自动回零
│                   │  VisualFeedbackRail          │  动作后视觉验证
├──────────────────────────────────────────────────┤
│  Tool Layer       │  build_robot_tools(api)      │  每个 @robot_tool → 一个 LLM 工具
│                   │  RobotControlTool(api)       │  单一入口 action/params 分发
│                   │  InProcessCodeTool           │  进程内 Python 代码执行
├──────────────────────────────────────────────────┤
│  Skill Layer      │  visual_pick/SKILL.md        │  预置操作流程文档
│                   │  visual_place/SKILL.md        │  SkillUseRail 自动加载
│                   │  slot_pick/SKILL.md           │
├──────────────────────────────────────────────────┤
│  API Layer        │  MotionMixin / VisionMixin   │  能力声明 + @robot_tool 方法
│  (Capability      │  SuctionMixin / etc.         │  运动/抓取/取图带默认委托
│   Mixins)         │  BaseRobotApi                │  持有 env 引用
├──────────────────────────────────────────────────┤
│  Env Layer        │  BaseRobotEnv                │  硬件契约面（唯一）
│  (Hardware        │  connect/disconnect/observe  │  能力声明 (env.capabilities)
│   Abstraction)    │  home/move_to_flange/...     │  运动/末端动词（默认委托驱动）
│                   │  home_pose/tool_offset_mm    │  机器人常量属性
│                   │  grab_rgb()                  │  单帧图像（默认走 get_observation）
│                   │  z_min_safe/workspace_bounds │  安全契约属性
│                   │  low_level: RobotDriver      │  受控穿透点（视觉标定/厂商特有）
├──────────────────────────────────────────────────┤
│  Hardware Layer   │  XxxDriver (lowlevel.py)     │  实现 RobotDriver Protocol
│  (Your Code)      │  串口/CAN/Socket 等          │  适配器开发者主要工作
└──────────────────────────────────────────────────┘
```

> Env 是 Agent/Rails/Tools 与硬件之间的**唯一契约面**：
> - 运动/末端经 Env 动词（`home`/`move_to_flange`/`move_joint`/`get_flange_pose`/`set_end_effector`）
> - 机器人常量经 Env 属性（`home_pose`/`tool_offset_mm`）
> - 安全边界经 Env 属性（`z_min_safe`/`workspace_bounds`）
> - 单帧图像经 Env 方法（`grab_rgb()`，默认委托 `get_observation().rgb`）
> - 视觉标定数据经 `env.low_level`（`RobotDriver` + 子 Protocol 类型约束的受控穿透）
>
> 上层经 Env 公开 API 访问硬件，不 `getattr` 私有驱动。`set_end_effector` 基于声明能力（`grasp.parallel`/`grasp.suction`）做确定性分发。

### 核心概念速览

| 概念 | 定义 | 谁定义 |
|------|------|--------|
| **Capability** | 硬件能力的命名字符串，如 `"motion.cartesian"` | `env/base.py:KNOWN_CAPABILITIES` |
| **Mixin** | 声明一个 capability 并提供 `@robot_tool` 方法的类（运动/抓取/取图带默认委托实现，高层视觉为抽象） | `api/mixins.py` |
| **Env** | 硬件驱动包装器，实现 `connect/disconnect/get_observation` | 适配器开发者 |
| **Api** | 继承 Mixin + BaseRobotApi，覆写带专属几何的方法 + 实现高层视觉 | 适配器开发者 |
| **Config** | hardware 参数的 dataclass，含 `from_yaml/from_dict` | 适配器开发者 |
| **Session** | 将 Env + Api + 子进程 打包为生命周期单元 | `make_builder()` 自动生成 |
| **Sidecar** | 随 Session 启停的子进程（如视觉检测服务器） | `_common/detector_sidecar.py` |

### 约定

- Env 的 `capabilities` 是**手动声明**的 frozenset
- Api 的 `capabilities` 是**从 MRO 自动推导**的（遍历所有 Mixin 父类的 `capability` 属性）
- 工具按 **`api.capabilities ∩ env.capabilities`** 门控：Api 有而 Env 无的能力，其工具**运行时不会暴露给 LLM**（`build_robot_tools` 强制交集，`validate_adapter` 的 A-08 会报 ERROR）。`session.describe()` 的 `effective_capabilities` 即此交集。Env 有而 Api 无的标记能力（如 `vision.camera`）不影响运行。

---

## 1. 新手速览：5 分钟跑通最小适配器

```bash
# 1. 复制模板
cp -r templates/xxx_adapter/ jiuwensymbiosis/adapters/my_robot/

# 2. 修改文件
#  - config.py: 填写硬件连接参数
#  - lowlevel.py: 实现硬件通信（或先写 Mock）
#  - env.py: 声明 capabilities + connect/disconnect/observe
#  - api.py: 选择 Mixin 组合，覆写带专属几何的方法
#  - session.py: 无需修改（make_builder 已封装；声明式 api_kwargs_from_cfg + make_detector_sidecar）

# 3. 验证（静态结构 + 运行时冒烟）
python scripts/validate_adapter.py --module jiuwensymbiosis.adapters.my_robot
python scripts/smoke_test_adapter.py --module jiuwensymbiosis.adapters.my_robot

# 4. 测试运行
python -c "
from jiuwensymbiosis.adapters.my_robot import build_my_robot_session
session = build_my_robot_session.from_yaml('configs/my_robot/default.yaml')
with session:
    print(session.describe())
"
```

---

## 2. 六步流程总览

```
步骤1        步骤2          步骤3           步骤4           步骤5         步骤6
定义能力  →  编写驱动  →  实现 Env  →  实现 Api  →  创建 Config  →  组装 Session
  │           │            │             │              │             │
  ▼           ▼            ▼             ▼              ▼             ▼
选择         实现         包装硬件      覆写专属方法     YAML配置      make_builder()
capability   RobotDriver  实现观测      返回 ok/error   必填/选填     一站接线
字符串       Protocol     暴露安全属性   调 env 动词     from_yaml    验证可用
```

> 主要工作量在**步骤 2（驱动）**：硬件通信、运动、传感全在这里实现。
> Env（步骤 3）通常只写 `connect/disconnect/get_observation` + 暴露 `low_level` 和安全属性——
> 运动/末端动词由 `BaseRobotEnv` 默认委托给驱动。Api（步骤 4）的运动/抓取/取图方法由 Mixin
> 默认委托，只需覆写带专属几何的方法（`get_pose`/`goto_xyzr`）并实现高层视觉——eye-in-hand
> 视觉可委托 `adapters/_common/vision.default_get_grasp_info_simple` /
> `default_pixel_to_base_xyz`，只补一个 `pose_to_tf` 回调与检测器 `seg_fn`。

每步详细说明见后续章节。

---

## 3. 步骤 1：定义硬件能力

### 3.1 对照 KNOWN_CAPABILITIES

在 `jiuwensymbiosis/env/base.py:39-51` 定义了 9 个能力字符串：

```python
KNOWN_CAPABILITIES = frozenset({
    "motion.cartesian",   # 笛卡尔空间运动 (XYZ/R 末端指令)
    "motion.joint",       # 关节空间运动
    "grasp.suction",      # 吸盘 开/关
    "grasp.parallel",     # 平行夹爪 开/关
    "vision.camera",      # 原始图像流
    "vision.depth",       # 深度流
    "vision.detection",   # 高层物体检测 (自然语言→3D 抓取位姿)
    "sorting.command",    # 不透明分拣协议 (无笛卡尔运动)
    "speech.tts",         # 文本转语音
})
```

请勾选你的硬件支持的能力：

- [ ] `motion.cartesian`  — 是否支持末端笛卡尔定位？
- [ ] `motion.joint`      — 是否支持关节角指令？
- [ ] `grasp.suction`     — 末端是否为吸盘？
- [ ] `grasp.parallel`    — 末端是否为平行夹爪？
- [ ] `vision.camera`     — 是否有相机可获取 RGB 图像？
- [ ] `vision.depth`      — 是否有深度传感器？
- [ ] `vision.detection`  — 是否需要自然语言目标检测？(需要部署检测服务)
- [ ] `sorting.command`   — 是否为专用分拣控制器？(极少使用)
- [ ] `speech.tts`        — 是否需要语音输出？

### 3.2 Capability ↔ Mixin 速查表

每个 Mixin 方法都带 `@robot_tool` 元数据。**运动 / 关节 / 抓取 / `get_home_pose` / `get_image`** 已提供委托默认实现，
组合 Mixin 即可调用；只有带专属几何或需细粒度控制时才覆写。`get_home_pose` 委托到 `env.home_pose`，
`get_image` 委托到 `env.grab_rgb()`（默认走 `get_observation().rgb`）。**高层视觉**方法没有通用默认（依赖检测服务
+ 手眼标定），由适配器实现。

| Capability 字符串 | 对应 Mixin 类 | 默认委托（继承即可） | 通常需覆写 / 必须实现 |
|---|---|---|---|
| `motion.cartesian` | `MotionMixin` | `home()`；`get_pose()` / `get_home_pose()` / `goto_xyzr()` 提供"顶视、tip==flange"默认 | 有工具偏移或倾斜几何时覆写 `get_pose()` / `goto_xyzr()` |
| `motion.joint` | `JointMotionMixin` | `move_joint()` | — |
| `grasp.suction` | `SuctionMixin` | `activate_suction()`, `deactivate_suction()` | — |
| `grasp.parallel` | `ParallelGripperMixin` | `open_gripper()`, `close_gripper()` | 需宽度/力控时覆写 |
| `vision.detection` | `VisionMixin` | `get_image()` | **必须实现** `get_grasp_info_simple()` / `pixel_to_base_xyz()` / `analyze_scene()` |
| `vision.camera` | ❌ 无对应 Mixin | 无（纯标记能力） | — |
| `vision.depth` | ❌ 无对应 Mixin | 无（纯标记能力） | — |
| `sorting.command` | ❌ 无对应 Mixin | 无（纯标记能力） | — |
| `speech.tts` | ❌ 无对应 Mixin | 无（纯标记能力） | — |

> **注意**：`vision.camera` / `vision.depth` 等无 Mixin 的能力是"标记能力"，它们告诉框架和 SafetyRail "硬件具备此传感器"，但不会生成 LLM 可调用的工具。这些标记能力通过在 `Env.capabilities` 中声明起作用。

### 3.3 声明规则

**Env.capabilities（手动声明）**：在 `env.py` 中声明硬件实际具备的能力：

```python
class MyEnv(BaseRobotEnv):
    capabilities = frozenset({
        "motion.cartesian",    # 支持笛卡尔运动
        "grasp.parallel",      # 使用平行夹爪
        "vision.camera",       # 有相机（标记能力）
        "vision.detection",    # 有检测能力
    })
```

**Api.capabilities（MRO 自动推导）**：通过多继承决定，无需手动声明：

```python
class MyApi(MotionMixin, ParallelGripperMixin, VisionMixin, BaseRobotApi):
    # capabilities 自动 = {"motion.cartesian", "grasp.parallel", "vision.detection"}
    ...
```

### 3.4 能力对齐检查清单

| 检查项 | 说明 |
|--------|------|
| ✅ Env.capabilities **包含** Api 所有 Mixin 的 capability | 否则 Tool Builder 不会生成对应工具 |
| ✅ Env.capabilities ⊆ KNOWN_CAPABILITIES | 否则 `__init_subclass__` 抛 ValueError |
| ⚠️ Env 声明了无 Mixin 的标记能力 | 正常现象，不影响运行 |
| ❌ Env 缺少 Api Mixin 的能力 | **严重**：工具不会生成 |

---

## 4. 步骤 2：编写低层驱动

### 4.1 驱动定位

驱动文件应放在 `jiuwensymbiosis/adapters/<your_robot>/lowlevel.py`，负责与真实硬件的通信（串口、CAN 总线、Socket 等）。**运动、抓取、传感这些"功能"最终都在驱动里实现**——上层的 Env 动词与 Api 工具只是把调用转发到这里。

驱动是一个**不受框架约束**的普通 Python 类——你只需要给它提供满足 `RobotDriver` Protocol 的接口供 Env 层调用。

### 4.2 Api → Env → 驱动 调用链

```
┌──────────────┐  运动/末端   ┌──────────────┐       ┌──────────────┐
│   Api 层     │──env 动词──→│   Env 层     │───→──│  驱动层       │
│ (Mixin方法)  │             │ (包装驱动)   │       │ (硬件通信)    │
│ self.env     │  视觉/标定   │ home()/...   │       │ 串口/CAN/    │
│  .home()...  │──low_level─→│ low_level    │       │ Socket       │
└──────────────┘             └──────────────┘       └──────────────┘
```

**关键约定**：
- `env.connect()` 实例化 `self.low_level = XxxDriver(...)`（或在构造里直接连）
- `env.disconnect()` 关闭驱动后将其置 `None`
- Api 的**运动/末端**经 Env 动词（`self.env.home()`/`move_to_flange()`/
  `set_end_effector()`/`move_joint()`/`get_flange_pose()`），不直接碰 `low_level`
- Api 的**机器人常量**经 Env 属性（`self.env.home_pose`/`self.env.tool_offset_mm`），
  不直接碰 `low_level`
- Api 的**视觉/标定数据**（`grab_frames()`/`tf_flange_cam`/`calibration`/`intrinsics`）
  经 `self.env.low_level` 访问——这是受控穿透，类型由 `RobotDriver`+子 Protocol 约束；
  访问前 Env 的 `_require_driver()` 已统一做未连接检查

### 4.3 驱动接口：`RobotDriver` Protocol

驱动**不强制继承**任何基类，但应满足 `jiuwensymbiosis/adapters/_common/protocol.py`
里的结构化 Protocol（`typing.Protocol`，按能力拆分）。`validate_adapter` 的 D-14
用 `isinstance(driver, RobotDriver)` 校验，缺方法会报 ERROR。

| Protocol | 何时需要 | 成员 |
|---|---|---|
| `RobotDriver` | `motion.cartesian` | `home() / get_pose() / move_to_pose_blocking(*) / close()`；属性 `home_pose / z_min_safe / flange_z_min_safe / tool_offset_mm` |
| `JointDriver` | `motion.joint` | `get_angles() / move_joint_blocking(q)` |
| `GripperDriver` | `grasp.parallel` | `set_gripper(on)`；属性 `gripper_state` |
| `SuctionDriver` | `grasp.suction` | `set_suction(on)`；属性 `suction_state / suction_di_last` |
| `CameraDriver` | `vision.camera` | `grab_frames()`；属性 `intrinsics` |
| `VisionDriver` | `vision.detection` | 属性 `tf_flange_cam / calibration` |

> 生命周期由 Env 的 `connect`/`disconnect` 驱动：驱动用 `connect()/disconnect()`
> （如模板/Mock）或 `close()`（如 Piper）皆可，Env 内部调用对应方法即可。
> `Env.set_end_effector(engaged)` 基于 `env.capabilities` 做确定性分发：
> `grasp.parallel` → `driver.set_gripper()`，`grasp.suction` → `driver.set_suction()`，
> 无 capability 时 raise `NotImplementedError`。下面是各成员的语义参考：

```python
class XxxDriver:
    """硬件通信驱动 — 你可以自由组织内部实现，满足上表 Protocol 即可。"""

    # ===================== 生命周期 [必填] =====================

    def connect(self) -> None:
        """打开硬件连接。必须幂等（重复调用不会出错）。"""
        ...

    def disconnect(self) -> None:
        """释放硬件资源。必须幂等，且在未连接时调用也不应报错。"""
        ...

    # ===================== 运动 [必填-仅 motion.*] =====================

    def get_pose(self) -> Any:
        """获取当前末端位姿。
        建议返回 namedtuple/SimpleNamespace，包含 x, y, z, rx, ry, rz 字段。
        对于 SCARA（4-DOF），rx/ry 可固定为 0 值。
        """
        ...

    def home(self) -> None:
        """执行回零/回原点动作。阻塞式。"""
        ...

    def move_to_pose_blocking(self, pose: Any) -> None:
        """笛卡尔阻塞式运动到目标位姿（FLANGE 坐标系）。
        pose 对象包含 x, y, z, rx, ry, rz 字段（由 Api 层构造）。
        """
        ...

    def move_joint_blocking(self, q: list[float]) -> None:
        """关节空间阻塞式运动。[选填-仅 motion.joint]"""
        ...

    # ===================== 末端执行器 [必填-仅 grasp.*] =====================

    def set_gripper(self, on: bool) -> None:
        """设定夹爪/吸盘状态。
        on=True  → 闭合/吸附
        on=False → 打开/释放
        [选填-仅 grasp.parallel 或 grasp.suction，吸盘用 set_suction]
        """
        ...

    # ===================== 传感器 [选填-仅 vision.*] =====================

    def grab_frames(self) -> tuple | None:
        """抓取一帧 RGB + 深度。
        Returns:
            (rgb: HxWx3 uint8 ndarray, depth: HxW float32 ndarray)
            或 None（如果相机不可用）
        [选填-仅 vision.camera]
        """
        ...

    # ===================== 其他属性 =====================

    # home 位姿对象（与 get_pose() 返回同类型）
    home_pose: Any = None

    # 工具偏移量 (mm)，flange → tip
    tool_offset_mm: float = 0.0
```

### 4.4 Mock 驱动模式（无硬件时）

在没有真实硬件时，可以先编写一个 Mock 驱动用于验证框架集成。

参考 `jiuwensymbiosis/env/mock.py:MockArmEnv`（~90 行）——它维护内存中的位姿状态，`connect()` 只设标志，`move()` 记录调用日志。

```python
class MockXxxDriver:
    """无硬件 Mock 驱动，供测试和框架验证使用。"""

    def __init__(self):
        self._pose = {"x": 200.0, "y": 0.0, "z": 250.0, "rx": 0.0, "ry": 90.0, "rz": 0.0}
        self.home_pose = SimpleNamespace(**self._pose)
        self.tool_offset_mm = 0.0
        self._connected = False

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_pose(self) -> Any:
        return SimpleNamespace(**self._pose)

    def home(self) -> None:
        self._pose = {"x": 200.0, "y": 0.0, "z": 250.0, "rx": 0.0, "ry": 90.0, "rz": 0.0}

    def move_to_pose_blocking(self, pose) -> None:
        self._pose["x"] = pose.x
        self._pose["y"] = pose.y
        self._pose["z"] = pose.z
        self._pose["rx"] = getattr(pose, "rx", 0.0)
        self._pose["ry"] = getattr(pose, "ry", 90.0)
        self._pose["rz"] = getattr(pose, "rz", 0.0)
```

Mock 驱动开发完成后，即可编写 `env.py`（步骤 3）和 `api.py`（步骤 4），然后用验证工具检查框架兼容性（附录 C）。

---

## 5. 步骤 3：实现 Env 环境类

### 5.1 继承 BaseRobotEnv

```python
from jiuwensymbiosis.env.base import BaseRobotEnv, RobotObservation

class MyEnv(BaseRobotEnv):
    capabilities = frozenset({
        "motion.cartesian",
        "grasp.parallel",
    })
    name = "my_robot"
```

`capabilities` 必须 ⊆ `KNOWN_CAPABILITIES`，否则 `__init_subclass__` 会抛 `ValueError`。

### 5.2 实现生命周期方法

Env 的核心职责是把驱动包装进 `connect / disconnect / get_observation` 三个抽象方法：

```python
class MyEnv(BaseRobotEnv):
    def __init__(self, cfg: MyConfig):
        self._cfg = cfg
        self.low_level: Optional[MyDriver] = None  # ← 委托目标

    def connect(self) -> None:
        """打开硬件连接。必须幂等。"""
        if self.low_level is not None:
            return
        self.low_level = MyDriver(port=self._cfg.can_port, ...)
        self.low_level.connect()

    def disconnect(self) -> None:
        """释放硬件。必须幂等，可安全在任意状态调用。"""
        if self.low_level is None:
            return
        try:
            self.low_level.disconnect()
        finally:
            self.low_level = None

    def get_observation(self) -> RobotObservation:
        """最佳努力快照。传感器短暂故障不应抛异常。"""
        ll = self.low_level
        if ll is None:
            return RobotObservation()  # 返回空观测而非抛异常

        pose_raw = ll.get_pose()
        pose = {"x": pose_raw.x, "y": pose_raw.y, "z": pose_raw.z,
                "rx": getattr(pose_raw, "rx", 0.0),
                "ry": getattr(pose_raw, "ry", 0.0),
                "rz": getattr(pose_raw, "rz", 0.0)}

        rgb = None
        depth = None
        if "vision.camera" in self.capabilities:
            frames = ll.grab_frames()
            if frames is not None:
                rgb, depth = frames

        return RobotObservation(pose=pose, rgb=rgb, depth=depth)
```

### 5.3 RobotObservation 字段填充规范

| 字段 | 类型 | 说明 |
|------|------|------|
| `pose` | `dict` or `None` | SCARA: `{"x","y","z","r"}`; 6-DOF: `{"x","y","z","rx","ry","rz"}` |
| `joints` | `list[float]` or `None` | 关节角，单位弧度或度（按机器人惯例） |
| `rgb` | `np.ndarray` or `None` | H×W×3 uint8，基座相机或腕部相机 |
| `depth` | `np.ndarray` or `None` | H×W float32 米，应与 `rgb` 对齐 |
| `extra` | `dict` | 其他状态（夹爪宽度、力/力矩、状态标志） |

> 所有字段均可为 `None`。下游代码通过检查 `None` 判断可用性，而非 `hasattr`。

### 5.4 暴露安全边界与机器人常量属性

SafetyRail 通过 `session.env` 读取安全边界；Api 层通过 Env 属性读取机器人常量。你的 Env 类应暴露以下属性：

```python
class MyEnv(BaseRobotEnv):
    # ---- 安全边界（SafetyRail 读取）----

    @property
    def z_min_safe(self) -> float:
        """安全 Z 下限 (mm)。SafetyRail 拒绝低于此值的 goto_xyzr 调用。"""
        return self._cfg.z_min_safe_mm  # 从 Config 读取

    @property
    def workspace_bounds(self) -> tuple | None:
        """XY 工作空间边界 (xmin, ymin, xmax, ymax)，或 None 表示不限制。"""
        if self._cfg.x_min_mm is not None:
            return (self._cfg.x_min_mm, self._cfg.y_min_mm,
                    self._cfg.x_max_mm, self._cfg.y_max_mm)
        return None

    # ---- 机器人常量（Api 层读取）----

    @property
    def home_pose(self):
        """Home 位姿 (驱动 Pose 对象)。Mixin.get_home_pose() 读此属性。"""
        if self.low_level is not None:
            return self.low_level.home_pose
        return None   # 连接前返回 None

    @property
    def tool_offset_mm(self) -> float:
        """法兰→工具末端的 Z 向偏移 (mm)。Api.get_pose() 用此做 tip↔flange 转换。"""
        if self.low_level is not None:
            return float(self.low_level.tool_offset_mm)
        return 0.0     # 连接前返回安全默认值
```

> `z_min_safe` 和 `workspace_bounds` 是 `BaseRobotEnv` 的合约属性（默认 `None`），
> SafetyRail 直接读取。`home_pose` 和 `tool_offset_mm` 同理，
> 默认值为 `None` 和 `0.0`，适配器覆盖为 `@property` 委托到驱动即可。

### 5.5 继承的运动动词（默认委托驱动）

`BaseRobotEnv` 已提供 `home / get_flange_pose / move_to_flange / move_joint /
set_end_effector / grab_rgb`，方法体默认转调 `low_level` 对应方法。

- `set_end_effector(engaged)` 基于 `self.capabilities` 确定性分发：
  `grasp.parallel` → `driver.set_gripper(engaged)`，
  `grasp.suction` → `driver.set_suction(engaged)`，二者实现其一即可。
- `grab_rgb()` 默认委托 `get_observation().rgb`；适配器可覆盖为更高效的路径
  （如直接从驱动取帧）。

只要驱动满足 `RobotDriver` Protocol，Env 无需实现它们；
仅当某动词需要 body 专属逻辑时才覆写。

### 5.6 可选覆写

```python
def reset(self) -> None:
    """将机械臂恢复到安全位姿。默认无操作，建议覆写。"""

def emergency_stop(self) -> None:
    """软件级急停。默认无操作。注意：物理急停必须由硬件层完成。"""
```

### 5.7 获取单帧图像 — grab_rgb()

`BaseRobotEnv.grab_rgb()` 是 Mixin `get_image()` 的底层 Env 方法：
默认实现返回 `get_observation().rgb`，对于大多数适配器已足够（不需要覆写）。

```python
# BaseRobotEnv 默认实现（无需覆写）
def grab_rgb(self) -> Optional[np.ndarray]:
    """单帧 RGB，供 Mixin.get_image() 使用。"""
    return self.get_observation().rgb
```

如果你的驱动有比 `get_observation()` 更快的取帧路径，可在 Env 中覆盖：

```python
class MyEnv(BaseRobotEnv):
    def grab_rgb(self) -> Optional[np.ndarray]:
        """优化路径：直接从驱动取帧，跳过 pose/joints 读取。"""
        ll = self.low_level
        if ll is None:
            return None
        frames = ll.grab_frames()
        return None if frames is None else frames[0]
```

---

## 6. 步骤 4：实现 Api 接口类

Api 是 LLM 看到的工具面。运动 / 关节 / 抓取 / `get_image` 的 Mixin 方法带有委托默认实现
（转发到 Env 动词、再到驱动），组合 Mixin 即可使用；你需要写的通常是两类：① 带专属几何的方法
（`get_pose` 的工具偏移、`goto_xyzr` 的 tip↔flange 补偿）；② 高层视觉方法
（`get_grasp_info_simple` / `pixel_to_base_xyz` / `analyze_scene`，无通用默认，必须实现）。

### 6.1 选择 Mixin 组合

Api 通过多继承组合 Mixin。三种典型场景：

**场景 A — SCARA + 吸盘**：
```python
from jiuwensymbiosis.api.mixins import MotionMixin, SuctionMixin
from jiuwensymbiosis.api.base import BaseRobotApi

class MyApi(MotionMixin, SuctionMixin, BaseRobotApi):
    """4-DOF SCARA + 吸盘末端"""
    # capabilities 自动 = {"motion.cartesian", "grasp.suction"}
```

**场景 B — 6-DoF + 平行夹爪**：
```python
from jiuwensymbiosis.api.mixins import (MotionMixin, JointMotionMixin,
                                          ParallelGripperMixin)

class MyApi(MotionMixin, JointMotionMixin, ParallelGripperMixin, BaseRobotApi):
    """6-DOF 机械臂 + 平行夹爪"""
    # capabilities 自动 = {"motion.cartesian", "motion.joint", "grasp.parallel"}
```

**场景 C — 6-DoF + 夹爪 + 视觉**：
```python
from jiuwensymbiosis.api.mixins import (MotionMixin, JointMotionMixin,
                                          ParallelGripperMixin, VisionMixin)

class MyApi(MotionMixin, JointMotionMixin, ParallelGripperMixin,
            VisionMixin, BaseRobotApi):
    """6-DOF + 夹爪 + 眼在手视觉"""
    # capabilities 自动 = {上述四个}
```

### 6.2 @robot_tool 方法覆写规则

**规则 1 — 默认实现，按需覆写**：运动 / 关节 / 抓取 / `get_image` 的 Mixin 方法已实现为委托
`self.env` 契约动词，继承即可调用，无需覆写；仅当行为需要机型专属逻辑时才覆写。
`get_grasp_info_simple` / `pixel_to_base_xyz` / `analyze_scene` 没有通用默认，**必须由适配器实现**。

**规则 2 — 覆写时无需重新装饰**：`@robot_tool` 的 `desc`/`tags` 会随函数一起被继承，覆写方法体即使不重新装饰也能正常生成工具。

**规则 3 — 重新装饰以提供硬件特定描述**：如果要自定义描述（推荐），在覆写方法上再次使用 `@robot_tool`。这会覆盖继承的 `desc`，但 `capability` 属性仍从 Mixin 自动继承。

```python
# 方式 A：不重新装饰（使用 Mixin 默认描述）
def goto_xyzr(self, x, y, z, r=None) -> None:
    ...

# 方式 B：重新装饰（提供硬件特定描述 — 推荐）
@robot_tool(desc="Move MyRobot tip to (x, y, z[, r]) in mm/deg, base frame.", tags=["motion"])
def goto_xyzr(self, x, y, z, r=None) -> None:
    ...
```

> **重要**：`capability` 属性和 `tags` 会从 Mixin 继承。如果你重新装饰但**未指定** `tags`，将使用装饰器默认值（`None`），而非继承 Mixin 的 tags。建议重新装饰时显式指定 `tags`。

**规则 4 — 新增独有工具**：如果硬件支持 Mixin 未定义的独有功能，可以独立声明 `@robot_tool` 方法（如 Piper 的 `goto_pose` 6-DOF 全位姿运动）。

### 6.3 实现示例 — 只写需要覆写的方法

`home` / `goto_xyzr` / `open_gripper` / `close_gripper` 由 Mixin 默认委托，常见机型无需在 api 层重写。
下面只覆写带工具偏移的 `get_pose` / `get_home_pose`：

```python
class MyApi(MotionMixin, ParallelGripperMixin, BaseRobotApi):
    """home / goto_xyzr / open_gripper / close_gripper 继承 Mixin 默认委托；
    仅覆写带工具偏移的 get_pose / get_home_pose。"""

    def __init__(self, env: MyEnv, *, gripper_open_mm: float = 70.0):
        super().__init__(env)
        self._gripper_open_mm = gripper_open_mm

    @robot_tool(desc="Get current end-effector pose in mm/deg, base frame.")
    def get_pose(self) -> dict:
        p = self.env.get_flange_pose()
        tool_off = self.env.tool_offset_mm   # 机器人常量经 Env 属性
        return {"x": p.x, "y": p.y, "z": p.z - tool_off,
                "rx": p.rx, "ry": p.ry, "rz": p.rz}

    @robot_tool(desc="Get the home pose constants.")
    def get_home_pose(self) -> dict:
        hp = self.env.home_pose              # 机器人常量经 Env 属性
        return {"x": hp.x, "y": hp.y, "z": hp.z, "rx": hp.rx, "ry": hp.ry, "rz": hp.rz}
```

当末端需要倾斜或非顶视姿态时，再覆写 `goto_xyzr`（tip↔flange 几何属 body 专属，留在 api 层）：

```python
    @robot_tool(desc="Move tip to absolute (x, y, z[, r]) in mm/deg, base frame.", tags=["motion"])
    def goto_xyzr(self, x: float, y: float, z: float, r: float | None = None) -> None:
        if r is None:
            r = self.env.get_flange_pose().rz
        pose = FlangePose(x, y, z, 180.0, 0.0, r)   # 默认顶视；倾斜机型改 ry（参考 Piper 的 _TOOL_DOWN_RY=30）
        self.env.move_to_flange(pose)
```

### 6.4 视觉适配器 — 使用 _common 检测管线

如果你的适配器支持 `VisionMixin`，需要集成 GroundingDINO + SAM2 视觉检测服务。

**6.4.1 检测客户端初始化**

```python
from jiuwensymbiosis.adapters._common.detector_client import init_detector

class MyApi(MotionMixin, VisionMixin, BaseRobotApi):
    def __init__(self, env, *, detector_service_url="http://127.0.0.1:8114", ...):
        super().__init__(env)
        self._detector_service_url = detector_service_url
        self._seg_fn = None  # 懒加载

    def _ensure_detector(self):
        """懒加载检测客户端。"""
        if self._seg_fn is not None:
            return
        self._seg_fn = init_detector(self._detector_service_url)
```

**6.4.2 实现 get_grasp_info_simple（核心方法）**

```python
from jiuwensymbiosis.adapters._common.vision import detect_and_centroid, apply_xy_correction

def get_grasp_info_simple(self, object_name: str) -> dict:
    # 视觉标定数据经 env.low_level 受控穿透（RobotDriver + CameraDriver + VisionDriver Protocol）
    ll = self.env.low_level
    frames = ll.grab_frames()
    if frames is None:
        return {"ok": False, "reason": "no_camera"}
    rgb, depth_img_m = frames

    self._ensure_detector()

    # 1. 检测 + 质心计算（位姿经 Env 动词，不走 low_level）
    det = detect_and_centroid(
        rgb=rgb, depth_img_m=depth_img_m,
        seg_fn=self._seg_fn, object_name=object_name,
        tcp_at_grab=self.env.get_flange_pose(),
    )
    if not det.get("ok"):
        return det

    # 2. 像素→基座 XYZ 反投影（需要相机标定）
    xyz_raw = self._pixel_to_base(det["u"], det["v"], det["depth_m"])

    # 3. XY 校正
    xyz_final, corr_desc = apply_xy_correction(xyz_raw)
    top_z = float(xyz_final[2])

    return {
        "ok": True,
        "object": object_name,
        "position": [float(xyz_final[0]), float(xyz_final[1]), top_z],
        "grasp_z": top_z + self._grasp_z_offset_mm,
        "grasp_position": [float(xyz_final[0]), float(xyz_final[1]),
                           top_z + self._grasp_z_offset_mm],
        "place_z": top_z + self._chip_thickness_mm,
        "place_position": [float(xyz_final[0]), float(xyz_final[1]),
                           top_z + self._chip_thickness_mm],
        "score": float(det["best"]["score"]),
        "pixel_uv": [det["u"], det["v"]],
        "depth_m": det["depth_m"],
    }
```

### 6.5 委托模式总结

运动 / 末端 / 安全 / 机器人常量这类"语义清晰、所有 body 通用"的操作经 Env 的公开方法或属性下发
（Env 是硬件契约面），不覆写时由 Mixin 默认实现按下表自动转发到驱动。单帧图像同理
（`env.grab_rgb()`）。视觉标定数据则保留经 `env.low_level`（Protocol 类型约束的受控穿透）
访问——刻意不上提为 Env 方法以避免 Env 膨胀。

```
Api 方法                  → Env 公开方法/属性             → Driver 调用
──────────────────────────────────────────────────────────────────────────
home()                   → env.home()                   → driver.home()
get_pose()               → env.get_flange_pose()        → driver.get_pose()   (+ env.tool_offset_mm 常量)
get_home_pose()          → env.home_pose [属性]          → driver.home_pose
goto_xyzr(x,y,z,r)       → env.move_to_flange(pose)      → driver.move_to_pose_blocking(pose)
move_joint(q)            → env.move_joint(q)             → driver.move_joint_blocking(q)
close_gripper(f)         → env.set_end_effector(True)    → driver.set_gripper(True)   [capability 分发]
open_gripper()           → env.set_end_effector(False)  → driver.set_gripper(False)  [capability 分发]
get_image()              → env.grab_rgb()                → get_observation().rgb (默认) 或 driver.grab_frames()[0]
──────────────────────────────────────────────────────────────────────────  (以下经 env.low_level 受控穿透)
get_grasp_info_simple()  → env.get_flange_pose() +      → driver.get_pose() + grab_frames() + tf_flange_cam/calibration + 检测服务
                          env.low_level(标定) + _common
pixel_to_base_xyz()      → env.get_flange_pose() +      → driver.get_pose() + driver.tf_flange_cam / intrinsics
                          env.low_level(标定)
```

> `BaseRobotEnv` 为 `home / get_flange_pose / move_to_flange / move_joint /
> set_end_effector / grab_rgb` 提供默认委托实现，`home_pose / tool_offset_mm /
> z_min_safe / workspace_bounds` 为合约属性——因此运动/末端/取图/安全常见情形下
> Api 与 Env 都无需写，只要驱动满足 `RobotDriver` Protocol。
> `set_end_effector` 基于 `env.capabilities` 确定性分发：`grasp.parallel` →
> `set_gripper`，`grasp.suction` → `set_suction`。

---

## 7. 步骤 5：创建 Config 数据类

### 7.1 完整 Config 结构

```python
import dataclasses
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any


@dataclass
class MyConfig:
    # ==================== 基本信息 [必填] ====================
    name: str = "my_robot"

    # ==================== 硬件连接 [必填] ====================
    can_port: str = "can0"              # CAN/串口/网络 端口
    move_speed: int = 50                # 运动速度百分比 (0-100)

    # ==================== 运动学 [选填] ====================
    tool_offset_mm: float = 0.0         # 法兰 → 工具末端的 Z 向偏移
    home_pose_xyzrxryrz_mm_deg: list[float] = field(
        default_factory=lambda: [200.0, 0.0, 400.0, 0.0, 90.0, 0.0]
    )
    home_use_init_pose: bool = False    # 是否用当前位置作为 home 位姿

    # ==================== 安全边界 [选填] ====================
    z_min_safe_mm: float = 50.0         # Z 向安全下限 (SafetyRail 使用)
    x_min_mm: Optional[float] = 0.0     # X 向工作空间下界 (None=不限制)
    x_max_mm: Optional[float] = 700.0
    y_min_mm: Optional[float] = -500.0
    y_max_mm: Optional[float] = 500.0

    # ==================== 夹爪 [选填-仅 grasp.*] ====================
    gripper_open_mm: float = 70.0       # 打开宽度 (mm)
    gripper_effort: int = 1000          # 夹持力 (单位为驱动定义)

    # ==================== 视觉 [选填-仅 vision.*] ====================
    camera_serial: Optional[str] = None # 相机序列号 (None=禁用)
    camera_resolution: tuple[int, int] = (640, 480)
    camera_fps: int = 30

    # ==================== 检测校正 [选填-仅 vision.detection] ====================
    z_correction_mm: float = 0.0        # Z 方向常值校正
    grasp_z_offset_mm: float = -25.0    # 抓取点相对于物体顶面的偏移
    chip_thickness_mm: float = 75.0     # 堆叠放置偏移

    # ==================== 检测服务 [选填-仅 vision.detection] ====================
    detector_spawn: bool = True         # 是否自动启动检测子进程
    detector_url: str = "http://127.0.0.1:8114"
    detector_host: str = "127.0.0.1"
    detector_port: int = 8114

    # ==================== 加载器 [必填] ====================

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MyConfig":
        """从 dict 构造 Config。使用平铺结构（推荐）。"""
        valid = {f.name for f in dataclasses.fields(cls)}
        clean = {k: v for k, v in data.items() if k in valid}
        if "camera_resolution" in clean:
            clean["camera_resolution"] = tuple(clean["camera_resolution"])
        return cls(**clean)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MyConfig":
        """从 YAML 文件加载 Config。"""
        path = Path(path).resolve()
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)
```

### 7.2 YAML 结构约定

**推荐**：平铺结构（简单直接，适合新适配器）

```yaml
name: "my_robot"
can_port: "can0"
move_speed: 50
tool_offset_mm: 95.0
z_min_safe_mm: 50.0
gripper_open_mm: 70.0
```

**可选兼容**：嵌套结构（Piper 的历史约定，你的新适配器无需实现此兼容）

```yaml
env:
  cfg:
    low_level:
      can_port: "can0"
      move_speed: 50
```

> 如果你的模型需要兼容嵌套结构（如复用现有 YAML 文件），请在 `from_dict()` 中添加类似 Piper 的双路由逻辑。详见 `jiuwensymbiosis/adapters/piper/config.py` 的 `from_dict`。

### 7.3 必填/选填字段标注说明

| 标注 | 含义 | 示例字段 |
|------|------|---------|
| `[必填]` | 所有适配器都必须提供 | `name`, `can_port` |
| `[选填]` | 不填写有合理默认值 | `move_speed`, `z_min_safe_mm` |
| `[选填-仅 motion.joint]` | 仅当 Env 声明了该 capability 时才需要 | `joint_limits` |
| `[选填-仅 vision.detection]` | 仅当使用视觉检测能力时才需要 | `detector_url`, `z_correction_mm` |

---

## 8. 步骤 6：组装会话构建器

### 8.1 make_builder() 参数

```python
from jiuwensymbiosis.adapters._common.builder import make_builder

build_my_session = make_builder(
    cfg_cls=MyConfig,           # 你的 Config dataclass
    env_cls=MyEnv,              # 你的 BaseRobotEnv 子类
    api_cls=MyApi,              # 你的 BaseRobotApi 子类
    api_kwargs_from_cfg=...,    # 可选：从 Config 提取 Api.__init__ 额外参数
    sidecar_builders=[...],     # 可选：子进程启动器列表
    decorate=...,               # 可选：session 追加装饰回调
)
```

### 8.2 api_kwargs_from_cfg — 传递 Api 构造参数

当 Api 的 `__init__` 需要 env 之外的参数时（如检测服务 URL、校正参数），使用此回调：

```python
def _api_kwargs_from_cfg(cfg: MyConfig) -> dict:
    return {
        "detector_service_url": cfg.detector_url,
        "z_correction_mm": cfg.z_correction_mm,
        "grasp_z_offset_mm": cfg.grasp_z_offset_mm,
        "chip_thickness_mm": cfg.chip_thickness_mm,
    }

build_my_session = make_builder(
    MyConfig, MyEnv, MyApi,
    api_kwargs_from_cfg=_api_kwargs_from_cfg,
)
```

### 8.3 sidecar_builders — 检测子进程管理

如果使用视觉检测，需要注册检测子进程启动器。参考 `jiuwensymbiosis/adapters/_common/detector_sidecar.py:detector_subprocess()`：

```python
from jiuwensymbiosis.adapters._common.detector_sidecar import detector_subprocess

def _detector_sidecar(cfg: MyConfig):
    """根据配置决定是否启动检测子进程。"""
    if not cfg.detector_spawn:
        return None  # 不启动（使用外部服务实例）
    return lambda: detector_subprocess(
        host=cfg.detector_host,
        port=cfg.detector_port,
    )

build_my_session = make_builder(
    MyConfig, MyEnv, MyApi,
    api_kwargs_from_cfg=_api_kwargs_from_cfg,
    sidecar_builders=[_detector_sidecar],
)
```

> **sidecar 生命周期**：当调用 `session.connect()` 时自动启动，`session.disconnect()` 时自动关闭。如果端口已被占用，`detector_subprocess` 会假设已有一个外部实例在运行并跳过启动。

### 8.4 decorate — 向 session 注入自定义对象

如果需要向 session 注入额外对象（如 InProcessCodeTool 中使用的全局变量）：

```python
def _decorate(session, cfg: MyConfig) -> None:
    """将 Config 对象注入 session.extra_globals。"""
    session.extra_globals["my_cfg"] = cfg

build_my_session = make_builder(
    MyConfig, MyEnv, MyApi,
    decorate=_decorate,
)
```

### 8.5 极简接线（无 sidecar、无额外参数）

最简单的适配器 session.py 可以只有 5 行：

```python
"""build_my_session — 从 YAML 到可连接 Session 的一站式调用。"""
from jiuwensymbiosis.adapters._common.builder import make_builder
from jiuwensymbiosis.adapters.my_robot.config import MyConfig
from jiuwensymbiosis.adapters.my_robot.env import MyEnv
from jiuwensymbiosis.adapters.my_robot.api import MyApi

build_my_session = make_builder(MyConfig, MyEnv, MyApi)
```

### 8.6 验证

```python
# 从 YAML 构建
session = build_my_session.from_yaml("configs/my_robot/default.yaml")

# 从 dict 构建
session = build_my_session.from_dict({"name": "test", "can_port": "can0"})

# 连接并描述
with session:
    print(session.describe())
    # Output: {'name': 'my_robot', 'env': 'my_robot',
    #          'env_capabilities': ['grasp.parallel', 'motion.cartesian'],
    #          'api_capabilities': ['grasp.parallel', 'motion.cartesian'],
    #          'effective_capabilities': ['grasp.parallel', 'motion.cartesian']}
```

---

## 9. _common 共享模块速查

`jiuwensymbiosis/adapters/_common/` 提供了所有适配器可复用的工具模块。不要重新发明轮子。

| 模块 | 主要接口 | 用途 | 何时使用 |
|------|---------|------|---------|
| `builder.py` | `make_builder()` | 通用会话构建工厂 | **所有适配器必用** |
| `detector_client.py` | `init_detector(service_url) → seg_fn` | 检测服务 HTTP 客户端 | 视觉适配器必用 |
| `detector_sidecar.py` | `detector_subprocess(host, port, ...)` | 检测子进程生命周期 | 视觉适配器必用 |
| `vision.py` | `detect_and_centroid(rgb, depth, seg_fn, object_name, ...)` | 检测 + 质心 + 中值深度 | 视觉适配器必用 |
| `vision.py` | `apply_xy_correction(xyz_raw, xy_transform, xy_correction_mm)` | XY 坐标校正 | 视觉适配器按需 |
| `vision.py` | `dump_grasp_debug(rgb, object_name, best, u, v, ...)` | 检测结果 dumping 到磁盘 | 视觉调试按需 |
| `calibration.py` | 手眼标定加载 | `tf_flange_cam` 变换矩阵 | 视觉适配器按需 |
| `camera.py` | 相机工具函数 | 相机初始化和帧抓取 | 视觉适配器按需 |
| `geometry.py` | 几何变换辅助 | 坐标系变换、投影 | 视觉适配器按需 |
| `protocol.py` | `RobotDriver`/`JointDriver`/`GripperDriver`/`SuctionDriver`/`CameraDriver`/`VisionDriver` | 驱动接口 Protocol（结构化类型，供标注 + D-14 校验）。多协议驱动可定义复合 Protocol（参考 `PiperFullDriver` = `RobotDriver + JointDriver + CameraDriver + GripperDriver + VisionDriver`）做 `low_level` 访问的类型收紧，mypy/pyright 会静态校验全部成员。 | **驱动应满足对应子集** |
| `safety.py` | `WorkspaceBounds(z_min_safe, tool_offset_mm, ...)` + `check_flange_z()` | tip/flange 双坐标系 Z 楼面换算与拦截 | 驱动层做 Z 防御按需 |

### 关键接口签名与用法

#### init_detector

```python
from jiuwensymbiosis.adapters._common.detector_client import init_detector

seg_fn = init_detector("http://127.0.0.1:8114")
# seg_fn 是可调用对象: seg_fn(image_ndarray, text_prompt="blue box") -> list[dict]
# 每个返回 dict: {"mask": np.ndarray[bool], "box": [x1,y1,x2,y2],
#                  "score": float, "label": str}
```

#### detect_and_centroid

```python
from jiuwensymbiosis.adapters._common.vision import detect_and_centroid

result = detect_and_centroid(
    rgb=rgb_ndarray,          # HxWx3 uint8
    depth_img_m=depth_ndarray, # HxW float32 (meters)
    seg_fn=seg_fn,            # 来自 init_detector 的结果
    object_name="red block",
    tcp_at_grab=pose_at_grab, # 仅用于日志记录
)
# 成功: {"ok": True, "u": float, "v": float, "depth_m": float,
#        "best": {...detector_result...}, "mask_shape": (h,w),
#        "img_shape": (w,h)}
# 失败: {"ok": False, "reason": "no_detection"|"empty_mask"|"no_valid_depth"}
```

#### apply_xy_correction

```python
from jiuwensymbiosis.adapters._common.vision import apply_xy_correction

xyz_final, corr_desc = apply_xy_correction(
    xyz_raw=np.array([x, y, z], dtype=np.float64),
    xy_transform=calib.get("xy_transform"),   # 优先：多点仿射/相似/平移校正
    xy_correction_mm=calib.get("xy_correction_mm"),  # 回退：单点平移校正
)
```

---

## 10. SafetyRail 安全边界配置指南

### 10.1 SafetyRail 数据流

```
Config                 Env                          SafetyRail
──────                 ───                          ──────────
z_min_safe_mm ───→  z_min_safe (property) ───────→ _resolve_z_floor() (读取)
x_min_mm       ───→  workspace_bounds (property) ─→ _resolve_xy_bounds() (读取)
x_max_mm       ───→  (xmin,ymin,xmax,ymax)
y_min_mm       ───→
y_max_mm       ───→
```

> `agent/builder.py` 用 `SafetyRail(session)` 构造（不显式传 bounds），
> SafetyRail 自动从 `env.z_min_safe` / `env.workspace_bounds` 回退读取。
> 可用 `enforce_xy_from_env=False` 关闭 env 回退。

### 10.2 SafetyRail Z 下限三级回退

```
显式传入 SafetyRail(z_floor_mm=30)  →  使用该值
        ↓ (未显式传入)
session.env.z_min_safe              →  使用该值（Env 属性）
        ↓ (env 也未暴露)
None                                →  不检查 Z 下限（不推荐生产环境）
```

### 10.3 拦截行为

SafetyRail 默认监控以下工具名称：
- `goto_xyzr` — 检查 z 是否 ≥ z_floor，x/y 是否在 workspace_bounds 内
- `goto_pose` — 同上
- 当使用 `RobotControlTool` 时，rail 会自动解包 `robot_control` 的 `action` 字段

被拦截时，SafetyRail 抛出 `ValueError`（而不是静默拒绝），LLM 看到错误消息后可自行修正。

### 10.4 推荐配置

```yaml
# config.yaml — 安全边界参数
z_min_safe_mm: 50.0     # 桌面以上 50mm 为安全下限
x_min_mm: 0.0           # 工作空间左边界
x_max_mm: 600.0         # 工作空间右边界
y_min_mm: -400.0        # 工作空间后边界
y_max_mm: 400.0         # 工作空间前边界
```

然后在 `env.py` 中读取并暴露（见 5.4）。

---

## 11. 编写单元测试

### 11.1 Mock 驱动开发模式

在没有真实硬件时，使用 Mock 驱动进行框架集成验证：

```
┌─────────────────────────────────────┐
│  测试文件 (test_my_adapter.py)       │
├─────────────────────────────────────┤
│  1. 创建 MockMyDriver (在内存中模拟) │
│  2. 用 Mock 驱动构造 MyEnv           │
│  3. 用 MyEnv 构造 MyApi              │
│  4. 构造 RobotSession                │
│  5. 验证 build_robot_tools 产出      │
│  6. 验证 session.describe()          │
│  7. 验证 session.connect/disconnect  │
└─────────────────────────────────────┘
```

### 11.2 测试模板

```python
"""tests/unit_tests/adapters/my_robot/test_session.py"""
import pytest
from jiuwensymbiosis.adapters.my_robot.config import MyConfig
from jiuwensymbiosis.adapters.my_robot.env import MyEnv
from jiuwensymbiosis.adapters.my_robot.api import MyApi
from jiuwensymbiosis.adapters.my_robot.session import build_my_session
from jiuwensymbiosis.tools.builder import build_robot_tools


class TestMyApiCapabilities:
    """验证 Api 中声明的能力是否与 Env 一致。"""

    def test_capabilities_match(self):
        cfg = MyConfig()
        env = MyEnv(cfg)
        api = MyApi(env)
        # Env 的能力应包含 Api 的所有 Mixin 能力
        assert api.capabilities.issubset(env.capabilities), \
            f"Api capabilities {api.capabilities} not in env {env.capabilities}"


class TestToolGeneration:
    """验证 @robot_tool 方法是否正确生成为 LLM 工具（含继承的默认实现）。"""

    def test_tools_generated(self):
        cfg = MyConfig()
        env = MyEnv(cfg)
        api = MyApi(env)
        tools = build_robot_tools(api)
        tool_names = {t.name for t in tools}
        assert "home" in tool_names       # 继承自 MotionMixin 默认委托
        assert "goto_xyzr" in tool_names
        assert "get_pose" in tool_names


class TestSessionBuilder:
    """验证 make_builder 构建的 Session 可用。"""

    def test_build_from_dict(self):
        session = build_my_session.from_dict({"name": "test", "can_port": "mock"})
        with session:
            desc = session.describe()
            assert desc["name"] == "test"
            assert "motion.cartesian" in desc["env_capabilities"]

    def test_connect_idempotent(self):
        """connect() 重复调用应是安全的。"""
        session = build_my_session.from_dict({"name": "test", "can_port": "mock"})
        with session:
            session.connect()  # 第二次调用应无操作
        session.disconnect()   # 退出后再次 disconnect 应安全
```

### 11.3 现有测试参考

| 测试文件 | 测试内容 | 参考价值 |
|---------|---------|---------|
| `tests/unit_tests/agent/test_builder.py` | `build_robot_agent()` 的完整测试 | 学习如何测试完整的 agent 构建流程 |
| `tests/unit_tests/agent/test_session.py` | `RobotSession` 生命周期测试 | 学习 session connect/disconnect 测试 |
| `tests/unit_tests/api/test_base.py` | `BaseRobotApi` 基类测试 | 学习 capability 推导测试 |
| `tests/unit_tests/api/test_mixins.py` | Mixin 方法元数据测试 | 学习 `@robot_tool` 元数据校验 |
| `tests/unit_tests/env/test_base.py` | `BaseRobotEnv` 基类测试 | 学习 capability 校验测试 |
| `tests/mocks/mock_api.py` | Mock API 参考实现 | 学习如何写 Mock Api |
| `tests/mocks/mock_env.py` | Mock Env 参考实现 | 学习如何写 Mock Env |

---

## 12. 常见问题与排错

### Q1: "ValueError: MyEnv declares unknown capabilities: {'my_custom_cap'}"
**原因**：在 `env.capabilities` 中声明了不在 `KNOWN_CAPABILITIES` 中的字符串。
**解决**：
- 方案 A：检查拼写错误，确保使用标准 capability 字符串
- 方案 B：如需新增能力，参见"附录 B — 进阶：扩展 Capability 词汇表"

### Q2: connect() 被重复调用导致硬件错误
**原因**：未在 `connect()` 中做幂等检查。
**解决**：使用 '已连接' 标志或检查 `self.low_level is not None`。

```python
def connect(self) -> None:
    if self.low_level is not None:
        return  # 已连接，幂等返回
    self.low_level = MyDriver(...)
    self.low_level.connect()
```

### Q3: vision.detection 无检测服务时如何处理
**原因**：检测服务未启动或不可达。
**解决**：`init_detector` 返回的 `seg_fn` 在调用时已内置容错——如果服务不可达，会返回 `[]`（空列表），而非抛异常。`detect_and_centroid` 会将空结果转为 `{"ok": False, "reason": "no_detection"}`。你的 `get_grasp_info_simple()` 应传递此错误。

### Q4: 工具未生成 — 预期的 home() / goto_xyzr() 不在工具列表中
**原因 A**：`Env.capabilities` 缺少对应的 capability 字符串（工具按 `api∩env` 门控）
**解决**：确保 `env.capabilities` 包含 Api 所有 Mixin 的 capability；`validate_adapter` 的 A-08 会以 ERROR 报出此不一致

**原因 B**：Api 未继承对应的 Mixin
**解决**：检查 Api 类的多继承列表

### Q5: 坐标系混淆 — goto_xyzr 的 z 是 TIP 还是 FLANGE？
**明确**：框架约定 `goto_xyzr(x, y, z, r)` 的 `z` 是 **TIP 坐标系**（工具末端）。tip↔flange 补偿在 `goto_xyzr()` 内做，最终经 `env.move_to_flange()` 下发：

```python
def goto_xyzr(self, x, y, z, r=None):
    tool_off = self.env.tool_offset_mm  # 机器人常量经 Env 属性
    flange_z = z + tool_off
    self.env.move_to_flange(FlangePose(x, y, flange_z, ...))
```

### Q6: SKILL.md 加载失败 — "SkillUseRail: no skills found"
**原因**：`workspace` 的 `restrict_to_work_dir=True`（默认行为）阻止了读取包内的 SKILL.md 文件。
**解决**：`build_robot_agent()` 内部已设置 `restrict_to_work_dir=False`，所以这不是你适配器的问题。如果自己调用 `create_deep_agent()`，确保传入 `restrict_to_work_dir=False`。

---

## 附录 A：最小可运行适配器 — SCARA + 吸盘

以下是一个完整的 ~150 行示例，展示了从驱动到 Session 的完整实现。api.py 只覆写两个带字段命名差异的方法，其余继承 Mixin 默认委托。

### A.1 lowlevel.py（Mock 驱动，~40 行）

```python
"""Mock SCARA 驱动 — 替换为真实串口/CAN 通信。"""
from types import SimpleNamespace

class MockScaraDriver:
    def __init__(self):
        self._pose = {"x": 200.0, "y": 0.0, "z": 250.0, "rz": 0.0}
        self.home_pose = SimpleNamespace(x=200.0, y=0.0, z=250.0, rz=0.0)
        self.tool_offset_mm = 0.0
        self._connected = False

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_pose(self):
        p = self._pose
        return SimpleNamespace(x=p["x"], y=p["y"], z=p["z"], rz=p["rz"])

    def home(self) -> None:
        self._pose = {"x": 200.0, "y": 0.0, "z": 250.0, "rz": 0.0}

    def move_to_pose_blocking(self, pose) -> None:
        self._pose["x"] = pose.x
        self._pose["y"] = pose.y
        self._pose["z"] = pose.z
        self._pose["rz"] = getattr(pose, "rz", 0.0)

    def set_suction(self, on: bool) -> None:
        pass
```

### A.2 config.py（~30 行）

```python
"""SCARA 吸盘机械臂配置。"""
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ScaraConfig:
    name: str = "scara"
    serial_port: str = "/dev/ttyUSB0"         # [必填] 串口路径
    z_min_safe_mm: float = 30.0               # [选填] 安全 Z 下限
    tool_offset_mm: float = 0.0               # [选填] 工具偏移
    home_xyzr: tuple = (200.0, 0.0, 250.0, 0.0)  # [选填] Home 位姿

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScaraConfig":
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScaraConfig":
        with Path(path).open("r") as f:
            return cls.from_dict(yaml.safe_load(f) or {})
```

### A.3 env.py（~30 行）

```python
"""SCARA Env — 包装 MockScaraDriver。"""
from jiuwensymbiosis.env.base import BaseRobotEnv, RobotObservation
from jiuwensymbiosis.adapters.my_scara.lowlevel import MockScaraDriver


class ScaraEnv(BaseRobotEnv):
    capabilities = frozenset({"motion.cartesian", "grasp.suction"})
    name = "scara"

    def __init__(self, cfg):
        self._cfg = cfg
        self.low_level = None

    def connect(self) -> None:
        if self.low_level is not None:
            return
        self.low_level = MockScaraDriver()
        self.low_level.connect()

    def disconnect(self) -> None:
        if self.low_level is not None:
            self.low_level.disconnect()
            self.low_level = None

    def get_observation(self) -> RobotObservation:
        ll = self.low_level
        if ll is None:
            return RobotObservation()
        p = ll.get_pose()
        return RobotObservation(pose={"x": p.x, "y": p.y, "z": p.z, "r": p.rz})

    @property
    def z_min_safe(self) -> float:
        return self._cfg.z_min_safe_mm

    @property
    def home_pose(self):
        if self.low_level is not None:
            return self.low_level.home_pose
        return None

    @property
    def tool_offset_mm(self) -> float:
        if self.low_level is not None:
            return self.low_level.tool_offset_mm
        return 0.0
```

> `home`/`get_flange_pose`/`move_to_flange`/`set_end_effector`/`grab_rgb` 由 `BaseRobotEnv`
> 默认委托给 `low_level`，ScaraEnv 无需实现。`home_pose`/`tool_offset_mm` 已暴露为属性。

### A.4 api.py（~20 行）

`home` / `goto_xyzr` / `activate_suction` / `deactivate_suction` / `get_home_pose` / `get_image` 继承 Mixin 默认委托——这里只覆写 `get_pose`，把驱动的 `rz` 暴露成 SCARA 习惯的 `r` 字段：

```python
"""SCARA Api — 4-DOF + 吸盘。home / goto_xyzr / 吸盘开关均继承 Mixin 默认委托。"""
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.api.decorators import robot_tool
from jiuwensymbiosis.api.mixins import MotionMixin, SuctionMixin


class ScaraApi(MotionMixin, SuctionMixin, BaseRobotApi):
    """4-DOF SCARA + 吸盘末端。仅覆写 get_pose/get_home_pose 以使用 'r' 字段命名。"""

    @robot_tool(desc="Get current tip pose in mm/deg.")
    def get_pose(self) -> dict:
        p = self.env.get_flange_pose()
        return {"x": p.x, "y": p.y, "z": p.z, "r": p.rz}

    @robot_tool(desc="Get home pose constants.")
    def get_home_pose(self) -> dict:
        hp = self.env.home_pose
        return {"x": hp.x, "y": hp.y, "z": hp.z, "r": hp.rz}
```

### A.5 session.py（~15 行）

```python
"""build_scara_session"""
from jiuwensymbiosis.adapters._common.builder import make_builder
from jiuwensymbiosis.adapters.my_scara.config import ScaraConfig
from jiuwensymbiosis.adapters.my_scara.env import ScaraEnv
from jiuwensymbiosis.adapters.my_scara.api import ScaraApi

build_scara_session = make_builder(ScaraConfig, ScaraEnv, ScaraApi)
```

### A.6 config_template.yaml（~15 行）

```yaml
# SCARA 吸盘机械臂配置
name: "scara"
serial_port: "/dev/ttyUSB0"     # [必填] 串口路径
z_min_safe_mm: 30.0             # [选填] 安全Z下限
tool_offset_mm: 0.0             # [选填] 工具偏移
home_xyzr: [200.0, 0.0, 250.0, 0.0]  # [选填] Home 位姿 (x,y,z,r)
```

### A.7 验证运行

```bash
python scripts/validate_adapter.py --module jiuwensymbiosis.adapters.my_scara
```

---

## 附录 B：进阶 — 扩展 Capability 词汇表

当现有 9 个 capability 字符串无法覆盖你的硬件能力时（如移动底盘导航、力/力矩传感），需要扩展词汇表。

### 步骤

**1. 在 `KNOWN_CAPABILITIES` 中添加新字符串**

编辑 `jiuwensymbiosis/env/base.py`：

```python
KNOWN_CAPABILITIES = frozenset({
    # ... 现有 9 个 ...
    "mobile.navigation",  # [新增] 移动底盘导航
})
```

> ⚠️ 不更新此处会导致 `__init_subclass__` 抛出 ValueError。

**2. 在 `api/mixins.py` 中创建新 Mixin 类**

```python
class MobileNavigationMixin:
    """移动底盘导航 capability mixin."""
    capability = "mobile.navigation"

    @robot_tool(desc="Navigate the mobile base to (x, y, theta) in world frame.", tags=["motion"])
    def navigate_to(self, x_m: float, y_m: float, theta_rad: float = 0.0) -> dict:
        """导航到世界坐标系的指定位置。"""
        raise NotImplementedError

    @robot_tool(desc="Get current mobile base pose in world frame.")
    def get_base_pose(self) -> dict:
        """获取移动底盘当前位姿。"""
        raise NotImplementedError
```

> 若新动作能映射到 Env 公开动词，也可像内置 Mixin 那样写成委托默认实现；否则保留 `raise NotImplementedError` 由适配器实现。

**3. 可选：编写新 Rail**

```python
class CollisionAvoidanceRail(AgentRail):
    """碰撞检测 — 适用于 mobile.navigation。"""
    async def before_tool_call(self, ctx):
        ...
```

**4. 在你的 Env 和 Api 中使用**

```python
class MyEnv(BaseRobotEnv):
    capabilities = frozenset({"motion.cartesian", "mobile.navigation"})

class MyApi(MobileNavigationMixin, MotionMixin, BaseRobotApi):
    def navigate_to(self, x_m, y_m, theta_rad=0.0):
        ...
    def get_base_pose(self):
        ...
```

---

## 附录 C：Piper 适配器对照拆解

按本指南的六步流程逐一标注 Piper 适配器的实现位置：

| 步骤 | 内容 | Piper 实现文件 | 关键位置 |
|------|------|---------------|---------|
| 步骤 1 | 定义能力 | `adapters/piper/env.py` | capabilities frozenset |
| 步骤 2 | 低层驱动 | `adapters/piper/lowlevel.py` | PiperLowLevel 类 |
| 步骤 2 | 几何变换 | `adapters/piper/geometry.py` | FlangePose, pixel_and_depth_to_base_xyz |
| 步骤 2 | 标定加载 | `adapters/piper/_calibration.py` | 手眼标定矩阵 |
| 步骤 3 | Env 实现 | `adapters/piper/env.py` | PiperEnv 类 |
| 步骤 4 | Api 实现 | `adapters/piper/api.py` | PiperApi 类（home/move_joint/夹爪/get_image 继承 Mixin 默认委托） |
| 步骤 5 | Config | `adapters/piper/config.py` | PiperConfig + DetectorServerConfig |
| 步骤 6 | Session | `adapters/piper/session.py` | make_builder 接线 |
| 步骤 6 | 检测 Sidecar | `adapters/piper/session.py` | _detector_sidecar_from_cfg |

**Piper 特定复杂度（你的适配器可能不需要）**：

| 特性 | 说明 | 是否需要 |
|------|------|---------|
| 倾斜工具补偿 (`_TOOL_DOWN_RX/RY`) | Piper 的末端工具需约 30° 倾斜才能可达，`goto_xyzr` 因此覆写 | 按需 |
| Z 方向常值校正 (`z_correction_mm`) | 手眼标定未重新标定前的临时补丁 | 按需 |
| 嵌套 YAML 兼容 (`env.cfg.low_level.*`) | 历史遗留的 YAML 兼容层 | ❌ 新适配器无需 |
| 抓取 debug dumping | 保存检测结果到磁盘供离线分析 | 按需 |
