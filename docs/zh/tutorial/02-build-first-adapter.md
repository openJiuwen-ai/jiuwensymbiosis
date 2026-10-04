# 构建第一个机器人适配器

> 类别：Tutorial。内容以治理前原始文档为基线重组。

本教程用一个无硬件依赖的 SCARA + 吸盘示例，带你第一次完成可运行适配器。这里负责“学会适配器边界”，不展开厂商 SDK、视觉标定和真机验收；Mock 示例跑通后，再进入[移植机器人硬件适配器](../how-to/port-hardware-adapter.md)完成生产接入。接口字段可随时查阅[机器人适配器参考](../reference/adapter-reference.md)。

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
│                   │  DiagnosisRail / TraceRail   │  可选诊断与执行追踪
├──────────────────────────────────────────────────┤
│  Tool Layer       │  build_robot_tools(api)      │  每个 @implements → 一个 LLM 工具
│                   │  RobotControlTool(api)       │  单一入口 action/params 分发
│                   │  InProcessCodeTool           │  进程内 Python 代码执行
├──────────────────────────────────────────────────┤
│  Skill Layer      │  visual_pick/SKILL.md        │  预置操作流程文档
│                   │  visual_place/SKILL.md       │  SkillUseRail 自动加载
│                   │  transport/SKILL.md          │
├──────────────────────────────────────────────────┤
│  API Layer        │  ActionSpec 词表             │  动作契约（名称/能力/前置/效果）
│  (Action          │  @implements(SPEC)           │  绑定一条契约 + ToolMeta
│   Contract)       │  api/defaults                │  通用实现（自由函数，非基类）
│                   │  BaseRobotApi                │  持有 env 引用
├──────────────────────────────────────────────────┤
│  Env Layer        │  BaseRobotEnv                │  硬件契约面（唯一）
│  (Hardware        │  connect/disconnect/observe  │  能力声明 (env.capabilities)
│   Abstraction)    │  move_to_flange/set_end_...  │  默认委托驱动；home 由本体实现
│                   │  home_pose/tool_offset_mm    │  机器人常量属性
│                   │  grab_rgb()                  │  单帧图像（默认走 get_observation）
│                   │  z_min_safe/workspace_bounds │  安全契约属性
│                   │  joint_limits                │  关节软限位（仅 motion.joint）
│                   │  low_level: RobotDriver      │  受控穿透点（视觉标定/厂商特有）
├──────────────────────────────────────────────────┤
│  Hardware Layer   │  XxxDriver (lowlevel.py)     │  实现 Driver Protocol
│  (Your Code)      │  串口/CAN/Socket 等          │  适配器开发者主要工作
└──────────────────────────────────────────────────┘
```

> Env 是 Agent/Rails/Tools 与硬件之间的**唯一契约面**：
> - 运动/末端经 Env 动词（`home`/`move_to_flange`/`move_joint`/`get_flange_pose`/`set_end_effector`）
> - 机器人常量经 Env 属性（`home_pose`/`tool_offset_mm`）
> - 安全边界经 Env 属性（`z_min_safe`/`workspace_bounds`/`joint_limits`）
> - 单帧图像经 Env 方法（`grab_rgb()`，默认委托 `get_observation().rgb`）
> - 视觉标定数据经 `env.low_level`（`RobotDriver` + 子 Protocol 类型约束的受控穿透）
>
> 上层经 Env 公开 API 访问硬件，不 `getattr` 私有驱动。`set_end_effector` 基于声明能力（`grasp.parallel`/`grasp.suction`）做确定性分发。

### 核心概念速览

| 概念 | 定义 | 谁定义 |
|------|------|--------|
| **Capability** | 硬件能力的命名字符串，如 `"motion.cartesian"` | `env/base.py:KNOWN_CAPABILITIES` |
| **ActionSpec** | 一个动作的契约：名称/描述/能力门/参数/结果形状/前置条件与效果/位置新鲜度 | 类定义在 `api/decorators.py`，词表实例在 `api/actions.py` |
| **Env** | 硬件驱动包装器，实现 `connect/disconnect/get_observation` | 适配器开发者 |
| **Api** | 继承 `BaseRobotApi`，用 `@implements(SPEC)` 绑定每条动作；无差异的转发 `api/defaults`，有几何或相机差异的显式实现 | 适配器开发者 |
| **Config** | hardware 参数的 dataclass，含 `from_yaml/from_dict` | 适配器开发者 |
| **Session** | 将 Env、Api、客户端和可选本地进程打包为生命周期单元 | `make_builder()` 自动生成 |
| **HTTP 推理服务** | 提供检测等模型接口；`remote` 连接外部管理的服务，包括本机手动启动的服务 | `perception/detector_client.py` |
| **本地托管子进程（sidecar）** | 仅 `detector.mode: local` 时随 Session 启停的检测进程 | `perception/detector_sidecar.py`（`_common/builder.py` 提供组装封装） |

### 约定

- Env 的 `capabilities` 是**手动声明**的 frozenset
- Api 的 `capabilities` 是**从动作的 spec 反推**的（每个 `@implements` 贡献自己 `ActionSpec` 的能力，再加上声明的 marker 能力类属性）
- 工具按 **`api.capabilities ∩ env.capabilities`** 门控：Api 有而 Env 无的能力，其工具**运行时不会暴露给 LLM**（`build_robot_tools` 强制交集，`validate_adapter` 会报 ERROR）。`session.describe()` 的 `effective_capabilities` 即此交集。Env 有而 Api 无的标记能力（如 `vision.camera`）不影响运行。

---

## 1. 新手速览：5 分钟跑通最小适配器

```bash
# 1. 复制模板
cp -r templates/xxx_adapter/ jiuwensymbiosis/adapters/my_robot/

# 2. 修改文件，并把所有 Xxx/xxx/build_xxx_session 占位符替换为机器人名称
#  - config.py: 填写硬件连接参数
#  - lowlevel.py: 实现硬件通信（或先写 Mock）
#  - env.py: 声明 capabilities + connect/disconnect/observe
#  - api.py: @implements(SPEC) 绑定每条动作；无差异的转发 defaults，有专属几何的写方法体
#  - session.py: 保留声明式 make_builder 接线，但必须重命名类和 build_xxx_session
#  - __init__.py: 同步重命名导出的 build_xxx_session
#  - config_template.yaml: 复制到 configs/my_robot/default.yaml 后填写

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

## 2. 实现 Mock Driver

下面开始实现完整的最小适配器。主要工作量在驱动；Api 只用 `@implements` 绑定没有本体差异的动作并转发 `api.defaults`，SCARA 的字段命名差异单独覆写。

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

    def close(self) -> None:
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

## 3. 定义 Config

```python
"""SCARA 吸盘机械臂配置。"""
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jiuwensymbiosis.adapters._common.config import load_yaml_config


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
        return load_yaml_config(cls, path)
```

## 4. 包装 Env

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
            self.low_level.close()
            self.low_level = None

    def get_observation(self) -> RobotObservation:
        ll = self.low_level
        if ll is None:
            return RobotObservation()
        p = ll.get_pose()
        return RobotObservation(pose={"x": p.x, "y": p.y, "z": p.z, "r": p.rz})

    def home(self) -> None:
        self._require_cartesian().home()

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

> `get_flange_pose`/`move_to_flange`/`set_end_effector`/`grab_rgb` 由 `BaseRobotEnv` 默认委托给
> `low_level`，ScaraEnv 无需实现。`home` 是抽象方法——每个本体都要自己交代安全姿态，机械臂委托给驱动，
> 移动本体则写自己的序列。`home_pose`/`tool_offset_mm` 已暴露为属性。

## 5. 组合 Api

每个方法用 `@implements(SPEC)` 绑定共享动作词表里的一条——动作名、能力门、参数与 effects 都来自
`api/actions.py`，不由这个类自己讲。没有本体差异的（`goto_xyzr` / 吸盘开关）转发给 `api.defaults`；
有差异的才写出方法体——这里就是把驱动的 `rz` 暴露成 SCARA 习惯的 `r` 字段：

```python
"""SCARA Api — 4-DOF + 吸盘。goto_xyzr / 吸盘开关转发 api.defaults。"""
from typing import Literal

from jiuwensymbiosis.api import defaults
from jiuwensymbiosis.api.actions import (
    ACTIVATE_SUCTION,
    GET_HOME_POSE,
    GET_POSE,
    GOTO_XYZR,
    implements,
)
from jiuwensymbiosis.api.base import BaseRobotApi


class ScaraApi(BaseRobotApi):
    """4-DOF SCARA + 吸盘末端。仅 get_pose/get_home_pose 有本体差异（'r' 字段命名）。"""

    @implements(GOTO_XYZR)
    def goto_xyzr(
        self,
        x: float,
        y: float,
        z: float,
        r: float | None = None,
        orientation_policy: Literal["top_down", "preserve"] = "top_down",
    ) -> None:
        return defaults.goto_xyzr(
            self, x, y, z, r, orientation_policy=orientation_policy
        )

    @implements(ACTIVATE_SUCTION)
    def activate_suction(self) -> dict:
        return defaults.activate_suction(self)

    @implements(GET_POSE)
    def get_pose(self) -> dict:
        p = self.env.get_flange_pose()
        return {"x": p.x, "y": p.y, "z": p.z, "r": p.rz}

    @implements(GET_HOME_POSE)
    def get_home_pose(self) -> dict:
        hp = self.env.home_pose
        return {"x": hp.x, "y": hp.y, "z": hp.z, "r": hp.rz}
```

`ScaraApi` 不自己声明 capability——`BaseRobotApi.capabilities` 由它实现的那些 spec 反推出来。

## 6. 组装 Session

```python
"""build_my_scara_session"""
from pathlib import Path

from jiuwensymbiosis.adapters._common.builder import make_builder
from jiuwensymbiosis.adapters.my_scara.config import ScaraConfig
from jiuwensymbiosis.adapters.my_scara.env import ScaraEnv
from jiuwensymbiosis.adapters.my_scara.api import ScaraApi

build_my_scara_session = make_builder(
    ScaraConfig, ScaraEnv, ScaraApi,
    resource_keys=lambda cfg: (f"serial:{Path(cfg.serial_port).expanduser().resolve()}",),
)
```

在 `adapters/my_scara/__init__.py` 导出 `build_my_scara_session`，与包名一致。`resource_keys(cfg)` 声明实际命令端点；同一串口的路径别名应生成相同键。这里的 Mock Driver 不会打开串口，仍声明身份以验证完整的接入契约。builder 只装配 Session，不取得资源锁；Runtime 或官方 CLI 的 `admitted_session` 负责准入。

## 7. 编写 YAML 配置

将下面配置保存为 `configs/my_scara/default.yaml`，并复制一份到适配器的 `config_template.yaml`。

```yaml
# SCARA 吸盘机械臂配置
adapter: my_scara
name: "scara"
serial_port: "/dev/ttyUSB0"     # [必填] 串口路径
z_min_safe_mm: 30.0             # [选填] 安全Z下限
tool_offset_mm: 0.0             # [选填] 工具偏移
home_xyzr: [200.0, 0.0, 250.0, 0.0]  # [选填] Home 位姿 (x,y,z,r)
```

## 8. 验证并进入生产移植

```bash
python scripts/validate_adapter.py --module jiuwensymbiosis.adapters.my_scara
python scripts/smoke_test_adapter.py --module jiuwensymbiosis.adapters.my_scara --config configs/my_scara/default.yaml
```

静态验证检查目录、签名和能力对齐；冒烟测试向 Env 注入桩驱动（不连接真实硬件）、调用生成工具并检查返回值可序列化。至此本教程目标已经完成。替换为真实驱动前，继续阅读[移植机器人硬件适配器](../how-to/port-hardware-adapter.md)中的厂商 SDK、坐标几何、工作空间、安全和真机验收要求。

资源身份也是完成标准：S-17 检查工厂与资源回调接口，smoke 再用实际配置验证设备键。没有可推导设备键时须显式提供顶层 `physical_device_id`；相机或检测服务的键不能替代机器人的身份。若配置新增标定、URDF 等路径字段，在 Config 上声明 `path_fields: ClassVar[tuple[str, ...]]`，并沿用 `load_yaml_config`；Runtime 和 smoke 会使用同一解析规则，缺失文件也按来源目录定位。
