# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Spec — the answers that drive adapter generation, plus the question flow.

The wizard asks in the vendor's own terms (DOF, end-effector, camera...) and
``Spec`` translates those answers into capabilities / mixins / driver members.
Nothing here imports the framework, so it stays import-light and unit-testable.
"""

from __future__ import annotations

import keyword
import logging
import re
from dataclasses import dataclass
from typing import Callable, Optional

# Shared with ``main.py`` — same logger object, configured once in ``main()``
# with a raw ``%(message)s`` handler so wizard output looks exactly like print.
logger = logging.getLogger("new_adapter")

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

END_EFFECTORS = ("none", "parallel", "suction")
TOOL_GEOMETRIES = ("straight_down", "tilted")
CONNECTIONS = ("can", "serial", "tcp", "usb", "ros", "custom")

# Inputs that trigger a per-question explanation instead of an answer.
_HELP_KEYS = ("?", "？", "help", "帮助", "h")

_HELP_DOF = """\
自由度 = 机械臂能独立控制的运动维度数。
  4 (SCARA)：平面四轴，末端位姿 = x, y, z + 绕 Z 的旋转 r；常见于桌面贴片/分拣臂。
  6 (六轴) ：末端可任意姿态，位姿 = x, y, z + 三个姿态角 rx, ry, rz；最常见的工业/协作臂。
怎么确定：数一下有几个电机/关节，或看 SDK 手册里末端位姿是 4 个数还是 6 个数。
不确定就选 6（更通用，之后也能改）。"""

_HELP_JOINT = """\
关节空间运动 = 直接给每个关节的目标角度让它运动（区别于给末端 x,y,z 的笛卡尔运动）。
SDK 里若有类似 move_joint(角度列表) 的接口就选「是」；只有末端坐标接口就选「否」。
不确定先选「否」，以后需要再加。"""

_HELP_EE = """\
末端执行器 = 装在臂末端用来抓取的工具。
  none     ：没有/你自己控制，不生成抓取工具。
  parallel ：平行夹爪——两根手指开合夹住物体，适合有侧壁可夹的物体。
  suction  ：吸盘——靠真空吸住物体顶面，适合平整、轻、顶面可吸的物体。"""

_HELP_DETECTION = """\
目标检测 = 用相机+模型把「自然语言物体名（如 红色方块）」定位成可抓取的 3D 坐标。
需要：相机、部署 GroundingDINO+SAM2 检测服务、做过手眼标定。
只是想先把运动跑通，选「否」即可。"""

_HELP_CAMERA = """\
相机 = 臂上或工位有一台能取 RGB（可含深度）图像的相机。仅做运动控制可选「否」。"""

_HELP_TILT = """\
工具是否垂直向下安装：
  是 —— 工具笔直朝下，末端朝向与法兰一致（最常见）。
  否 —— 工具有倾斜角或横向偏移，需要在 api 里做 tip↔flange 的几何换算（参考 piper）。
不确定通常选「是」。"""

_HELP_CONNECTION = """\
连接方式 = 你的 Python 代码第一次接触硬件的入口。
  can    ：CAN 总线（当前向导会生成较完整模板：can_port/bitrate/SDK client 占位）。
  serial ：串口（目前生成空模板，后续会补更完整模板）。
  tcp    ：TCP/IP（目前生成空模板，后续会补更完整模板）。
  usb    ：USB/设备序列号（目前生成空模板，后续会补更完整模板）。
  ros    ：ROS/ROS2 节点、话题、服务或 action（目前生成空模板，后续会补更完整模板）。
  custom ：其他硬件 SDK/特殊连接方式；生成最空模板，由你完全填充。
不确定且机器人走 CAN，就选 can；否则选 custom。"""


def validate_name(name: str) -> Optional[str]:
    """Return an error string if ``name`` is not a usable package name, else None."""
    if not name:
        return "名字不能为空"
    if not _NAME_RE.match(name):
        return "只能用小写字母/数字/下划线，且以字母开头（如 my_robot）"
    if keyword.iskeyword(name):
        return f"'{name}' 是 Python 关键字，换一个"
    return None


@dataclass
class Spec:
    """All choices that determine the generated adapter."""

    name: str
    dof: int = 6  # 4 (SCARA x,y,z,r) | 6 (x,y,z,rx,ry,rz)
    joint: bool = False  # joint-space motion
    end_effector: str = "none"  # none | suction | parallel
    camera: bool = False  # raw RGB(+depth) stream
    detection: bool = False  # NL object detection → 3D grasp
    tool_geometry: str = "straight_down"  # straight_down | tilted
    connection: str = "can"  # can | serial | tcp | usb | ros | custom

    # ---- derived identifiers ------------------------------------------------

    @property
    def prefix(self) -> str:
        """CamelCase class prefix, e.g. ``my_robot`` → ``MyRobot``."""
        return "".join(part.capitalize() for part in self.name.split("_") if part)

    @property
    def builder_name(self) -> str:
        return f"build_{self.name}_session"

    @property
    def config_cls(self) -> str:
        return f"{self.prefix}Config"

    @property
    def env_cls(self) -> str:
        return f"{self.prefix}Env"

    @property
    def api_cls(self) -> str:
        return f"{self.prefix}Api"

    @property
    def driver_cls(self) -> str:
        return f"{self.prefix}Driver"

    # ---- derived capability model ------------------------------------------

    @property
    def capabilities(self) -> list[str]:
        """The capability strings this adapter declares, in a stable order."""
        caps = ["motion.cartesian"]
        if self.joint:
            caps.append("motion.joint")
        if self.end_effector == "suction":
            caps.append("grasp.suction")
        elif self.end_effector == "parallel":
            caps.append("grasp.parallel")
        if self.camera or self.detection:
            caps.append("vision.camera")
        if self.detection:
            caps.append("vision.detection")
        return caps

    @property
    def has_grasp(self) -> bool:
        return self.end_effector in ("suction", "parallel")

    @property
    def has_camera(self) -> bool:
        return self.camera or self.detection

    @property
    def rot_fields(self) -> list[str]:
        """Rotation field names by DOF: SCARA → [r]; 6-DoF → [rx, ry, rz]."""
        return ["r"] if self.dof == 4 else ["rx", "ry", "rz"]

    @property
    def pose_fields(self) -> list[str]:
        return ["x", "y", "z"] + self.rot_fields

    def normalized(self) -> "Spec":
        """Resolve implied flags (detection ⇒ camera) and return self."""
        if self.detection:
            self.camera = True
        return self


# ---------------------------------------------------------------------------
# Interactive question flow
# ---------------------------------------------------------------------------


def _show_help(text: str) -> None:
    for line in text.strip("\n").splitlines():
        logger.info(f"      │ {line}")


def _ask_str(
    prompt: str,
    default: str,
    validate: Optional[Callable[[str], Optional[str]]] = None,
    help_text: Optional[str] = None,
) -> str:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if help_text and raw in _HELP_KEYS:
            _show_help(help_text)
            continue
        value = raw or default
        if validate is not None:
            err = validate(value)
            if err is not None:
                logger.info(f"  ✗ {err}")
                continue
        return value


def _ask_bool(prompt: str, default: bool, help_text: Optional[str] = None) -> bool:
    hint = "Y/n" if default else "y/N"
    suffix = "（或 ? 看说明）" if help_text else ""
    while True:
        raw = input(f"{prompt} ({hint}): ").strip().lower()
        if help_text and raw in _HELP_KEYS:
            _show_help(help_text)
            continue
        if not raw:
            return default
        if raw in ("y", "yes", "是", "1"):
            return True
        if raw in ("n", "no", "否", "0"):
            return False
        logger.info(f"  ✗ 请输入 y 或 n{suffix}")


def _ask_choice(
    prompt: str, choices: tuple[str, ...], default: str, help_text: Optional[str] = None
) -> str:
    options = " / ".join(f"[{c}]" if c == default else c for c in choices)
    suffix = "（或 ? 看说明）" if help_text else ""
    while True:
        raw = input(f"{prompt} ({options}): ").strip().lower()
        if help_text and raw in _HELP_KEYS:
            _show_help(help_text)
            continue
        value = raw or default
        if value in choices:
            return value
        logger.info(f"  ✗ 请从 {', '.join(choices)} 中选择{suffix}")


def ask_interactive() -> Spec:
    """Walk the engineer through the choices and return a normalized Spec."""
    logger.info("=" * 60)
    logger.info(" jiuwensymbiosis 适配器生成向导")
    logger.info(" 用你自己的话回答几个问题，我来生成一致的骨架")
    logger.info("=" * 60)
    logger.info(" 提示：方括号 [] 里是默认值，直接回车即采用；任何一题不清楚就输入 ? 看说明。")
    name = _ask_str("适配器/机器人名字 (小写, 如 my_robot)", "my_robot", validate=validate_name)
    dof = int(_ask_choice("自由度 (4=SCARA / 6=六轴)", ("4", "6"), "6", help_text=_HELP_DOF))
    joint = _ask_bool("支持关节空间运动吗？", False, help_text=_HELP_JOINT)
    end_effector = _ask_choice(
        "末端执行器 (none=无/自控  parallel=平行夹爪·两指夹取  suction=吸盘·真空吸顶面)",
        END_EFFECTORS,
        "none",
        help_text=_HELP_EE,
    )
    detection = _ask_bool(
        "需要自然语言目标检测吗？(需检测服务+手眼标定)", False, help_text=_HELP_DETECTION
    )
    camera = (
        True if detection else _ask_bool("有相机可取 RGB 图像吗？", False, help_text=_HELP_CAMERA)
    )
    tilt = _ask_bool("工具是垂直向下安装的吗？(否=倾斜/有偏移几何)", True, help_text=_HELP_TILT)
    tool_geometry = "straight_down" if tilt else "tilted"
    connection = _ask_choice("硬件连接方式", CONNECTIONS, "can", help_text=_HELP_CONNECTION)
    spec = Spec(
        name=name,
        dof=dof,
        joint=joint,
        end_effector=end_effector,
        camera=camera,
        detection=detection,
        tool_geometry=tool_geometry,
        connection=connection,
    )
    return spec.normalized()
