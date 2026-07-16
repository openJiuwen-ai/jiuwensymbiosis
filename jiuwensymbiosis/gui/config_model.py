# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""任务配置的单一真源:dict ↔ YAML 文本 ↔ 分组表单 的双向绑定。

设计原则(见实现计划"任务配置页"):不是每个命令行参数都做成一行控件。常用
设置按类别分组放进统一表单;其余字段用"原始 YAML"编辑器兜底。dict 是单一真
源,YAML 编辑器是它的序列化视图,表单控件按 ``FieldSpec.path`` 绑定到 dict 的
点分路径。

本模块纯逻辑(不依赖 Qt),便于单元测试往返一致性与校验。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import yaml

__all__ = [
    "FieldSpec",
    "FIELD_GROUPS",
    "GROUP_ORDER",
    "ConfigModel",
]

FieldKind = Literal["text", "str", "int", "float", "bool", "choice"]

# 分组顺序(表单左侧小导航按此顺序展示)。
GROUP_ORDER: tuple[str, ...] = ("基础", "执行方式", "安全与反馈", "机器人参数", "模型")


@dataclass(frozen=True)
class FieldSpec:
    """一个可编辑配置项的声明。

    Attributes:
        path: 在配置 dict 中的点分路径,例如 ``env.cfg.prompt``。
        label: 中文标签。
        kind: 控件类型(决定 GUI 用哪种输入控件)。
        group: 所属分组(见 ``GROUP_ORDER``)。
        choices: ``kind="choice"`` 时的 ``(底层值, 显示短语)`` 对;界面显示短语,
            存回 dict 的是底层值。
        help: 一句话说明(悬停提示)。
        default: 字段缺省值(路径不存在时表单显示它;应与框架真实默认一致)。
        min_value / max_value: 数字类字段(int/float)的上下限,喂给 ``ui.number`` 的
            ``min`` / ``max`` —— 上下箭头步进不会越界(如温度限 [0, 2])。
        step: 数字类字段每次步进的增量(缺省走控件默认 1;如温度用 0.1)。
        on_value / off_value: ``kind="bool"`` 时若给出,复选框存的不是 True/False
            而是这两个值(如 exec_mode 的 ``"fast"`` / ``"agent"``)。
        disable_in_mock: 模拟模式下置灰该字段(如 fast 需真实模型+硬件)。
    """

    path: str
    label: str
    kind: FieldKind
    group: str
    choices: tuple[tuple[str, str], ...] = ()
    help: str = ""
    default: Any = None
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    on_value: Any = None
    off_value: Any = None
    disable_in_mock: bool = False


# 常用设置的分组表单。刻意只挑"用户常改、能看懂"的字段;其余走原始 YAML。
# 各字段 default 与框架真实默认一致,使界面显示 = 实际运行行为。
FIELD_GROUPS: tuple[FieldSpec, ...] = (
    # -- 基础 --
    FieldSpec("env.cfg.prompt", "任务指令", "text", "基础", help="用自然语言描述要完成的任务;留空则用内置默认。"),
    # -- 执行方式 --
    FieldSpec(
        "agent.mode",
        "智能体模式",
        "choice",
        "执行方式",
        choices=(("hybrid", "自动选择"), ("tool", "逐步工具调用"), ("code", "构建程序批量执行")),
        help=(
            "自动选择:AI 自行决定用哪种;"
            "逐步工具调用:每步只下一个明确指令(最稳);"
            "构建程序批量执行:用一段程序一次安排多步(含循环/判断)。"
        ),
        default="hybrid",
    ),
    FieldSpec(
        "agent.max_iterations", "最大步数", "int", "执行方式", help="智能体循环的上限步数。", default=15, min_value=1
    ),
    FieldSpec(
        "agent.enable_skill",
        "启用技能工作流",
        "bool",
        "执行方式",
        help="开启后由 visual_pick / visual_place 技能文档驱动抓放流程。",
        default=False,
    ),
    FieldSpec(
        "agent.exec_mode",
        "快速模式(fast)",
        "bool",
        "执行方式",
        help=(
            "开(默认):开头用一次 LLM 规划出整条动作序列,之后不再逐步调用 LLM——更快、"
            "可重复,但适应性弱,且需真实模型与(真机)伺服。关:逐步智能体,每步问一次"
            "LLM。模拟模式不可用(无真实模型),届时自动回退逐步。"
        ),
        default="fast",
        on_value="fast",
        off_value="agent",
        disable_in_mock=True,
    ),
    # -- 安全与反馈 --
    FieldSpec(
        "agent.enable_safety",
        "运动前边界检查",
        "bool",
        "安全与反馈",
        help="移动前校验 Z 下限与 XY 工作空间。",
        default=True,
    ),
    FieldSpec(
        "agent.enable_recovery",
        "失败自动回零",
        "bool",
        "安全与反馈",
        help="动作失败时自动回到安全位姿并松开末端。",
        default=True,
    ),
    FieldSpec(
        "agent.enable_visual_feedback",
        "每步拍照校验",
        "bool",
        "安全与反馈",
        help="每次动作后拍一帧供模型核验。",
        default=True,
    ),
    FieldSpec(
        "agent.enable_tracing",
        "记录执行轨迹",
        "bool",
        "安全与反馈",
        help="记录每步用于「历史」页回放。",
        default=False,
    ),
    # -- 机器人参数 --
    FieldSpec("env.cfg.low_level.move_speed", "运动速度", "int", "机器人参数", help="真机上建议从小值起步。"),
    FieldSpec("env.cfg.low_level.gripper_open_mm", "夹爪开度(mm)", "float", "机器人参数"),
    FieldSpec("env.cfg.low_level.gripper_effort", "夹爪力度", "int", "机器人参数"),
    FieldSpec(
        "env.cfg.low_level.tool_offset_mm", "工具偏置(mm)", "float", "机器人参数", help="末端工具相对法兰的长度。"
    ),
    FieldSpec(
        "env.cfg.low_level.camera_serial",
        "相机序列号",
        "str",
        "机器人参数",
        help=(
            "腕部 RealSense 相机的序列号。真机视觉任务(识别/定位物体)必填——留空则不会启用"
            "相机,视觉工具会返回 no_camera。模拟模式无需设置。"
        ),
    ),
    # -- 模型(模拟模式下由界面置灰) --
    FieldSpec("model.model_name", "模型名称", "str", "模型"),
    FieldSpec("model.api_base", "服务端点", "str", "模型", help="不要包含 /chat/completions。"),
    FieldSpec("model.api_key", "API Key", "str", "模型", help="留空表示端点无需鉴权。"),
    FieldSpec("model.temperature", "采样温度", "float", "模型", min_value=0.0, max_value=2.0, step=0.1),
)


_MISSING = object()


class ConfigModel:
    """一份任务配置的可编辑封装;dict 为单一真源。

    表单按 ``FieldSpec.path`` 读写这里的点分路径;"原始 YAML"标签页则整体
    ``to_yaml()`` / ``replace_from_yaml()``。两者共享同一个 ``data``。
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        """用一个配置 dict 初始化(浅拷贝顶层引用,直接持有传入结构)。"""
        self.data: dict[str, Any] = data if isinstance(data, dict) else {}

    # ------------------------------------------------------------- 构造
    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ConfigModel:
        """从内存 dict 构造。"""
        return cls(dict(data) if isinstance(data, dict) else {})

    @classmethod
    def from_yaml_text(cls, text: str) -> ConfigModel:
        """从 YAML 文本构造;非法 YAML 或顶层非映射时抛 ``ValueError``。"""
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"YAML 解析失败:{exc}") from exc
        if parsed is None:
            parsed = {}
        if not isinstance(parsed, dict):
            raise ValueError("配置顶层必须是一个映射(键值对),而不是列表或标量。")
        return cls(parsed)

    # ------------------------------------------------------------- 点分路径读写
    def get(self, path: str, default: Any = None) -> Any:
        """按点分路径读取;任一层缺失则返回 ``default``。"""
        node: Any = self.data
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def set(self, path: str, value: Any) -> None:
        """按点分路径写入,自动创建中间字典。"""
        keys = path.split(".")
        node = self.data
        for key in keys[:-1]:
            child = node.get(key)
            if not isinstance(child, dict):
                child = {}
                node[key] = child
            node = child
        node[keys[-1]] = value

    def patch_detector(self, **fields: Any) -> bool:
        """把检测器字段(如 ``gdino_model_id`` / ``hf_endpoint``)写进 ``api_servers`` 里的检测器项。

        供运行页的一键修复把改动**沉淀进配置**(便于导出 / 另存为新任务);运行期本身另由
        环境变量立即生效。检测器项按 ``_target_`` 含 ``gdino`` 识别。返回是否写入成功。
        """
        servers = self.data.get("api_servers")
        if not isinstance(servers, list):
            return False
        for server in servers:
            # 与 piper 配置识别检测器项的方式一致(_target_ 含 grounding_dino 或 gdino)。
            target = str(server.get("_target_", "")).lower() if isinstance(server, dict) else ""
            if "grounding_dino" in target or "gdino" in target:
                server.update(fields)
                return True
        return False

    # ------------------------------------------------------------- YAML 视图
    def to_yaml(self) -> str:
        """序列化为 YAML 文本(保留中文,不排序键)。"""
        return str(yaml.safe_dump(self.data, allow_unicode=True, sort_keys=False, default_flow_style=False))

    def replace_from_yaml(self, text: str) -> None:
        """用 YAML 文本整体替换 ``data``;非法时抛 ``ValueError`` 且不改动原数据。"""
        other = ConfigModel.from_yaml_text(text)
        self.data = other.data

    # ------------------------------------------------------------- 表单绑定
    def field_value(self, spec: FieldSpec) -> Any:
        """取某个表单字段当前值,缺失时回落到 ``spec.default``。"""
        val = self.get(spec.path, _MISSING)
        if val is _MISSING:
            return spec.default
        return val

    # ------------------------------------------------------------- 校验
    def validate(self) -> list[str]:
        """返回一组人类可读的告警(不阻断运行,供界面提示)。"""
        warnings: list[str] = []
        speed = self.get("env.cfg.low_level.move_speed")
        if isinstance(speed, int | float) and not (0 < speed <= 100):
            warnings.append(f"运动速度 {speed} 超出常规范围 (0, 100]。")
        temp = self.get("model.temperature")
        if isinstance(temp, int | float) and not (0.0 <= temp <= 2.0):
            warnings.append(f"采样温度 {temp} 超出常规范围 [0, 2]。")
        return warnings
