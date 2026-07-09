# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""把机器人工具调用翻译成"普通人看得懂"的中文短语与叙述。

运行页的核心诉求:不像命令行那样刷屏原始日志,而是把每一步工具调用显示为
一句人话。原始 ``tool_name`` / ``input_params`` 只在用户点开某步时展示。

本模块是纯逻辑(不依赖 Qt),便于单元测试。``UIBridgeRail`` 与运行页都调用
它来生成步骤标签和当前动作叙述。
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "unwrap_robot_control",
    "friendly_label",
    "narration",
    "FRAME_AFTER_TOOLS",
    "TOOL_LABELS",
]

# 工具名 → 中文短语。未知工具回落到工具名本身。
TOOL_LABELS: dict[str, str] = {
    "home": "回到初始位置",
    "goto_xyzr": "移动机械臂",
    "goto_pose": "移动机械臂",
    "get_pose": "读取当前位姿",
    "get_home_pose": "读取初始位姿",
    "get_grasp_info_simple": "识别并定位物体",
    "analyze_scene": "观察场景",
    "pixel_to_base_xyz": "计算物体坐标",
    "open_gripper": "张开夹爪",
    "close_gripper": "闭合夹爪抓取",
    "suction_on": "开启吸盘",
    "suction_off": "关闭吸盘",
}

# 这些工具执行后值得抓一帧相机画面刷新到界面(运动/抓取类)。
FRAME_AFTER_TOOLS: frozenset[str] = frozenset(
    {
        "home",
        "goto_xyzr",
        "goto_pose",
        "open_gripper",
        "close_gripper",
        "suction_on",
        "suction_off",
    }
)

# 从工具参数里提取"物体名"时依次尝试的键。
_OBJECT_KEYS = ("object_name", "chip_object_name", "slot_object_name", "target")


def unwrap_robot_control(tool_name: str, tool_args: Any) -> tuple[str, dict]:
    """把 ``robot_control{action, params}`` 还原为真实动作名与参数。

    ``RobotControlTool`` 把所有动作收敛到单一 ``robot_control`` 入口,其它 rail
    都会解包,这里同样解包,使步骤显示为 ``goto_xyzr`` 而非 ``robot_control``。
    与 ``jiuwensymbiosis.agent.trace._unwrap_robot_control`` 语义一致。
    """
    # openjiuwen 在 before_tool_call 把 tool_args 作为 **JSON 字符串** 传入
    # (ToolCall.arguments 类型为 str);先解析,否则下方 isinstance(dict) 会把它丢成 {},
    # 界面上所有带参工具都显示「参数:{}」。与 agent.trace._unwrap_robot_control 一致。
    if isinstance(tool_args, str) and tool_args:
        try:
            parsed = json.loads(tool_args)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            tool_args = parsed
    args: dict = tool_args if isinstance(tool_args, dict) else {}
    if tool_name == "robot_control":
        action = args.get("action", "")
        params = args.get("params", {})
        if action:
            return str(action), params if isinstance(params, dict) else {}
    return tool_name, args


def _object_hint(params: dict) -> str | None:
    """从参数里找一个可读的"物体名",用于把标签写得更具体。"""
    for key in _OBJECT_KEYS:
        val = params.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def friendly_label(tool_name: str, tool_args: Any = None) -> str:
    """返回一步的简短中文标签,例如"识别并定位物体「black box」"。

    先解包 ``robot_control``,再查表;能提取到物体名时附在后面。
    """
    name, params = unwrap_robot_control(tool_name, tool_args)
    label = TOOL_LABELS.get(name, name)
    hint = _object_hint(params)
    if hint and name in ("get_grasp_info_simple", "analyze_scene"):
        return f"{label}「{hint}」"
    return label


def narration(tool_name: str, tool_args: Any = None) -> str:
    """返回"正在做什么"的一句话叙述,用于主视觉区下方的当前动作提示。"""
    return f"正在{friendly_label(tool_name, tool_args)}…"
