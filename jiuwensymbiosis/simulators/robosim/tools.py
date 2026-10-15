"""Explicit workspace tools; these are not shared robot ActionSpecs."""

from __future__ import annotations

from typing import Any

from jiuwensymbiosis.agent.abstractions import LocalFunction, ToolCard
from jiuwensymbiosis.simulators.robosim.workspace import RobosimWorkspace


def build_workspace_tools(workspace: RobosimWorkspace, *, allow_load: bool = True) -> list[Any]:
    """Expose named queries and optional empty-world loading, never arbitrary HTTP."""
    definitions = [
        ("robosim_status", "查询仿真健康、服务/机器人实例身份、在途或结果未知的命令。", workspace.refresh),
        ("robosim_list_robots", "列出后端机器人规格；当前操控适配仅支持所选 MOZ1 单臂。", workspace.list_robots),
        ("robosim_robot_spec", "查询 MOZ1 的能力与执行资源。", workspace.robot_spec),
        ("robosim_scene_state", "查询场景登记状态，返回值不是实时 PhysX 位姿。", workspace.scene_state),
        ("robosim_observe", "读取所选机械臂的实际关节、TCP、坐标变换与模拟时间。", workspace.observe),
    ]
    if allow_load:
        definitions.append(
            ("robosim_load_robot", "仅在空仿真加载 MOZ1；已有 MOZ1 保留，其他机器人拒绝替换。", workspace.load_robot)
        )
    return [
        LocalFunction(
            card=ToolCard(
                name=name,
                description=description,
                input_params={"type": "object", "properties": {}, "additionalProperties": False},
            ),
            func=func,
        )
        for name, description, func in definitions
    ]
