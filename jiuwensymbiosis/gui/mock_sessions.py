# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""无硬件模拟会话 + 脚本化模拟模型 —— 让 GUI 在"模拟优先"下端到端跑通。

两块内容:

1. :func:`build_mock_robot_session` —— 用 ``MockArmEnv`` + 通用能力 Mixin 组装
   一个可运行的模拟机械臂会话(等价于 ``examples/piper_pick_demo.py`` 的 mock
   分支,但收进库内、可被 GUI 与测试复用,不依赖 ``examples/`` 或 ``piper_sdk``)。

2. :class:`ScriptedMockModelClient` / :func:`build_scripted_mock_model` ——
   ``jiuwensymbiosis.agent.mock_model`` 的离线模型只回一句话、**不调用任何工具**,
   于是运行页的步骤时间线会是空的。脚本化模型改为按预设序列**逐轮返回工具调用**
   (第 1 轮 home、第 2 轮识别、第 3 轮移动……最后返回文本收尾),使模拟运行也能
   完整演示"实时看得懂的执行过程",无需真实 LLM 或硬件。

脚本序列本身是数据,由 ``registry`` 里的任务提供,故本模块与具体本体/任务解耦。
"""

from __future__ import annotations

import json
from typing import Any

# openjiuwen 消息/工具调用 schema(经 abstractions 之外的原始 schema 模块)。
from openjiuwen.core.foundation.llm.schema.message import AssistantMessage
from openjiuwen.core.foundation.llm.schema.tool_call import ToolCall

from jiuwensymbiosis.agent.abstractions import Model
from jiuwensymbiosis.agent.mock_model import MockModelClient, build_mock_model
from jiuwensymbiosis.agent.session import RobotSession
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.api.decorators import robot_tool
from jiuwensymbiosis.api.mixins import MotionMixin, ParallelGripperMixin, VisionMixin
from jiuwensymbiosis.env.mock import MockArmEnv

__all__ = [
    "MockPiperApi",
    "build_mock_robot_session",
    "ScriptedMockModelClient",
    "build_scripted_mock_model",
    "count_tool_messages",
]


class MockPiperApi(MotionMixin, ParallelGripperMixin, VisionMixin, BaseRobotApi):
    """无硬件环境下的机械臂模拟 API(镜像 demo 的 ``_MockPiperApi``)。

    运动/夹爪/视觉工具都委托给 ``MockArmEnv`` 的内存状态;视觉工具返回基于
    home 位姿的合成结果,足以驱动一条完整的拾放序列。
    """

    # 把基类 env(BaseRobotEnv)收窄到 mock 子类型,便于访问 move / set_suction 等。
    env: MockArmEnv

    @robot_tool(desc="home", tags=["motion"])
    def home(self) -> None:
        """回归机械臂初始位姿。"""
        self.env.home()

    @robot_tool
    def get_pose(self) -> dict:
        """获取当前位姿。"""
        return self.env.get_observation().pose or {}

    @robot_tool
    def get_home_pose(self) -> dict:
        """获取初始位姿。"""
        return self.env.home_pose

    @robot_tool(tags=["motion"])
    def goto_xyzr(self, x: float, y: float, z: float, r: float | None = None) -> None:
        """移动到指定坐标 (x, y, z, r)。"""
        self.env.move(x, y, z, r)

    @robot_tool(tags=["grasp"])
    def close_gripper(self, force_n: float | None = None) -> dict:
        """关闭夹爪(模拟吸合)。"""
        self.env.set_suction(True)
        return {"ok": True, "state": "closed"}

    @robot_tool(tags=["grasp"])
    def open_gripper(self, width_mm: float = 70.0) -> dict:
        """打开夹爪(模拟释放)。"""
        self.env.set_suction(False)
        return {"ok": True, "state": "open"}

    @robot_tool
    def get_grasp_info_simple(self, object_name: str) -> dict:
        """获取抓取目标的位姿信息(返回模拟值)。"""
        hp = self.env.home_pose
        return {
            "ok": True,
            "position": [hp["x"] + 30, hp["y"], hp["z"] - 200],
            "score": 0.9,
            "pixel_uv": [320, 240],
            "depth_m": 0.20,
        }

    @robot_tool
    def pixel_to_base_xyz(self, u: float, v: float, depth_m: float) -> dict:
        """将像素坐标 + 深度转换为基坐标系下的三维坐标(返回模拟值)。"""
        hp = self.env.home_pose
        return {"x": hp["x"] + 30, "y": hp["y"], "z": hp["z"] - 200}


def build_mock_robot_session(name: str = "piper_mock") -> RobotSession:
    """组装一个无硬件的模拟机械臂会话(未连接;调用方用 ``with session:``)。"""
    env = MockArmEnv()
    api = MockPiperApi(env)
    return RobotSession(env=env, api=api, name=name)


# ------------------------------------------------------------------ 脚本化模型


def count_tool_messages(messages: Any) -> int:
    """统计对话里已产生的工具消息数,用于判断当前处于脚本的第几步。

    每返回一个工具调用,下一轮 ``messages`` 里就多一条 role=="tool" 的消息,
    因此"已完成的工具消息数"恰好是下一步在脚本中的索引。
    """
    if not isinstance(messages, list):
        return 0
    count = 0
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "tool":
            count += 1
    return count


def _available_tool_names(tools: Any) -> set[str]:
    """从 ``invoke`` 收到的工具定义列表里提取工具名(兼容 dict 与对象两种形态)。"""
    names: set[str] = set()
    for tool in tools or []:
        name: Any = None
        if isinstance(tool, dict):
            fn = tool.get("function")
            name = fn.get("name") if isinstance(fn, dict) else None
            name = name or tool.get("name")
        else:
            fn = getattr(tool, "function", None)
            name = getattr(fn, "name", None) or getattr(tool, "name", None)
        if isinstance(name, str):
            names.add(name)
    return names


class ScriptedMockModelClient(MockModelClient):
    """离线模型:按预设脚本逐轮返回工具调用,最后返回一句收尾文本。

    ``steps`` 每项形如 ``{"tool": "goto_xyzr", "args": {...}, "say": "..."}``。
    只有在当轮 ``tools`` 里实际可用的工具才会被排进计划(未注册的工具自动跳过,
    绝不因脚本与实际能力不符而崩溃)。脚本走完即返回 ``final_text`` 收尾。
    """

    # 区别于父类的 "mock",避免在客户端注册表里覆盖同名项。
    __client_name__ = "mock_scripted"

    def __init__(
        self,
        model_config: Any,
        model_client_config: Any,
        *,
        steps: list[dict] | None = None,
        final_text: str = "任务完成。",
    ) -> None:
        """记录脚本序列与收尾文本。"""
        super().__init__(model_config, model_client_config)
        self.steps: list[dict] = list(steps or [])
        self.final_text = final_text

    async def invoke(
        self,
        messages: str | list[Any] | list[dict],
        *,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> AssistantMessage:
        """返回脚本的下一步(工具调用),或走完后返回收尾文本。"""
        available = _available_tool_names(tools)
        plan = [s for s in self.steps if s.get("tool") in available] if available else list(self.steps)
        turn = count_tool_messages(messages)
        if turn < len(plan):
            step = plan[turn]
            call = ToolCall(
                id=f"call-{turn}",
                type="function",
                name=str(step["tool"]),
                arguments=json.dumps(step.get("args", {}), ensure_ascii=False),
                index=0,
            )
            return AssistantMessage(content=str(step.get("say", "")), tool_calls=[call])
        return AssistantMessage(content=self.final_text)


def build_scripted_mock_model(steps: list[dict], *, final_text: str = "任务完成。") -> Model:
    """构建一个由脚本驱动的离线 ``Model``。

    先建普通 mock ``Model``(拿到框架构造好的 config),再把它内部的 client 换成
    脚本化实例。``Model`` 无公开的 client setter,仅此一处替换。
    """
    model = build_mock_model()
    scripted = ScriptedMockModelClient(
        model.model_config,
        model.model_client_config,
        steps=steps,
        final_text=final_text,
    )
    # Model 无公开的 client setter,只能按名注入;变量属性名避免直接私有赋值/常量 setattr。
    client_attr = "_client"
    setattr(model, client_attr, scripted)
    return model
