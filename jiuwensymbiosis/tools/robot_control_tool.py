# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``RobotControlTool`` —— openjiuwen ``Tool`` 抽象的子类，作为单一入口派发到 api。

与 ``build_robot_tools``（把每个 ``@robot_tool`` 方法各包成一个 ``LocalFunction``）
的关系是**互补**：

- ``build_robot_tools``：多个 tool 展开到 OpenAI ``tools`` 字段，让 LLM 直接看到每
  个动作的 schema。适合 mode="tool" / 工具数量较少 / prompt 容量充裕的场景。
- ``RobotControlTool``：单一入口 + ``action`` 字段派发。适合配 SKILL.md 走 workflow
  式控制 / 多机器人共享一个 prompt 入口 / 想缩短 tool list。

两者并存：``build_robot_agent(..., enable_skill=True)`` 会同时挂上两边。

实现要点（避免 openjiuwen 元类 ``_ToolMeta`` 的坑）：

- ``__init__`` 必须先把 ``self._api`` / ``self._action_index`` 赋好 → 再 ``super
  ().__init__(card)`` → 不能在元类包装完成之后才补字段（元类 ``__call__`` 会
  立刻访问 ``instance.card``）。
- ``stream`` 必须是 async-generator（保留 ``if False: yield None``），让元类的
  ``inspect.isasyncgenfunction`` 检测命中正确分支。
"""

from __future__ import annotations

import inspect
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from jiuwensymbiosis.agent.abstractions import Tool, ToolCard, ToolOutput


def _build_action_index(api: Any, env: Any = None) -> dict[str, Callable[..., Any]]:
    """扫描 ``type(api).__mro__`` 收集 ``@robot_tool`` 标注的方法，按 ``meta.name`` 索引。

    与 ``builder.py`` 的扫描策略一致：

    - ``seen`` 按 ``attr_name`` 去重（子类 override 优先）。
    - capability gate 与 ``build_robot_tools`` 一致：传入 ``env`` 时使用
      ``api.capabilities & env.capabilities``，否则仅使用 API 能力。

    返回 ``{action_name: bound_method}``。
    """
    api_caps = frozenset(getattr(api, "capabilities", None) or frozenset())
    effective_caps = api_caps
    if env is not None:
        env_caps = frozenset(getattr(env, "capabilities", None) or frozenset())
        effective_caps = api_caps & env_caps
    index: dict[str, Callable[..., Any]] = {}
    seen: set[str] = set()
    api_type = type(api)
    for cls in type(api).__mro__:
        for attr_name, attr_value in cls.__dict__.items():
            if attr_name in seen:
                continue
            if not callable(attr_value):
                continue
            meta = getattr(attr_value, "__robot_tool__", None)
            if meta is None:
                continue
            seen.add(attr_name)
            owning_capability = meta.capability
            if owning_capability is None:
                for owner in api_type.__mro__:
                    cap = owner.__dict__.get("capability")
                    if isinstance(cap, str) and attr_name in owner.__dict__:
                        owning_capability = cap
                        break
            if owning_capability and owning_capability not in effective_caps:
                continue
            index[meta.name] = getattr(api, attr_name)
    return index


def _default_description() -> str:
    """Return the default tool description for RobotControlTool."""
    return (
        "统一机器人控制入口：通过 ``action`` 字段派发到具体能力方法（运动 / 视觉 / 抓取）。"
        "调用前请先阅读 visual_pick 等 SKILL.md 文档，了解可用的 action 名与 params 规约；"
        "未知 action 或 params 不匹配会返回 success=False 而非抛异常。"
    )


def _default_input_params() -> dict[str, Any]:
    """Return the default JSON-Schema input params for RobotControlTool."""
    return {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": (
                    "要执行的动作名。可用值因 robot 能力而异；常见：home / get_pose / "
                    "goto_xyzr / analyze_scene / pixel_to_base_xyz / activate_suction / "
                    "deactivate_suction / open_gripper / close_gripper。完整列表见 SKILL.md。"
                ),
            },
            "params": {
                "type": "object",
                "description": "传给该 action 的关键字参数 dict；为空可省略。",
            },
        },
        "required": ["action"],
    }


class RobotControlTool(Tool):
    """openjiuwen ``Tool`` 派发器（线上路径）。"""

    def __init__(
        self,
        api: Any,
        *,
        env: Any = None,
        name: str = "robot_control",
        description: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        self._api = api
        self._action_index: dict[str, Callable[..., Any]] = _build_action_index(api, env=env)
        tool_id = f"{name}_{agent_id}" if agent_id else f"{name}_{uuid.uuid4().hex}"
        card = ToolCard(
            id=tool_id,
            name=name,
            description=description or _default_description(),
            input_params=_default_input_params(),
        )
        super().__init__(card)

    @property
    def available_actions(self) -> list[str]:
        """Return sorted list of available action names."""
        return sorted(self._action_index.keys())

    async def invoke(self, inputs: dict[str, Any], **kwargs: Any) -> ToolOutput:
        """Dispatch an action to the corresponding api method."""
        payload = inputs or {}
        action = payload.get("action")
        params = payload.get("params") or {}
        if not isinstance(action, str) or not action:
            return ToolOutput(
                success=False,
                error="action is required and must be a non-empty string",
            )
        if not isinstance(params, dict):
            return ToolOutput(
                success=False,
                error=f"params must be an object, got {type(params).__name__}",
            )
        method = self._action_index.get(action)
        if method is None:
            return ToolOutput(
                success=False,
                error=(f"unknown action '{action}'; available={self.available_actions}"),
            )
        try:
            result = method(**params)
            if inspect.isawaitable(result):
                result = await result
            return ToolOutput(
                success=True,
                data={"action": action, "result": result},
            )
        except TypeError as exc:
            return ToolOutput(
                success=False,
                error=f"bad params for '{action}': {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - convert to ToolOutput
            return ToolOutput(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def stream(self, inputs: dict[str, Any], **kwargs: Any) -> AsyncIterator[Any]:
        """Stream stub — kept as async-generator for openjiuwen meta-class compat."""
        if False:
            yield None
