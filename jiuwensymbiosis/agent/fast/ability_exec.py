# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Ability-manager-backed executor — run fast-path ops through the agent's rails.

This is what unifies the fast and agent paths. Instead of fast calling api
methods directly (bypassing the rail stack), each primitive op is dispatched via
the SAME ``agent.ability_manager.execute`` the agent loop uses — so SafetyRail,
VisualFeedbackRail and RecoveryRail all fire, with NO per-step LLM call.

The op is wrapped as a ``robot_control`` ToolCall ``{action, params}`` (the same
shape the agent's LLM emits); SafetyRail unwraps and bounds-checks it, the
``RobotControlTool`` dispatches it, and rails wrap the call. Returns the same
``{ok, result?, reason?}`` contract the runner's ``Executor`` expects.

openjiuwen symbols are imported lazily so the slow path never pays for them.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def build_ability_executor(agent: Any) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    """Build an ``Executor`` that runs ops via ``agent.ability_manager.execute``.

    Args:
        agent: a built ``DeepAgent`` (from ``build_robot_agent``). Must expose
            ``ability_manager``, ``react_agent`` and ``loop_session``.

    Returns:
        ``run(op, params) -> {ok, result?, reason?}`` — dispatches one op through
        the agent's rails + tool index, no LLM.
    """
    from openjiuwen.core.foundation.llm.schema.tool_call import ToolCall
    from openjiuwen.core.single_agent.rail.base import AgentCallbackContext

    ability_manager = agent.ability_manager
    react_agent = agent.react_agent
    session = agent.loop_session

    async def _execute(op: str, params: dict[str, Any]) -> dict[str, Any]:
        tc = ToolCall(
            id=f"fast-{op}",
            type="function",
            name="robot_control",
            arguments=json.dumps({"action": op, "params": params}),
        )
        # A fresh per-call context whose .agent carries the registered rails
        # (ctx.fire → react_agent.agent_callback_manager → rails).
        ctx = AgentCallbackContext(agent=react_agent, session=session)
        results = await ability_manager.execute(ctx, [tc], session)
        output = results[0][0]  # ToolOutput
        if not getattr(output, "success", False):
            return {"ok": False, "reason": getattr(output, "error", "tool failed")}
        # RobotControlTool wraps the api return under data["result"].
        data = getattr(output, "data", None) or {}
        return {"ok": True, "result": data.get("result", data)}

    def run(op: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            return asyncio.run(_execute(op, params))
        except Exception as exc:  # noqa: BLE001 - surface as structured failure
            logger.warning("[ability_exec] %s(%s) raised: %s", op, params, exc)
            return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}

    return run
