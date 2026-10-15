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

from jiuwensymbiosis.errors import error_code

logger = logging.getLogger(__name__)


def _failure_reason(output: Any, result_pair: Any) -> str:
    """Best available failure text for a call that produced no successful ToolOutput.

    ``ability_manager.execute`` returns ``(None, ToolMessage)`` when the call raised
    before a ToolOutput existed (unknown action, malformed args, a rail that aborts).
    Falling back to the ToolMessage keeps the real cause instead of "tool failed".
    """
    error = getattr(output, "error", None)
    if error:
        return str(error)
    content = getattr(result_pair[1] if len(result_pair) > 1 else None, "content", "")
    return str(content) or "tool failed"


def _failure_code(output: Any) -> str:
    """The failure's machine code, parked in ``ToolOutput.data`` by RobotControlTool."""
    data = getattr(output, "data", None)
    code = data.get("error_code") if isinstance(data, dict) else None
    return str(code) if code else ""


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
        output = results[0][0]  # ToolOutput, or None when the call never produced one
        if not getattr(output, "success", False):
            # The ability manager has already fired the exception hooks, so
            # RecoveryRail (when applicable) owns recovery for this failure.
            # The fast runner must not immediately repeat release/home.
            data = getattr(output, "data", None) or {}
            return {
                "ok": False,
                "reason": _failure_reason(output, results[0]),
                "error_code": _failure_code(output),
                "recovery_managed": True,
                "outcome_unknown": isinstance(data, dict) and data.get("outcome_unknown") is True,
                "skip_recovery": isinstance(data, dict) and data.get("skip_recovery") is True,
            }
        # RobotControlTool wraps the api return under data["result"].
        data = getattr(output, "data", None) or {}
        result = data.get("result", data)
        # A tool that ran without raising can still report business ok=False
        # (e.g. approach_for_grasp nav_failed/too_close); surface it as a step
        # failure like direct_executor so the runner aborts instead of grasping
        # an unreached target. No exception fired, so RecoveryRail did not run —
        # leave recovery_managed unset for the runner's own safe retreat.
        ok = result.get("ok", True) if isinstance(result, dict) else True
        if not ok:
            return {"ok": False, "reason": result.get("reason") or f"{op} reported not-ok", "result": result}
        return {"ok": True, "result": result}

    def run(op: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            return asyncio.run(_execute(op, params))
        except Exception as exc:  # noqa: BLE001 - surface as structured failure
            logger.warning("[ability_exec] %s(%s) raised: %s", op, params, exc)
            return {
                "ok": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "error_code": error_code(exc),
                "outcome_unknown": bool(getattr(exc, "outcome_unknown", False)),
                "skip_recovery": bool(getattr(exc, "skip_recovery", False)),
            }

    return run
