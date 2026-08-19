# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``build_ability_executor`` must surface a tool's business-level ``ok=False``.

A tool that RAN without raising (``ToolOutput.success is True``) can still report
a business failure inside its return, e.g. ``approach_for_grasp -> {"ok": False}`` on
``nav_failed``/``too_close``. If the executor reported that as a step success, the
fast runner would march on to grasp/place on an unreached target (the observed
"detect box → skip to placement" bug). The executor must mirror ``direct_executor``
and propagate the inner ``ok`` — but, because no exception fired, RecoveryRail did
NOT run, so ``recovery_managed`` must stay unset (the runner does its own retreat).

The openjiuwen ``ToolCall`` / ``AgentCallbackContext`` constructors accept plain
namespaces, so a scripted ability_manager isolates the ok-extraction logic without
a real agent, rails, or LLM.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from jiuwensymbiosis.agent.fast.ability_exec import build_ability_executor


class _FakeAbilityManager:
    """Returns one scripted ToolOutput and records the dispatched action/params."""

    def __init__(self, output):
        self._output = output
        self.dispatched: list[dict] = []

    async def execute(self, ctx, tool_calls, session):
        self.dispatched.append(json.loads(tool_calls[0].arguments))
        return [[self._output]]


def _run_for(output):
    agent = SimpleNamespace(
        ability_manager=_FakeAbilityManager(output),
        react_agent=SimpleNamespace(),
        loop_session=SimpleNamespace(),
    )
    return build_ability_executor(agent)


def _tool_output(*, success, result=None, error=None):
    data = None if result is None else {"result": result}
    return SimpleNamespace(success=success, data=data, error=error)


class TestAbilityExecutorSurfacesBusinessOk:
    def test_business_ok_false_becomes_step_failure(self):
        run = _run_for(_tool_output(success=True, result={"ok": False, "reason": "nav_failed"}))
        res = run("approach_for_grasp", {})
        assert res["ok"] is False
        assert res["reason"] == "nav_failed"
        assert res["result"] == {"ok": False, "reason": "nav_failed"}
        # No exception fired → RecoveryRail did not run → runner owns the retreat.
        assert "recovery_managed" not in res

    def test_business_ok_false_without_reason_gets_fallback(self):
        run = _run_for(_tool_output(success=True, result={"ok": False}))
        res = run("approach_for_grasp", {})
        assert res["ok"] is False
        assert res["reason"] == "approach_for_grasp reported not-ok"

    def test_business_ok_true_passes_through(self):
        run = _run_for(_tool_output(success=True, result={"ok": True, "status": "in_band"}))
        assert run("approach_for_grasp", {}) == {"ok": True, "result": {"ok": True, "status": "in_band"}}

    def test_missing_ok_key_defaults_to_success(self):
        # A result without an explicit ok (e.g. approach_for_place {"status": "in_range"})
        # is a success, matching direct_executor's result.get("ok", True).
        run = _run_for(_tool_output(success=True, result={"status": "in_range"}))
        assert run("approach_for_place", {})["ok"] is True

    def test_rail_rejection_keeps_recovery_managed(self):
        # output.success False (a rail raised) is the pre-existing path: the ability
        # manager already fired exception hooks, so recovery_managed stays True.
        run = _run_for(_tool_output(success=False, error="SafetyRail: z below floor"))
        res = run("goto_xyzr", {"x": 0, "y": 0, "z": -1})
        assert res["ok"] is False
        assert res["recovery_managed"] is True
        assert res["reason"] == "SafetyRail: z below floor"
