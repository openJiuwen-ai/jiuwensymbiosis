# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``build_ability_executor`` must surface tool failures with the right recovery owner.

A tool that RAN without raising (``ToolOutput.success is True``) can still report
a business failure inside its return, e.g. ``approach_for_grasp -> {"ok": False}`` on
``nav_failed``/``too_close``. If the executor reported that as a step success, the
fast runner would march on to grasp/place on an unreached target (the observed
"detect box → skip to placement" bug). The executor must mirror ``direct_executor``
and propagate the inner ``ok`` — but, because no exception fired, RecoveryRail did
NOT run, so ``recovery_managed`` must stay unset (the runner does its own retreat).

``ToolOutput(success=False)`` is different: ``RobotControlTool`` converts adapter
exceptions into that shape without firing the exception hook, so RecoveryRail never
saw them. Only a failure that reached the adapter (``data["dispatched"]``) owes a
safe retreat; a call rejected before dispatch never moved anything.

The openjiuwen ``ToolCall`` / ``AgentCallbackContext`` constructors accept plain
namespaces, so a scripted ability_manager isolates the ok-extraction logic without
a real agent, rails, or LLM.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from jiuwensymbiosis.agent.fast.ability_exec import build_ability_executor


class _FakeAbilityManager:
    """Returns one scripted ToolOutput/exception and records the dispatched action/params."""

    def __init__(self, output, *, recovered=False):
        self._output = output
        self._recovered = recovered
        self.dispatched: list[dict] = []

    async def execute(self, ctx, tool_calls, session):
        self.dispatched.append(json.loads(tool_calls[0].arguments))
        if self._recovered:
            # What RecoveryRail.after_tool_call leaves behind after release/home.
            ctx.extra["jiuwensymbiosis.recovery_performed"] = True
        if isinstance(self._output, Exception):
            raise self._output
        return [[self._output]]


def _run_for(output, *, recovered=False):
    agent = SimpleNamespace(
        ability_manager=_FakeAbilityManager(output, recovered=recovered),
        react_agent=SimpleNamespace(),
        loop_session=SimpleNamespace(),
    )
    return build_ability_executor(agent)


def _tool_output(*, success, result=None, error=None, data=None):
    if data is None:
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

    def test_validation_rejection_needs_no_retreat(self):
        # No "dispatched" marker: the call was rejected before the adapter ran,
        # so nothing moved and the runner must not home over it.
        run = _run_for(_tool_output(success=False, error="unknown action 'goto_xyzr'"))
        res = run("goto_xyzr", {"x": 0, "y": 0, "z": -1})
        assert res["ok"] is False
        assert res["recovery_managed"] is True
        assert res["reason"] == "unknown action 'goto_xyzr'"

    def test_adapter_failure_recovers_through_the_recovery_rail(self):
        # RobotControlTool converted a mid-motion adapter exception into
        # success=False; RecoveryRail sees it at after_tool_call and recovers,
        # so the runner must not retreat a second time.
        run = _run_for(
            _tool_output(
                success=False,
                error="RuntimeError: driver lost connection mid-move",
                data={"action": "goto_xyzr", "dispatched": True},
            ),
            recovered=True,
        )
        res = run("goto_xyzr", {"x": 0, "y": 0, "z": 30})
        assert res["ok"] is False
        assert res["recovery_managed"] is True
        assert res["reason"] == "RuntimeError: driver lost connection mid-move"

    def test_adapter_failure_without_rail_falls_back_to_runner_retreat(self):
        # RecoveryRail disabled or declined: the runner still owes a retreat.
        run = _run_for(
            _tool_output(
                success=False,
                error="RuntimeError: driver lost connection mid-move",
                data={"action": "goto_xyzr", "dispatched": True},
            )
        )
        res = run("goto_xyzr", {"x": 0, "y": 0, "z": 30})
        assert res["ok"] is False
        assert "recovery_managed" not in res
        assert res["reason"] == "RuntimeError: driver lost connection mid-move"

    def test_aborted_call_keeps_recovery_managed(self):
        # No ToolOutput at all: a rail aborted the call before the tool ran.
        run = _run_for(None)
        res = run("goto_xyzr", {"x": 0, "y": 0, "z": 30})
        assert res["ok"] is False
        assert res["recovery_managed"] is True
        assert res["error_code"] == ""  # same key as every other failure branch

    def test_wrapped_safety_rejection_needs_no_retreat(self):
        """The framework wraps failures; the skip marker lives on the cause."""

        class _WrappedFailure(RuntimeError):
            def __init__(self, cause):
                super().__init__("tool failed")
                self.cause = cause

        from jiuwensymbiosis.errors import SafetyViolationError

        wrapped = _WrappedFailure(SafetyViolationError("SafetyRail: refusing goto_xyzr"))
        res = _run_for(wrapped)("goto_xyzr", {"x": 0, "y": 0, "z": -10})
        assert res["ok"] is False
        assert res["recovery_managed"] is True

    def test_safety_rejection_raised_through_rails_needs_no_retreat(self):
        from jiuwensymbiosis.errors import SafetyViolationError

        run = _run_for(SafetyViolationError("SafetyRail: refusing goto_xyzr: z=-1 below z_floor=0"))
        res = run("goto_xyzr", {"x": 0, "y": 0, "z": -1})
        assert res["ok"] is False
        assert res["recovery_managed"] is True
        assert res["error_code"] == "safety_rejected"

    def test_unexpected_exception_leaves_recovery_to_the_runner(self):
        run = _run_for(RuntimeError("ability manager exploded"))
        res = run("goto_xyzr", {"x": 0, "y": 0, "z": 30})
        assert res["ok"] is False
        assert "recovery_managed" not in res
