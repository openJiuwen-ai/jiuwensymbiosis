# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for DiagnosisRail — see ``docs/trace-feedback-loop-design.md`` §7.1."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from jiuwensymbiosis.agent.abstractions import ToolOutput
from jiuwensymbiosis.agent.trace import TraceRail
from jiuwensymbiosis.rails.diagnosis import DiagnosisRail
from tests.helpers import FakeCtx as _FakeCtx
from tests.helpers import FakeInputs as _FakeInputs
from tests.helpers import RecordingModelContext, make_mock_session


@pytest.fixture
def mock_session():
    return make_mock_session()


@pytest.fixture
def trace_rail(mock_session, tmp_path):
    return TraceRail(mock_session, workspace=str(tmp_path), save_frames=False)


@pytest.fixture
def diagnosis_rail(mock_session):
    return DiagnosisRail(mock_session)


async def _step(
    trace_rail: TraceRail,
    diagnosis_rail: DiagnosisRail,
    *,
    extra: dict[str, Any],
    tool_name: str = "goto_xyzr",
    tool_args: Any = None,
    tool_result: Any = None,
    exception: Exception | None = None,
) -> _FakeCtx:
    """Run one step through both rails, sharing ctx.extra.

    Mirrors openjiuwen's lifecycle: before_tool_call → on_tool_exception (if
    raised) → after_tool_call (finally). Staging happens in after/exception;
    flushing in before_model_call.
    """
    inputs = _FakeInputs(tool_name=tool_name, tool_args=tool_args or {})
    ctx = _FakeCtx(inputs, extra=extra)
    await trace_rail.before_tool_call(ctx)
    if exception is not None:
        exc_ctx = _FakeCtx(inputs, extra=extra, exception=exception)
        await trace_rail.on_tool_exception(exc_ctx)
        await diagnosis_rail.on_tool_exception(exc_ctx)
    # after_tool_call sees the tool_result (catch-path) or the post-exception state.
    fin_ctx = _FakeCtx(inputs, extra=extra)
    fin_ctx.inputs.tool_result = tool_result
    await trace_rail.after_tool_call(fin_ctx)
    await diagnosis_rail.after_tool_call(fin_ctx)
    return fin_ctx


# --------------------------------------------------------------------------- helpers


def _staged_texts(ctx_extra: dict[str, Any]) -> list[str]:
    return list(ctx_extra.get("diagnosis_pending", []))


class TestRepeatedSafetyReject:
    @pytest.mark.asyncio
    async def test_message_has_params_history_and_repeat(self, trace_rail, diagnosis_rail):
        extra: dict[str, Any] = {}
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="rep", query="pick")))
        extra["trace_rail"] = trace_rail

        # Two earlier rejected steps (same tool) so the causal chain has matches.
        await _step(
            trace_rail,
            diagnosis_rail,
            extra=extra,
            tool_name="goto_xyzr",
            tool_args={"x": 1, "y": 0, "z": -50},
            tool_result=ToolOutput(success=False, error="SafetyRail: z below floor"),
        )
        await _step(
            trace_rail,
            diagnosis_rail,
            extra=extra,
            tool_name="goto_xyzr",
            tool_args={"x": 2, "y": 0, "z": -50},
            tool_result=ToolOutput(success=False, error="SafetyRail: z below floor"),
        )

        texts = _staged_texts(extra)
        assert len(texts) == 2  # two failed steps → two staged diagnoses
        last = texts[-1]
        assert "goto_xyzr" in last
        assert "z below floor" in last
        assert "x" in last
        assert "#1" in last or "#2" in last  # causal chain references an earlier step


class TestToolExceptionCatchPath:
    @pytest.mark.asyncio
    async def test_message_contains_error_params_and_recovery(self, trace_rail, diagnosis_rail):
        extra: dict[str, Any] = {}
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="catch", query="pick")))
        extra["trace_rail"] = trace_rail

        # RecoveryRail fires in on_tool_exception, before after_tool_call —
        # record its event against the current step to populate system_state.
        ctx = _FakeCtx(_FakeInputs(tool_name="goto_xyzr", tool_args={"x": 5, "y": 5, "z": 5}), extra=extra)
        await trace_rail.before_tool_call(ctx)
        trace_rail.record_rail_event(
            rail_name="RecoveryRail",
            kind="recover",
            detail={"tool_name": "goto_xyzr", "home_ok": True, "released_ok": False},
            success=True,
        )
        ctx.inputs.tool_result = ToolOutput(success=False, error="ik solver diverged")
        await trace_rail.after_tool_call(ctx)
        await diagnosis_rail.after_tool_call(ctx)

        texts = _staged_texts(extra)
        assert len(texts) == 1
        msg = texts[0]
        assert "ik solver diverged" in msg
        assert "goto_xyzr" in msg
        assert "home_ok=True" in msg
        assert "released_ok=False" in msg


class TestFailureChannelMutualExclusion:
    @pytest.mark.asyncio
    async def test_exception_path_injects_once(self, trace_rail, diagnosis_rail):
        # Type B: on_tool_exception fires, then after_tool_call in finally —
        # the per-step dedup must keep this to one staged diagnosis.
        extra: dict[str, Any] = {}
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="exc", query="pick")))
        extra["trace_rail"] = trace_rail

        await _step(
            trace_rail,
            diagnosis_rail,
            extra=extra,
            tool_name="goto_xyzr",
            tool_args={"x": 1, "y": 2, "z": 3},
            exception=ValueError("SafetyRail: refusing goto_xyzr: z below floor"),
            tool_result=None,
        )

        texts = _staged_texts(extra)
        assert len(texts) == 1, "on_tool_exception + after_tool_call must not double-inject"
        assert "SafetyRail" in texts[0]
        assert "ValueError" in texts[0]

    @pytest.mark.asyncio
    async def test_catch_path_does_not_fire_on_tool_exception(self, trace_rail, diagnosis_rail):
        # Type A: no exception → on_tool_exception stages nothing; only after_tool_call.
        extra: dict[str, Any] = {}
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="catch2", query="pick")))
        extra["trace_rail"] = trace_rail

        ctx = _FakeCtx(_FakeInputs(tool_name="goto_xyzr", tool_args={"x": 1}), extra=extra)
        await trace_rail.before_tool_call(ctx)
        exc_ctx = _FakeCtx(ctx.inputs, extra=extra, exception=None)
        await diagnosis_rail.on_tool_exception(exc_ctx)
        assert _staged_texts(extra) == []


class TestTracingOffNoOp:
    @pytest.mark.asyncio
    async def test_no_trace_rail_is_silent(self, diagnosis_rail):
        extra: dict[str, Any] = {}  # no "trace_rail" key → no-op
        ctx = _FakeCtx(_FakeInputs(tool_name="goto_xyzr", tool_args={"x": 1}), extra=extra)
        ctx.inputs.tool_result = ToolOutput(success=False, error="boom")
        await diagnosis_rail.after_tool_call(ctx)  # must not raise
        assert _staged_texts(extra) == []


class TestTokenDegrade:
    @pytest.mark.asyncio
    async def test_small_cap_keeps_current_and_system_state(self, mock_session, trace_rail):
        diag = DiagnosisRail(mock_session, max_chars=180, history_steps=5)
        extra: dict[str, Any] = {}
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="trunc", query="pick")))
        extra["trace_rail"] = trace_rail

        for i in range(4):  # prior failed steps populate the causal chain
            await _step(
                trace_rail,
                diag,
                extra=extra,
                tool_name="goto_xyzr",
                tool_args={"x": i, "y": 0, "z": -50},
                tool_result=ToolOutput(success=False, error="z below floor"),
            )

        texts = _staged_texts(extra)
        assert texts
        last = texts[-1]
        assert "goto_xyzr" in last
        assert "z below floor" in last
        assert "请据此修正" in last
        assert len(last) <= 180 + 24

    def test_truncate_drops_causal_before_system_state(self, mock_session):
        # Cap fits current+system_state+instruction but not causal → causal
        # dropped, system_state (recovery/pose) retained.
        diag = DiagnosisRail(mock_session, max_chars=320, history_steps=5)
        # A message with all three sections, causal long enough to exceed the cap.
        msg = (
            "### 诊断：上一步失败\n"
            "[diagnosis] step failed: goto_xyzr\n"
            "  error: ik diverged\n"
            "  params: {'x': 9, 'y': 9, 'z': 9}\n"
            "  rail: RecoveryRail/recover {'home_ok': True, 'released_ok': False}\n"
            "\n"
            "### 相关历史（可能反复失败）\n"
            "  - #1 goto_xyzr({'x': 1}) → FAIL: z below floor\n"
            "  - #2 goto_xyzr({'x': 2}) → FAIL: z below floor\n"
            "  - #3 goto_xyzr({'x': 3}) → FAIL: z below floor\n"
            "  - #4 goto_xyzr({'x': 4}) → FAIL: z below floor\n"
            "\n"
            "### 系统状态\n"
            "  recovery: home_ok=True, released_ok=False\n"
            "  pose: {'x': 0, 'y': 0, 'z': 200}\n"
            "\n"
            "请据此修正参数或换策略，不要用相同参数重试。"
        )
        assert len(msg) > 320, "precondition: message must exceed the cap"
        out = diag._truncate(msg)
        assert "### 相关历史" not in out  # causal dropped first
        assert "home_ok=True" in out  # system_state retained
        assert "released_ok=False" in out
        assert "ik diverged" in out  # current step retained
        assert len(out) <= 320 + 24

    def test_truncate_drops_system_state_only_after_causal_gone(self, mock_session):
        # Cap too small for current+system_state even after causal dropped →
        # system_state goes next (drop order: causal → system_state).
        diag = DiagnosisRail(mock_session, max_chars=120, history_steps=5)
        msg = (
            "### 诊断：上一步失败\n"
            "[diagnosis] step failed: goto_xyzr\n"
            "  error: ik diverged\n"
            "\n"
            "### 相关历史（可能反复失败）\n"
            "  - #1 goto_xyzr → FAIL\n"
            "\n"
            "### 系统状态\n"
            "  recovery: home_ok=True, released_ok=False\n"
            "\n"
            "请据此修正参数或换策略，不要用相同参数重试。"
        )
        out = diag._truncate(msg)
        assert "### 相关历史" not in out
        assert "ik diverged" in out  # current step kept + hard-truncated


class TestRobotControlUnwrap:
    # SKILL mode dispatches as robot_control{action,params}; the diagnosis must
    # show the real action. TraceRail already unwraps into entry.tool_name.

    @pytest.mark.asyncio
    async def test_entry_tool_name_preferred_over_robot_control(self, trace_rail, diagnosis_rail):
        extra: dict[str, Any] = {}
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="rc", query="pick")))
        extra["trace_rail"] = trace_rail

        ctx = _FakeCtx(
            _FakeInputs(
                tool_name="robot_control",
                tool_args={"action": "goto_xyzr", "params": {"x": 1, "y": 2, "z": 3}},
            ),
            extra=extra,
        )
        await trace_rail.before_tool_call(ctx)
        ctx.inputs.tool_result = ToolOutput(success=False, error="ik failed")
        await trace_rail.after_tool_call(ctx)
        await diagnosis_rail.after_tool_call(ctx)

        last = _staged_texts(extra)[-1]
        assert "goto_xyzr" in last
        assert "robot_control" not in last

    @pytest.mark.asyncio
    async def test_unwrap_falls_back_when_no_entry(self, mock_session, trace_rail):
        diag = DiagnosisRail(mock_session)
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="rc2", query="pick")))
        # entry=None forces the tool_args unwrap fallback; tool_args is the JSON
        # string form openjiuwen delivers at hook time.
        inputs = _FakeInputs(
            tool_name="robot_control",
            tool_args='{"action": "close_gripper", "params": {}}',
        )
        ctx = _FakeCtx(inputs, extra={"trace_rail": trace_rail})
        text = diag._build_message(ctx, entry=None, error="some error", trace=trace_rail.trace)
        assert text is not None
        assert "close_gripper" in text
        assert "robot_control" not in text


class TestFastPathCtxIsolation:
    @pytest.mark.asyncio
    async def test_no_model_context_does_not_raise(self, trace_rail, diagnosis_rail):
        extra: dict[str, Any] = {}
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="fast", query="pick")))
        extra["trace_rail"] = trace_rail

        await _step(
            trace_rail,
            diagnosis_rail,
            extra=extra,
            tool_name="goto_xyzr",
            tool_args={"x": 1, "y": 2, "z": 3},
            tool_result=ToolOutput(success=False, error="driver lost"),
        )
        assert _staged_texts(extra)  # staged even without a ModelContext

        # Fast-path op-ctx has no ModelContext: flush must no-op, never raise.
        flush_ctx = _FakeCtx(_FakeInputs(), extra=extra, context=None)
        await diagnosis_rail.before_model_call(flush_ctx)


class TestTwoPhaseInjectionOrder:
    # after_tool_call must only stage; before_model_call flushes — injecting in
    # after_tool_call would precede the ToolMessage (illegal message order).

    @pytest.mark.asyncio
    async def test_after_does_not_inject_before_flush(self, trace_rail, diagnosis_rail):
        extra: dict[str, Any] = {}
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="order", query="pick")))
        extra["trace_rail"] = trace_rail
        mc = RecordingModelContext()

        await _step(
            trace_rail,
            diagnosis_rail,
            extra=extra,
            tool_name="goto_xyzr",
            tool_args={"x": 1, "y": 2, "z": 3},
            tool_result=ToolOutput(success=False, error="oops"),
        )
        assert mc.added == []  # after_tool_call must not touch the ModelContext
        assert _staged_texts(extra)

        flush_ctx = _FakeCtx(_FakeInputs(), extra=extra, context=mc)
        await diagnosis_rail.before_model_call(flush_ctx)
        assert len(mc.added) == 1
        assert _staged_texts(extra) == []  # pending cleared

    @pytest.mark.asyncio
    async def test_after_invoke_drops_unflushed(self, trace_rail, diagnosis_rail):
        extra: dict[str, Any] = {}
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="drop", query="pick")))
        extra["trace_rail"] = trace_rail
        await _step(
            trace_rail,
            diagnosis_rail,
            extra=extra,
            tool_name="goto_xyzr",
            tool_args={"x": 1, "y": 2, "z": 3},
            tool_result=ToolOutput(success=False, error="oops"),
        )
        assert _staged_texts(extra)
        end_ctx = _FakeCtx(_FakeInputs(), extra=extra)
        await diagnosis_rail.after_invoke(end_ctx)
        assert _staged_texts(extra) == []


class TestInjectFailureIsSwallowed:
    # A failing add_messages must never escape into the tool lifecycle.

    @pytest.mark.asyncio
    async def test_add_messages_error_does_not_raise(self, trace_rail, diagnosis_rail):
        extra: dict[str, Any] = {}
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="inj", query="pick")))
        extra["trace_rail"] = trace_rail
        await _step(
            trace_rail,
            diagnosis_rail,
            extra=extra,
            tool_name="goto_xyzr",
            tool_args={"x": 1, "y": 2, "z": 3},
            tool_result=ToolOutput(success=False, error="oops"),
        )

        class _BrokenMC:
            async def add_messages(self, _msg):
                raise RuntimeError("boom")

        flush_ctx = _FakeCtx(_FakeInputs(), extra=extra, context=_BrokenMC())
        await diagnosis_rail.before_model_call(flush_ctx)


# --------------------------------------------------------------------------- builder wiring


class TestDiagnosisBuilderWiring:
    def test_priority_below_trace_and_default(self):
        from jiuwensymbiosis.agent.abstractions import AgentRail

        assert DiagnosisRail.priority < TraceRail.priority
        assert DiagnosisRail.priority < AgentRail.priority

    def test_attached_via_build_capture(self, mock_session, tmp_path, monkeypatch):
        from jiuwensymbiosis.agent import builder as builder_mod
        from jiuwensymbiosis.agent.config import RobotAgentConfig

        captured: dict[str, Any] = {}

        def _fake_create(**kwargs):
            captured.update(kwargs)
            return kwargs

        monkeypatch.setattr(builder_mod, "create_deep_agent", _fake_create)
        cfg = RobotAgentConfig(
            enable_tracing=True,
            enable_diagnosis=True,
            workspace=str(tmp_path),
            model=object(),
            log_dir=None,
        )
        builder_mod.build_robot_agent(mock_session, cfg)
        rails = captured["rails"]
        assert any(isinstance(r, DiagnosisRail) for r in rails)
        assert any(isinstance(r, TraceRail) for r in rails)

    def test_disabled_when_tracing_off(self, mock_session, tmp_path, monkeypatch, caplog):
        from jiuwensymbiosis.agent import builder as builder_mod
        from jiuwensymbiosis.agent.config import RobotAgentConfig

        captured: dict[str, Any] = {}

        def _fake_create(**kwargs):
            captured.update(kwargs)
            return kwargs

        monkeypatch.setattr(builder_mod, "create_deep_agent", _fake_create)
        cfg = RobotAgentConfig(
            enable_tracing=False,
            enable_diagnosis=True,
            workspace=str(tmp_path),
            model=object(),
            log_dir=None,
        )
        with caplog.at_level(logging.WARNING):
            builder_mod.build_robot_agent(mock_session, cfg)
        rails = captured["rails"]
        assert not any(isinstance(r, DiagnosisRail) for r in rails)
        assert not any(isinstance(r, TraceRail) for r in rails)
        assert any("enable_diagnosis=True requires enable_tracing" in rec.message for rec in caplog.records)
