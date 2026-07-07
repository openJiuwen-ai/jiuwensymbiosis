# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.rails.visual_feedback."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from jiuwensymbiosis.rails.visual_feedback import VisualFeedbackRail, _encode_jpeg_b64
from tests.helpers import FakeCtx, RecordingModelContext, RecordingRailSink, make_mock_session


@pytest.fixture
def mock_session():
    return make_mock_session()


class TestShouldTrigger:
    @pytest.mark.parametrize(
        ("tool_name", "tool_args", "watch_tools", "expected"),
        [
            ("goto_xyzr", None, None, True),
            ("close_gripper", None, None, True),
            ("get_pose", {}, None, False),
            ("custom_tool", None, {"custom_tool"}, True),
            ("robot_control", {"action": "goto_xyzr", "params": {}}, None, True),
        ],
        ids=["motion", "grasp", "other", "watch-tool", "robot-control"],
    )
    def test_should_trigger_policy(self, mock_session, tool_name, tool_args, watch_tools, expected):
        rail = VisualFeedbackRail(mock_session, watch_tools=watch_tools)
        assert rail._should_trigger(tool_name, tool_args=tool_args) is expected


class TestEncodeJpegB64:
    def test_valid_image(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = _encode_jpeg_b64(img, quality=80)
        assert result is not None
        import base64

        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    @pytest.mark.parametrize("bad_input", ["not_array", np.zeros((10, 10))], ids=["not-array", "wrong-ndim"])
    def test_none_on_bad_input(self, bad_input):
        assert _encode_jpeg_b64(bad_input) is None


class TestGrabFrameB64:
    def test_grabs_from_env(self, mock_session):
        rail = VisualFeedbackRail(mock_session)
        result = rail._grab_frame_b64()
        assert result is not None


def _make_ctx(mc=None, *, tool_name="goto_xyzr", tool_args=None, extra=None):
    """Build a ctx mimicking AgentCallbackContext for VFR hooks."""
    return FakeCtx(tool_name=tool_name, tool_args=tool_args, context=mc, extra=extra)


class TestAfterToolCallStagesOnly:
    """Phase 1 of the two-phase fix: ``after_tool_call`` must only capture +
    stage, never touch ModelContext. Injecting here would precede the
    ToolMessage openjiuwen writes after ``execute()`` returns, producing
    ``… → user(image) → tool(result)`` which OpenAI-style APIs reject."""

    @pytest.mark.asyncio
    async def test_after_tool_call_does_not_touch_context(self, mock_session):
        rail = VisualFeedbackRail(mock_session)
        mc = RecordingModelContext()
        ctx = _make_ctx(mc)
        await rail.after_tool_call(ctx)
        assert mc.added == []
        assert len(ctx.extra["visual_feedback_pending"]) == 1

    @pytest.mark.asyncio
    async def test_max_frames_counts_pending_plus_injected(self, mock_session):
        rail = VisualFeedbackRail(mock_session, max_frames_per_invoke=2)
        ctx = _make_ctx(RecordingModelContext())
        await rail.after_tool_call(ctx)  # 1 pending
        await rail.after_tool_call(ctx)  # 2 pending
        await rail.after_tool_call(ctx)  # cap reached — no new stage
        assert len(ctx.extra["visual_feedback_pending"]) == 2


class TestBeforeModelCallFlush:
    """Phase 2: ``before_model_call`` flushes staged frames now that
    ToolMessages are in the context, yielding the legal order
    ``assistant → tool(result) → user(image) → model call``."""

    @pytest.mark.asyncio
    async def test_flush_injects_usermessage_after_toolmessage(self, mock_session):
        """Simulate the real order: ToolMessage written by react_agent AFTER
        after_tool_call fires, then before_model_call flushes the staged frame."""
        from openjiuwen.core.foundation.llm.schema.message import (
            AssistantMessage,
            ToolMessage,
            UserMessage,
        )

        rail = VisualFeedbackRail(mock_session)
        mc = RecordingModelContext()
        ctx = _make_ctx(mc)
        # iteration: assistant emits tool_call → after_tool_call stages frame
        mc.added.append(AssistantMessage(content="tool_calls", tool_calls=[]))
        await rail.after_tool_call(ctx)
        # react_agent writes ToolMessage AFTER execute() returns
        mc.added.append(ToolMessage(content="result", tool_call_id="t1"))
        # next iteration top: before_model_call flushes
        await rail.before_model_call(ctx)
        assert len(mc.added) == 3
        assert isinstance(mc.added[0], AssistantMessage)
        assert isinstance(mc.added[1], ToolMessage)
        assert isinstance(mc.added[2], UserMessage)
        blocks = mc.added[2].content
        assert any(b.get("type") == "image_url" for b in blocks)
        assert "visual_feedback_pending" not in ctx.extra
        assert ctx.extra["visual_feedback_injected"] == ["goto_xyzr"]

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_one_iteration_flushes_all(self, mock_session):
        """One iteration, two sequential tool calls: pending accumulates, next
        before_model_call flushes both → ``… → tool1 → tool2 → user(img1) →
        user(img2) → model call`` (legal)."""
        rail = VisualFeedbackRail(mock_session)
        mc = RecordingModelContext()
        ctx = _make_ctx(mc)
        await rail.after_tool_call(ctx)  # tool 1
        await rail.after_tool_call(ctx)  # tool 2
        assert len(ctx.extra["visual_feedback_pending"]) == 2
        await rail.before_model_call(ctx)
        assert len(mc.added) == 2
        assert len(ctx.extra["visual_feedback_injected"]) == 2


class TestInjectFailureSafety:
    """Injection failures must never escape to the tool lifecycle — an escaped
    exception in after_tool_call lands in ON_TOOL_EXCEPTION and turns a
    successful action into a tool failure (RecoveryRail would even home+release
    a robot that didn't fail)."""

    @pytest.mark.asyncio
    async def test_runtime_error_does_not_escape(self, mock_session):
        class _Exploding:
            async def add_messages(self, msg):
                raise RuntimeError("store down")

        rail = VisualFeedbackRail(mock_session)
        ctx = _make_ctx(_Exploding())
        await rail.after_tool_call(ctx)
        # before_model_call must swallow, not raise
        await rail.before_model_call(ctx)
        assert ctx.extra.get("visual_feedback_injected", []) == []

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self, mock_session):
        """``asyncio.CancelledError`` must propagate — task cancellation is not
        an injection failure to swallow. It inherits BaseException, so
        ``except Exception`` does not catch it."""

        class _Cancel:
            async def add_messages(self, msg):
                raise asyncio.CancelledError()

        rail = VisualFeedbackRail(mock_session)
        ctx = _make_ctx(_Cancel())
        await rail.after_tool_call(ctx)
        with pytest.raises(asyncio.CancelledError):
            await rail.before_model_call(ctx)


class TestTraceSink:
    @pytest.mark.asyncio
    async def test_failed_flush_notifies_sink_failure(self, mock_session):
        class _Exploding:
            async def add_messages(self, msg):
                raise RuntimeError("store down")

        sink = RecordingRailSink()
        rail = VisualFeedbackRail(mock_session, trace_sink=sink)
        ctx = _make_ctx(_Exploding())
        await rail.after_tool_call(ctx)
        await rail.before_model_call(ctx)
        assert len(sink.events) == 1
        assert sink.events[0][3] is False  # success=False, not a false positive

    @pytest.mark.asyncio
    async def test_legacy_sink_receives_event_without_step_kwarg(self, mock_session):
        """A custom sink implementing only the base 4-arg ``record_rail_event``
        must not lose the delayed-injection event to a TypeError on a ``step``
        kwarg it doesn't accept. VFR duck-types ``record_rail_event_at_step``
        and falls back to the base method."""

        class _LegacySink:
            def __init__(self):
                self.events = []

            def record_rail_event(self, *, rail_name, kind, detail, success):
                self.events.append((rail_name, kind, detail, success))

        sink = _LegacySink()
        # trace_step is None here (no TraceRail) → base path; but even with a
        # step, a legacy sink has no record_rail_event_at_step → base path.
        rail = VisualFeedbackRail(mock_session, trace_sink=sink)
        ctx = _make_ctx(RecordingModelContext())
        await rail.after_tool_call(ctx)
        await rail.before_model_call(ctx)
        assert len(sink.events) == 1
        assert sink.events[0] == (
            "VisualFeedback",
            "inject_frame",
            {"tool_name": "goto_xyzr", "frame_path": None},
            True,
        )


class TestAfterInvokeCleanup:
    @pytest.mark.asyncio
    async def test_unconsumed_pending_cleared(self, mock_session):
        """Final iteration with no next model call leaves staged frames
        unconsumed — after_invoke must drop them so they don't leak across
        invokes."""
        rail = VisualFeedbackRail(mock_session)
        ctx = _make_ctx(RecordingModelContext())
        await rail.after_tool_call(ctx)
        assert "visual_feedback_pending" in ctx.extra
        await rail.after_invoke(ctx)
        assert "visual_feedback_pending" not in ctx.extra


def _make_lifecycle_ctx(mc=None, *, tool_name="goto_xyzr", tool_args=None) -> FakeCtx:
    return FakeCtx(tool_name=tool_name, tool_args=tool_args, context=mc)


class TestStepPreciseFlush:
    """Delayed-injection regression: when two tool calls happen in one
    iteration (step 1, step 2) and both stage a frame, ``before_model_call``
    must attach each event to the *correct* entry — not ``entries[-1]`` (which
    by flush time is step 2). ``_PendingFrame.trace_step`` captured at staging
    time is what makes this work."""

    @pytest.mark.asyncio
    async def test_two_step_flush_lands_on_correct_entries(self, mock_session, tmp_path):
        from jiuwensymbiosis.agent.trace import TraceRail

        trace_rail = TraceRail(mock_session, workspace=str(tmp_path), save_frames=False)
        # before_invoke initializes the trace + stashes trace_rail on ctx.extra
        invoke_ctx = _make_lifecycle_ctx()
        invoke_ctx.inputs.conversation_id = "c1"
        invoke_ctx.inputs.query = "q"
        await trace_rail.before_invoke(invoke_ctx)

        vfr = VisualFeedbackRail(mock_session, trace_sink=trace_rail)
        mc = RecordingModelContext()

        # step 1
        ctx1 = _make_lifecycle_ctx(mc, tool_name="goto_xyzr")
        ctx1.extra = invoke_ctx.extra  # share the extra that has _TRACE_RAIL_KEY
        await trace_rail.before_tool_call(ctx1)
        await vfr.after_tool_call(ctx1)
        await trace_rail.after_tool_call(ctx1)

        # step 2
        ctx2 = _make_lifecycle_ctx(mc, tool_name="close_gripper")
        ctx2.extra = invoke_ctx.extra
        await trace_rail.before_tool_call(ctx2)
        await vfr.after_tool_call(ctx2)
        await trace_rail.after_tool_call(ctx2)

        # flush both staged frames in one before_model_call
        flush_ctx = _make_lifecycle_ctx(mc)
        flush_ctx.extra = invoke_ctx.extra
        await vfr.before_model_call(flush_ctx)

        entries = trace_rail.trace.entries
        assert len(entries) == 2
        # entry 1 (step 1) got the goto_xyzr inject event
        e1_events = [e for e in entries[0].rail_events if e.get("rail_name") == "VisualFeedback"]
        assert len(e1_events) == 1
        assert e1_events[0]["detail"]["tool_name"] == "goto_xyzr"
        assert e1_events[0]["success"] is True
        # entry 2 (step 2) got the close_gripper inject event
        e2_events = [e for e in entries[1].rail_events if e.get("rail_name") == "VisualFeedback"]
        assert len(e2_events) == 1
        assert e2_events[0]["detail"]["tool_name"] == "close_gripper"

    @pytest.mark.asyncio
    async def test_frame_path_preserved_in_event(self, mock_session, tmp_path):
        """frame_sink returns a path; after_tool_call must stash it on the
        _PendingFrame and before_model_call must pass it through to the trace
        event (contract: docs/trace.md detail={tool_name, frame_path})."""
        from jiuwensymbiosis.agent.trace import TraceRail

        trace_rail = TraceRail(mock_session, workspace=str(tmp_path), save_frames=False)
        invoke_ctx = _make_lifecycle_ctx()
        invoke_ctx.inputs.conversation_id = "c"
        invoke_ctx.inputs.query = "q"
        await trace_rail.before_invoke(invoke_ctx)

        saved_path = tmp_path / "step_001.jpg"
        saved_path.write_bytes(b"\xff\xd8\xff\xe0dummy")

        def _frame_sink(rgb, tool_name):
            return str(saved_path)

        vfr = VisualFeedbackRail(mock_session, trace_sink=trace_rail, frame_sink=_frame_sink)
        ctx = _make_lifecycle_ctx(RecordingModelContext(), tool_name="goto_xyzr")
        ctx.extra = invoke_ctx.extra
        await trace_rail.before_tool_call(ctx)
        await vfr.after_tool_call(ctx)
        await trace_rail.after_tool_call(ctx)

        flush_ctx = _make_lifecycle_ctx(RecordingModelContext())
        flush_ctx.extra = invoke_ctx.extra
        await vfr.before_model_call(flush_ctx)

        entry = trace_rail.trace.entries[-1]
        ev = [e for e in entry.rail_events if e.get("rail_name") == "VisualFeedback"][0]
        assert ev["detail"]["frame_path"] == str(saved_path)
        assert saved_path.exists()
