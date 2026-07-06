# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.agent.trace."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jiuwensymbiosis.agent.abstractions import ToolOutput
from jiuwensymbiosis.agent.session import RobotSession
from jiuwensymbiosis.agent.trace import (
    StepAwareTraceEventSink,
    TraceEventSink,
    TraceRail,
    _unwrap_robot_control,
)
from jiuwensymbiosis.env.mock import MockArmEnv
from tests.mocks.mock_api import MockApi


class _FakeInputs:
    """Mimics openjiuwen ToolCallInputs / InvokeInputs."""

    def __init__(
        self,
        *,
        tool_name: str = "",
        tool_args: dict | None = None,
        tool_result=None,
        conversation_id: str = "",
        query: str | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.tool_result = tool_result
        self.conversation_id = conversation_id
        self.query = query


class _FakeCtx:
    """Mimics AgentCallbackContext."""

    def __init__(self, inputs: _FakeInputs, *, exception: Exception | None = None) -> None:
        self.inputs = inputs
        self.extra: dict = {}
        self.exception = exception


@pytest.fixture
def mock_session(tmp_path):
    env = MockArmEnv()
    api = MockApi(env)
    return RobotSession(env=env, api=api, name="test")


@pytest.fixture
def trace_rail(mock_session, tmp_path):
    return TraceRail(mock_session, workspace=str(tmp_path), save_frames=False)


class TestUnwrapRobotControl:
    def test_plain_tool(self):
        name, args = _unwrap_robot_control("goto_xyzr", {"x": 1, "y": 2, "z": 3})
        assert name == "goto_xyzr"
        assert args == {"x": 1, "y": 2, "z": 3}

    def test_robot_control_unwrap(self):
        name, args = _unwrap_robot_control("robot_control", {"action": "goto_xyzr", "params": {"x": 1}})
        assert name == "goto_xyzr"
        assert args == {"x": 1}

    def test_robot_control_unwrap_string_args(self):
        """openjiuwen delivers tool_args as a JSON string at before_tool_call
        time; the unwrap must parse it so the trace shows the real action."""
        name, args = _unwrap_robot_control(
            "robot_control",
            '{"action": "goto_xyzr", "params": {"x": 1, "y": 2, "z": 3}}',
        )
        assert name == "goto_xyzr"
        assert args == {"x": 1, "y": 2, "z": 3}

    def test_plain_tool_string_args_passthrough(self):
        """A non-robot_control tool with string args is parsed to a dict so the
        trace entry's input_params are readable (not the raw JSON string)."""
        name, args = _unwrap_robot_control("goto_xyzr", '{"x": 1, "y": 2, "z": 3}')
        assert name == "goto_xyzr"
        assert args == {"x": 1, "y": 2, "z": 3}

    def test_malformed_string_args_passthrough_unchanged(self):
        """Malformed JSON passes through unchanged — the trace still shows
        something honest rather than crashing or faking a parse."""
        name, args = _unwrap_robot_control("goto_xyzr", "not-json{")
        assert name == "goto_xyzr"
        assert args == "not-json{"


class TestBeforeAfterToolCall:
    @pytest.mark.asyncio
    async def test_records_step_and_duration(self, trace_rail):
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="c1", query="pick box")))
        ctx = _FakeCtx(_FakeInputs(tool_name="goto_xyzr", tool_args={"x": 100, "y": 0, "z": 80}))
        await trace_rail.before_tool_call(ctx)
        entry = ctx.extra["trace_current_step"]
        assert entry.tool_name == "goto_xyzr"
        assert entry.input_params == {"x": 100, "y": 0, "z": 80}
        # tool_result filled by after_tool_call
        ctx.inputs.tool_result = {"ok": True}
        await trace_rail.after_tool_call(ctx)
        assert entry.duration_s >= 0.0
        assert entry.success is True
        assert "ok" in entry.output_summary

    @pytest.mark.asyncio
    async def test_robot_control_entry_unwraps_action(self, trace_rail):
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="c2")))
        ctx = _FakeCtx(
            _FakeInputs(
                tool_name="robot_control",
                tool_args={"action": "close_gripper", "params": {"force_n": 10}},
            )
        )
        await trace_rail.before_tool_call(ctx)
        entry = ctx.extra["trace_current_step"]
        assert entry.tool_name == "close_gripper"
        assert entry.input_params == {"force_n": 10}

    @pytest.mark.asyncio
    async def test_observation_snapshot_captured(self, trace_rail):
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="c3")))
        ctx = _FakeCtx(_FakeInputs(tool_name="goto_xyzr", tool_args={"x": 1, "y": 2, "z": 3, "r": 0}))
        await trace_rail.before_tool_call(ctx)
        await trace_rail.after_tool_call(ctx)
        # after_tool_call pops the ctx.extra key, so read from trace
        last = trace_rail.trace.entries[-1]
        assert last.observation is not None
        assert "pose" in last.observation

    @pytest.mark.asyncio
    async def test_max_entries_truncates(self, mock_session, tmp_path):
        rail = TraceRail(mock_session, workspace=str(tmp_path), max_entries=3)
        await rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="c4")))
        for i in range(10):
            ctx = _FakeCtx(_FakeInputs(tool_name="goto_xyzr", tool_args={"x": i}))
            await rail.before_tool_call(ctx)
            ctx.inputs.tool_result = None
            await rail.after_tool_call(ctx)
        assert len(rail.trace.entries) == 3
        # Most recent 3 kept (steps 8,9,10).
        assert [e.step for e in rail.trace.entries] == [8, 9, 10]


class TestOnToolException:
    @pytest.mark.asyncio
    async def test_records_failure_and_error(self, trace_rail):
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="c5")))
        ctx = _FakeCtx(_FakeInputs(tool_name="goto_xyzr", tool_args={"x": 1, "y": 2, "z": 3}))
        await trace_rail.before_tool_call(ctx)
        exc = ValueError("ik failed")
        err_ctx = _FakeCtx(ctx.inputs, exception=exc)
        # carry the current-step entry over (same ctx.extra dict)
        err_ctx.extra = ctx.extra
        await trace_rail.on_tool_exception(err_ctx)
        last = trace_rail.trace.entries[-1]
        assert last.success is False
        assert "ValueError" in (last.error or "")
        assert "ik failed" in (last.error or "")

    @pytest.mark.asyncio
    async def test_rejected_call_keeps_success_false(self, trace_rail):
        # Regression: openjiuwen's @rail decorator fires ON_TOOL_EXCEPTION (the
        # except block) before AFTER_TOOL_CALL (the finally block) when a
        # before_tool_call raises (e.g. a SafetyRail rejection). Both hooks share
        # the same ctx.extra, so after_tool_call must NOT re-infer success=True
        # and overwrite the failure recorded by on_tool_exception.
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="c-rej")))
        ctx = _FakeCtx(_FakeInputs(tool_name="goto_xyzr", tool_args={"x": 1, "y": 2, "z": -50}))
        await trace_rail.before_tool_call(ctx)

        # 1) on_tool_exception fires first, recording the failure.
        exc_ctx = _FakeCtx(ctx.inputs, exception=ValueError("SafetyRail: z below floor"))
        exc_ctx.extra = ctx.extra  # same extra dict openjiuwen reuses
        await trace_rail.on_tool_exception(exc_ctx)

        # 2) after_tool_call then fires in the finally block — must keep failure.
        fin_ctx = _FakeCtx(ctx.inputs)
        fin_ctx.extra = ctx.extra
        fin_ctx.inputs.tool_result = None  # no result → default-inferred True if guard absent
        await trace_rail.after_tool_call(fin_ctx)

        last = trace_rail.trace.entries[-1]
        assert last.success is False
        assert "SafetyRail" in (last.error or "")
        assert last.duration_s >= 0.0

    @pytest.mark.asyncio
    async def test_catch_path_backfills_error_from_tool_output(self, trace_rail):
        # Regression: when a tool swallows its exception into ToolOutput
        # (success=False, error=...) — the RobotControlTool catch-path —
        # ON_TOOL_EXCEPTION does NOT fire, so on_tool_exception never sets
        # entry.error. after_tool_call must backfill entry.error from the
        # ToolOutput so a failed step doesn't record error=None (which makes
        # the trace JSON + replay HTML's "❌ ERROR" callout empty and forces
        # consumers to dig through output_summary).
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="c-catch")))
        ctx = _FakeCtx(_FakeInputs(tool_name="goto_xyzr", tool_args={"x": 1, "y": 2, "z": 3}))
        await trace_rail.before_tool_call(ctx)
        # No on_tool_exception fires (catch-path). after_tool_call sees the
        # failed ToolOutput directly.
        ctx.inputs.tool_result = ToolOutput(success=False, error="bad params for 'goto_xyzr': missing 3 args")
        await trace_rail.after_tool_call(ctx)

        last = trace_rail.trace.entries[-1]
        assert last.success is False
        assert last.error is not None, "error must be backfilled, not None"
        assert "bad params" in last.error
        assert "missing 3 args" in last.error


class TestRailEventSink:
    def test_protocols_preserve_legacy_sink_compatibility(self, trace_rail):
        from jiuwensymbiosis.agent import StepAwareTraceEventSink as PublicStepAwareTraceEventSink

        class _LegacySink:
            def record_rail_event(self, *, rail_name, kind, detail, success):
                pass

        legacy = _LegacySink()
        assert PublicStepAwareTraceEventSink is StepAwareTraceEventSink
        assert isinstance(legacy, TraceEventSink)
        assert not isinstance(legacy, StepAwareTraceEventSink)
        assert isinstance(trace_rail, StepAwareTraceEventSink)

    @pytest.mark.asyncio
    async def test_record_rail_event_into_current_step(self, trace_rail):
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="c6")))
        ctx = _FakeCtx(_FakeInputs(tool_name="goto_xyzr", tool_args={"x": 1, "y": 2, "z": 3}))
        await trace_rail.before_tool_call(ctx)
        trace_rail.record_rail_event(
            rail_name="SafetyRail",
            kind="reject",
            detail={"tool_name": "goto_xyzr", "reason": "z below floor"},
            success=False,
        )
        last = trace_rail.trace.entries[-1]
        assert any(e["rail_name"] == "SafetyRail" for e in last.rail_events)

    @pytest.mark.asyncio
    async def test_event_before_any_step_goes_pending(self, trace_rail):
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="c7")))
        trace_rail.record_rail_event(
            rail_name="RecoveryRail",
            kind="recover",
            detail={"home_ok": True},
            success=True,
        )
        # No entry yet → pending, then flushed into the first entry created next.
        ctx = _FakeCtx(_FakeInputs(tool_name="goto_xyzr", tool_args={}))
        await trace_rail.before_tool_call(ctx)
        first = trace_rail.trace.entries[-1]
        assert any(e["rail_name"] == "RecoveryRail" for e in first.rail_events)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("missing_step", [1, 99], ids=["evicted", "future"])
    async def test_missing_explicit_step_event_dropped(self, mock_session, tmp_path, missing_step):
        """An explicit missing target is dropped, never attached to a later step."""
        rail = TraceRail(mock_session, workspace=str(tmp_path), max_entries=1)
        await rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="c_evict")))

        async def _run_step(i):
            ctx = _FakeCtx(_FakeInputs(tool_name="goto_xyzr", tool_args={"x": i}))
            await rail.before_tool_call(ctx)
            ctx.inputs.tool_result = None
            await rail.after_tool_call(ctx)

        await _run_step(1)  # step 1 created
        await _run_step(2)  # step 1 evicted (max_entries=1) → entries=[step 2]
        assert [e.step for e in rail.trace.entries] == [2]

        # Target either evicted step 1 or a future step that does not exist.
        rail.trace.record_rail_event(
            rail_name="VisualFeedback",
            kind="inject_frame",
            detail={},
            success=True,
            step=missing_step,
        )
        step2 = rail.trace.entries[-1]
        assert not any(e["rail_name"] == "VisualFeedback" for e in step2.rail_events)
        assert rail.trace._pending_events == []

        # Running further steps must not flush the dropped event into them.
        await _run_step(3)
        step3 = rail.trace.entries[-1]
        assert not any(e["rail_name"] == "VisualFeedback" for e in step3.rail_events)
        assert rail.trace._pending_events == []


class TestLogCapture:
    @pytest.mark.asyncio
    async def test_warning_lands_in_step_log_events(self, trace_rail):
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="c8")))
        ctx = _FakeCtx(_FakeInputs(tool_name="goto_xyzr", tool_args={"x": 1, "y": 2, "z": 3}))
        await trace_rail.before_tool_call(ctx)
        trace_rail.record_log_event(
            logger_name="jiuwensymbiosis.rails.recovery",
            level="WARNING",
            msg="home() failed",
            ts=0.0,
        )
        last = trace_rail.trace.entries[-1]
        assert any("home() failed" in e["msg"] for e in last.log_events)

    @pytest.mark.asyncio
    async def test_log_without_step_goes_trace_log(self, trace_rail):
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="c9")))
        trace_rail.record_log_event(
            logger_name="jiuwensymbiosis.detector",
            level="WARNING",
            msg="detector unreachable",
            ts=0.0,
        )
        assert any("detector unreachable" in e["msg"] for e in trace_rail.trace.trace_log)


class TestSerialization:
    @pytest.mark.asyncio
    async def test_save_produces_valid_json(self, trace_rail, tmp_path):
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="conv-abc", query="pick")))
        ctx = _FakeCtx(_FakeInputs(tool_name="goto_xyzr", tool_args={"x": 1, "y": 2, "z": 3, "r": 0}))
        await trace_rail.before_tool_call(ctx)
        ctx.inputs.tool_result = {"ok": True, "position": [1, 2, 3]}
        await trace_rail.after_tool_call(ctx)
        path = trace_rail.finalize()
        assert path is not None
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["conversation_id"] == "conv-abc"
        assert data["query"] == "pick"
        assert len(data["entries"]) == 1
        assert data["entries"][0]["tool_name"] == "goto_xyzr"
        # round-trip re-serialize
        json.dumps(data)

    @pytest.mark.asyncio
    async def test_finalize_is_idempotent(self, trace_rail):
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="c10")))
        p1 = trace_rail.finalize()
        p2 = trace_rail.finalize()
        assert p1 is not None
        assert p2 is None  # trace cleared after first finalize


class TestCloseDetachesHandler:
    """close() = finalize + detach: the handler must not leak on loggers."""

    @pytest.mark.asyncio
    async def test_close_detaches_log_handler(self, trace_rail):
        import logging as _logging

        from jiuwensymbiosis.utils.logging import TraceLogHandler

        handler = TraceLogHandler(sink=trace_rail, level=_logging.WARNING)
        loggers = ("jiuwensymbiosis.trace_test_close",)
        for name in loggers:
            _logging.getLogger(name).addHandler(handler)
        trace_rail.attach_log_handler(handler, loggers)

        # Sanity: handler is attached.
        assert any(h is handler for h in _logging.getLogger(loggers[0]).handlers)

        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="c-close")))
        trace_rail.close()

        # Handler removed from the logger — no leak across builds.
        assert not any(h is handler for h in _logging.getLogger(loggers[0]).handlers)
        # And the rail's reference is cleared.
        assert trace_rail._log_handler is None

    @pytest.mark.asyncio
    async def test_close_keeps_handler_between_invokes_until_close(self, trace_rail):
        # finalize() (used between invokes) must NOT detach — only close() does.
        import logging as _logging

        from jiuwensymbiosis.utils.logging import TraceLogHandler

        handler = TraceLogHandler(sink=trace_rail, level=_logging.WARNING)
        loggers = ("jiuwensymbiosis.trace_test_finalize",)
        _logging.getLogger(loggers[0]).addHandler(handler)
        trace_rail.attach_log_handler(handler, loggers)

        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="c-fin")))
        trace_rail.finalize()
        # After finalize the handler is still attached (sink just nulled).
        assert any(h is handler for h in _logging.getLogger(loggers[0]).handlers)
        # A second invoke can rebind the sink (Fix1 regression guard).
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="c-fin2")))
        assert trace_rail._log_handler is not None
        trace_rail.close()


class TestSaveFrames:
    @pytest.mark.asyncio
    async def test_frame_saved_when_enabled(self, mock_session, tmp_path):
        rail = TraceRail(mock_session, workspace=str(tmp_path), save_frames=True, max_frames=5)
        await rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="c11")))
        ctx = _FakeCtx(_FakeInputs(tool_name="goto_xyzr", tool_args={"x": 1, "y": 2, "z": 3, "r": 0}))
        await rail.before_tool_call(ctx)
        ctx.inputs.tool_result = None
        await rail.after_tool_call(ctx)
        last = rail.trace.entries[-1]
        assert last.frame_path is not None
        assert Path(last.frame_path).exists()
        assert Path(last.frame_path).suffix == ".jpg"

    @pytest.mark.asyncio
    async def test_before_invoke_captures_initial_frame(self, mock_session, tmp_path):
        # The invoke-start frame is saved as step_000.jpg and exposed on the
        # trace as initial_frame_path, so step 1 has a before-frame for replay.
        rail = TraceRail(mock_session, workspace=str(tmp_path), save_frames=True, max_frames=5)
        await rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="c-init", query="pick")))
        assert rail.trace.initial_frame_path is not None
        p = Path(rail.trace.initial_frame_path)
        assert p.exists()
        assert p.name == "step_000.jpg"
        # The initial frame shares the max_frames budget.
        assert rail._frames_saved == 1
        # Survives serialization.
        path = rail.finalize()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["initial_frame_path"] == str(p)

    @pytest.mark.asyncio
    async def test_initial_frame_disabled_when_save_frames_off(self, mock_session, tmp_path):
        rail = TraceRail(mock_session, workspace=str(tmp_path), save_frames=False, max_frames=5)
        await rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="c-noinit")))
        assert rail.trace.initial_frame_path is None
        assert rail._frames_saved == 0

    @pytest.mark.asyncio
    async def test_max_frames_cap(self, mock_session, tmp_path):
        # max_frames covers ALL saved frames incl. the invoke-start initial
        # frame: with max_frames=2, the initial frame fills one slot, so only
        # one more (step) frame fits across the 5 tool calls below.
        rail = TraceRail(mock_session, workspace=str(tmp_path), save_frames=True, max_frames=2)
        await rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="c12")))
        for i in range(5):
            ctx = _FakeCtx(_FakeInputs(tool_name="goto_xyzr", tool_args={"x": i, "y": 0, "z": 50, "r": 0}))
            await rail.before_tool_call(ctx)
            ctx.inputs.tool_result = None
            await rail.after_tool_call(ctx)
        # Frames now live under a per-run subdir, so glob recursively.
        saved = list((tmp_path / "traces" / "frames").rglob("*.jpg"))
        assert len(saved) == 2  # 1 initial + 1 step (cap reached)

    @pytest.mark.asyncio
    async def test_two_invokes_do_not_overwrite_each_others_frames(self, mock_session, tmp_path):
        # Regression: each invoke must write frames into its own run-named subdir
        # (frames/{token}/step_NNN.jpg), so a later invoke's step_001.jpg does
        # NOT overwrite an earlier invoke's step_001.jpg — the historical trace
        # JSON's frame_path references must stay valid for replay.
        rail = TraceRail(mock_session, workspace=str(tmp_path), save_frames=True, max_frames=5)

        async def one_run(cid):
            await rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id=cid)))
            ctx = _FakeCtx(_FakeInputs(tool_name="goto_xyzr", tool_args={"x": 1, "y": 2, "z": 3, "r": 0}))
            await rail.before_tool_call(ctx)
            ctx.inputs.tool_result = None
            await rail.after_tool_call(ctx)
            frame_path = Path(rail.trace.entries[-1].frame_path)
            rail.finalize()
            return frame_path

        f1 = await one_run("runA")
        f2 = await one_run("runB")

        # Same step number, different run subdir → distinct files, both survive.
        assert f1.name == f2.name == "step_001.jpg"
        assert f1 != f2
        assert f1.parent != f2.parent
        assert f1.exists()
        assert f2.exists()
        # The frames subdir is named after the same run token as the trace JSON.
        frames_root = tmp_path / "traces" / "frames"
        subdirs = [p for p in frames_root.iterdir() if p.is_dir()]
        assert len(subdirs) == 2


class TestCurrentStepAndFrameSink:
    @pytest.mark.asyncio
    async def test_current_step_tracks_new_entry(self, trace_rail):
        # current_step is 0 before any entry, then equals the last created step.
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="cs1")))
        assert trace_rail.trace.current_step == 0
        ctx = _FakeCtx(_FakeInputs(tool_name="goto_xyzr", tool_args={"x": 1}))
        await trace_rail.before_tool_call(ctx)
        assert trace_rail.trace.current_step == 1
        ctx2 = _FakeCtx(_FakeInputs(tool_name="home", tool_args={}))
        await trace_rail.before_tool_call(ctx2)
        assert trace_rail.trace.current_step == 2

    @pytest.mark.asyncio
    async def test_frame_sink_returns_none_when_save_frames_off(self, trace_rail):
        """Contract: save_frame_for_sink is a no-op (returns None, writes
        nothing) when save_frames=False — the builder also skips installing
        the frame_sink, but this self-check is the backstop."""
        from jiuwensymbiosis.env.mock import MockArmEnv

        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="cs3")))
        rgb = MockArmEnv().get_observation().rgb
        path = trace_rail.save_frame_for_sink(rgb)
        assert path is None

    @pytest.mark.asyncio
    async def test_frame_sink_filename_aligns_with_entry_step(self, mock_session, tmp_path):
        # Contract: save_frame_for_sink must name the frame after the
        # active step so it matches entry.step (previously it read the private
        # _step_counter; now it uses the public current_step property).
        # save_frames=True is required: the self-check returns None when off.
        from jiuwensymbiosis.agent.trace import TraceRail

        trace_rail = TraceRail(mock_session, workspace=str(tmp_path), save_frames=True)
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="cs2")))
        # Create two steps so the active step is 2, then sink a frame.
        for name in ("goto_xyzr", "home"):
            ctx = _FakeCtx(_FakeInputs(tool_name=name, tool_args={}))
            await trace_rail.before_tool_call(ctx)
        # Active step is now 2 (last new_entry). Sink a frame directly.
        from jiuwensymbiosis.env.mock import MockArmEnv

        rgb = MockArmEnv().get_observation().rgb
        path = trace_rail.save_frame_for_sink(rgb)
        assert path is not None
        assert path.endswith("step_002.jpg")
        assert trace_rail.trace.current_step == 2


class TestConsoleDashboard:
    @pytest.mark.asyncio
    async def test_console_prints_lines(self, trace_rail, capsys):
        trace_rail.console = True
        await trace_rail.before_invoke(_FakeCtx(_FakeInputs(conversation_id="c13")))
        ctx = _FakeCtx(_FakeInputs(tool_name="goto_xyzr", tool_args={"x": 1, "y": 2, "z": 3}))
        await trace_rail.before_tool_call(ctx)
        ctx.inputs.tool_result = {"ok": True}
        await trace_rail.after_tool_call(ctx)
        out = capsys.readouterr().out
        assert "#1" in out
        assert "goto_xyzr" in out
