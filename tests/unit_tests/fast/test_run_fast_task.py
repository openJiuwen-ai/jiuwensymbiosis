# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the unified fast entry (``run_fast_task``) — in particular that the
TraceRail records + persists a trace JSON under the fast path, which (unlike the
agent path) never calls ``agent.invoke()``.

The fast path drives ``ability_manager.execute`` directly per op. Without the
manual invoke-lifecycle priming in ``run_fast_task`` (``_prime_fast_agent`` +
``_fire_invoke_event`` BEFORE/AFTER), TraceRail's ``self._trace`` would stay
``None`` (``before_invoke`` never fires) and no trace JSON would be written.

``compile_sequence`` is monkeypatched to a fixed sequence so no real LLM is
needed; the assertions are about trace persistence, not planning.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from jiuwensymbiosis.agent import ModelSpec, RobotAgentConfig
from jiuwensymbiosis.agent.run import run_fast_task
from jiuwensymbiosis.agent.trace import TraceRail
from jiuwensymbiosis.tools.robot_control_tool import _build_action_index
from tests.helpers import FakeCtx, make_mock_session

# A minimal pick-like sequence (no track_detect → no servo threads, deterministic).
_FIXED_SEQUENCE = [
    {"op": "home"},
    {"op": "open_gripper"},
    {"op": "get_grasp_info_simple", "params": {"object_name": "box"}, "bind": "b"},
    {"op": "goto_xyzr", "params": {"x": "b.x", "y": "b.y", "z": "b.z"}},
    {"op": "close_gripper"},
]


def _make_session():
    return make_mock_session(name="piper_mock")


def _install_fast_agent_doubles(monkeypatch, session):
    """Keep run_fast_task unit-level: real runner + TraceRail, fake DeepAgent."""
    state = SimpleNamespace(agent=None)

    def _build_robot_agent(_session, config):
        trace = None
        if config.enable_tracing:
            trace = TraceRail(
                _session,
                workspace=config.workspace,
                max_entries=config.trace_max_entries,
                max_frames=config.trace_max_frames,
                save_frames=config.trace_save_frames,
            )
            _session._trace_rail = trace
        state.agent = SimpleNamespace(trace_rail=trace)
        return state.agent

    def _prime_fast_agent(_agent):
        return None

    def _fire_invoke_event(agent, event, *, conversation_id, query):
        trace = agent.trace_rail
        if trace is None:
            return
        ctx = FakeCtx(conversation_id=conversation_id, query=query)
        if getattr(event, "name", "") == "BEFORE_INVOKE":
            asyncio.run(trace.before_invoke(ctx))
        else:
            asyncio.run(trace.after_invoke(ctx))

    def _build_ability_executor(agent):
        action_index = _build_action_index(session.api)

        def _run(op, params):
            trace = agent.trace_rail
            ctx = FakeCtx(tool_name="robot_control", tool_args={"action": op, "params": params})
            if trace is not None:
                asyncio.run(trace.before_tool_call(ctx))
            try:
                result = action_index[op](**params)
            except Exception as exc:
                if trace is not None:
                    ctx.exception = exc
                    asyncio.run(trace.on_tool_exception(ctx))
                return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
            ctx.inputs.tool_result = {"ok": True, "result": result}
            if trace is not None:
                asyncio.run(trace.after_tool_call(ctx))
            return {"ok": True, "result": result}

        return _run

    monkeypatch.setattr("jiuwensymbiosis.agent.run.build_robot_agent", _build_robot_agent)
    monkeypatch.setattr("jiuwensymbiosis.agent.run._prime_fast_agent", _prime_fast_agent)
    monkeypatch.setattr("jiuwensymbiosis.agent.run._fire_invoke_event", _fire_invoke_event)
    monkeypatch.setattr(
        "jiuwensymbiosis.agent.fast.ability_exec.build_ability_executor",
        _build_ability_executor,
    )
    return state


def _patched_run_fast_task(monkeypatch, session, cfg, query, conv_id):
    """Call run_fast_task with compile_sequence stubbed and DeepAgent faked."""
    _install_fast_agent_doubles(monkeypatch, session)
    with mock.patch(
        "jiuwensymbiosis.agent.fast.compile_sequence",
        return_value=_FIXED_SEQUENCE,
    ):
        return run_fast_task(session, query, cfg, conversation_id=conv_id)


class TestFastTracePersistence:
    """The fast path must persist a non-empty trace when tracing is on."""

    def test_trace_json_persisted_with_one_entry_per_step(self, tmp_path, monkeypatch):
        session = _make_session()
        cfg = RobotAgentConfig()
        cfg.model_spec = ModelSpec()  # non-None so run_fast_task proceeds past the spec check
        cfg.exec_mode = "fast"
        cfg.enable_tracing = True
        cfg.workspace = str(tmp_path)
        cfg.enable_visual_feedback = False
        cfg.enable_skill = True  # robot_control tool — ability_exec dispatches via it

        conv_id = "fast-trace-test"
        with session:
            result = _patched_run_fast_task(monkeypatch, session, cfg, "pick the box", conv_id)

        assert result["ok"] is True
        assert result["steps_done"] == len(_FIXED_SEQUENCE)

        traces_dir = Path(tmp_path) / "traces"
        json_files = list(traces_dir.glob("*.json"))
        assert len(json_files) == 1, "exactly one trace JSON should be written per run"

        data = json.loads(json_files[0].read_text(encoding="utf-8"))
        assert data["conversation_id"] == conv_id
        assert data["query"] == "pick the box"
        # One trace entry per discrete sequence step (servo ticks are NOT traced —
        # they bypass ability_manager and call api.servo_to_* directly).
        assert len(data["entries"]) == len(_FIXED_SEQUENCE)
        # Every entry corresponds to a robot_control dispatch of one sequence op.
        assert all(e["success"] for e in data["entries"])
        assert [e["step"] for e in data["entries"]] == list(range(1, len(_FIXED_SEQUENCE) + 1))

    def test_no_trace_written_when_tracing_off(self, tmp_path, monkeypatch):
        session = _make_session()
        cfg = RobotAgentConfig()
        cfg.model_spec = ModelSpec()
        cfg.exec_mode = "fast"
        cfg.enable_tracing = False  # default — zero overhead, no trace file
        cfg.workspace = str(tmp_path)
        cfg.enable_visual_feedback = False
        cfg.enable_skill = True

        with session:
            result = _patched_run_fast_task(monkeypatch, session, cfg, "noop", "fast-off")

        assert result["ok"] is True
        traces_dir = Path(tmp_path) / "traces"
        assert not traces_dir.exists() or not list(traces_dir.glob("*.json"))

    def test_trace_run_token_embeds_conversation_id(self, tmp_path, monkeypatch):
        """The trace JSON filename derives from conversation_id (replay lookup)."""
        session = _make_session()
        cfg = RobotAgentConfig()
        cfg.model_spec = ModelSpec()
        cfg.exec_mode = "fast"
        cfg.enable_tracing = True
        cfg.workspace = str(tmp_path)
        cfg.enable_visual_feedback = False
        cfg.enable_skill = True

        conv_id = "lookup-me-123"
        with session:
            _patched_run_fast_task(monkeypatch, session, cfg, "pick", conv_id)

        json_files = list((Path(tmp_path) / "traces").glob("*.json"))
        assert len(json_files) == 1
        # run_token = sanitized conversation_id + timestamp + pid — starts with cid.
        assert json_files[0].name.startswith(conv_id)
