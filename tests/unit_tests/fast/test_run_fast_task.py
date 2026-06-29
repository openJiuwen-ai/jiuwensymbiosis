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

import json
from pathlib import Path
from unittest import mock

from jiuwensymbiosis.agent import ModelSpec, RobotAgentConfig
from jiuwensymbiosis.agent.run import run_fast_task
from jiuwensymbiosis.agent.session import RobotSession
from jiuwensymbiosis.env.mock import MockArmEnv
from tests.mocks.mock_api import MockApi

# A minimal pick-like sequence (no track_detect → no servo threads, deterministic).
_FIXED_SEQUENCE = [
    {"op": "home"},
    {"op": "open_gripper"},
    {"op": "get_grasp_info_simple", "params": {"object_name": "box"}, "bind": "b"},
    {"op": "goto_xyzr", "params": {"x": "b.x", "y": "b.y", "z": "b.z"}},
    {"op": "close_gripper"},
]


def _make_session() -> RobotSession:
    env = MockArmEnv()
    api = MockApi(env)
    return RobotSession(env=env, api=api, name="piper_mock")


def _patched_run_fast_task(session, cfg, query, conv_id):
    """Call run_fast_task with compile_sequence stubbed to the fixed sequence."""
    with mock.patch(
        "jiuwensymbiosis.agent.fast.compile_sequence",
        return_value=_FIXED_SEQUENCE,
    ):
        return run_fast_task(session, query, cfg, conversation_id=conv_id)


class TestFastTracePersistence:
    """The fast path must persist a non-empty trace when tracing is on."""

    def test_trace_json_persisted_with_one_entry_per_step(self, tmp_path):
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
            result = _patched_run_fast_task(session, cfg, "pick the box", conv_id)

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

    def test_no_trace_written_when_tracing_off(self, tmp_path):
        session = _make_session()
        cfg = RobotAgentConfig()
        cfg.model_spec = ModelSpec()
        cfg.exec_mode = "fast"
        cfg.enable_tracing = False  # default — zero overhead, no trace file
        cfg.workspace = str(tmp_path)
        cfg.enable_visual_feedback = False
        cfg.enable_skill = True

        with session:
            result = _patched_run_fast_task(session, cfg, "noop", "fast-off")

        assert result["ok"] is True
        traces_dir = Path(tmp_path) / "traces"
        assert not traces_dir.exists() or not list(traces_dir.glob("*.json"))

    def test_trace_run_token_embeds_conversation_id(self, tmp_path):
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
            _patched_run_fast_task(session, cfg, "pick", conv_id)

        json_files = list((Path(tmp_path) / "traces").glob("*.json"))
        assert len(json_files) == 1
        # run_token = sanitized conversation_id + timestamp + pid — starts with cid.
        assert json_files[0].name.startswith(conv_id)
