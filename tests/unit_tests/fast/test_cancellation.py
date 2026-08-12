# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cancellation wiring in the fast path: a set ``CancelToken`` must end the run
promptly at each blocking stage, without a failed step, ``_safe_retreat`` motion,
or a swallowed cancel in a retry loop."""

from __future__ import annotations

import threading
import time
import types

import pytest

from jiuwensymbiosis.agent.cancel import CancelToken, RunCancelled
from jiuwensymbiosis.agent.fast import planner as planner_module
from jiuwensymbiosis.agent.fast import runner as runner_module
from jiuwensymbiosis.agent.fast.realtime.servo import ServoConfig, ServoController
from jiuwensymbiosis.agent.fast.runner import run_sequence
from jiuwensymbiosis.agent.fast.sequence import parse_sequence


def _trip_after(token: CancelToken, delay: float) -> None:
    threading.Thread(target=lambda: (time.sleep(delay), token.set()), daemon=True).start()


def _steps(op: str):
    return parse_sequence([{"op": op, "params": {}}], allowed_ops={op})


def test_run_sequence_cancels_mid_op_without_retreat_or_failed_step(monkeypatch):
    retreats = []
    monkeypatch.setattr(runner_module, "_safe_retreat", lambda session: retreats.append(session))
    token = CancelToken()
    session = types.SimpleNamespace(api=None, env=None, cancel_token=token)

    def slow_executor(op, params):
        time.sleep(3.0)
        return {"ok": True, "result": {}}

    _trip_after(token, 0.1)
    start = time.monotonic()
    with pytest.raises(RunCancelled):
        run_sequence(session, _steps("home"), executor=slow_executor)
    assert time.monotonic() - start < 0.8
    assert retreats == []  # cancel must NOT trigger a safe-retreat motion


def test_run_sequence_cancel_before_first_step(monkeypatch):
    monkeypatch.setattr(runner_module, "_safe_retreat", lambda session: None)
    token = CancelToken()
    token.set()
    session = types.SimpleNamespace(api=None, env=None, cancel_token=token)
    called = []

    def executor(op, params):
        called.append(op)
        return {"ok": True, "result": {}}

    with pytest.raises(RunCancelled):
        run_sequence(session, _steps("home"), executor=executor)
    assert called == []  # bailed at the per-step token check, op never dispatched


def test_run_sequence_no_token_runs_normally():
    session = types.SimpleNamespace(api=None, env=None)  # no cancel_token attribute
    ran = []

    def executor(op, params):
        ran.append(op)
        return {"ok": True, "result": {}}

    result = run_sequence(session, _steps("home"), executor=executor)
    assert ran == ["home"]
    assert result["ok"] is True


def test_servo_controller_stops_within_a_tick_on_token():
    token = CancelToken()
    token.set()
    pose = {"x": 0.0, "y": 0.0, "z": 0.0, "rz": 0.0}
    far = {"x": 500.0, "y": 500.0, "z": 500.0, "rz": 0.0}  # never reached
    result = ServoController(
        lambda: dict(pose),
        lambda _p: True,
        lambda: dict(far),
        config=ServoConfig(timeout_s=1.0, absolute_timeout_s=2.0),
        target_is_live=lambda: True,
        should_continue=lambda: not token.is_set(),
    ).run()
    assert result.reason == "stopped"
    assert result.ok is False


def test_compile_sequence_bails_before_calling_chat_when_cancelled(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("_chat must not be called once cancelled")

    monkeypatch.setattr(planner_module, "_chat", _fail)
    token = CancelToken()
    token.set()
    with pytest.raises(RunCancelled):
        planner_module.compile_sequence(
            "pick the box",
            skills_md=[],
            action_vocab=["home"],
            allowed_ops={"home"},
            api_base="http://localhost:9",
            model_name="test",
            cancel_token=token,
        )


def test_chat_does_not_swallow_cancel_in_retry_loop(monkeypatch):
    # A present token + a post that raises RunCancelled must propagate, NOT be
    # caught by _chat's transient-failure retry loop.
    token = CancelToken()

    def fake_post(client, url, payload, headers, cancel_token):
        raise RunCancelled

    monkeypatch.setattr(planner_module, "_post", fake_post)
    with pytest.raises(RunCancelled):
        planner_module._chat(
            "sys",
            "user",
            api_base="http://localhost:9",
            api_key="",
            model_name="test",
            timeout_s=1.0,
            temperature=0.0,
            proxy=None,
            attempts=4,
            max_tokens=16,
            cancel_token=token,
        )
