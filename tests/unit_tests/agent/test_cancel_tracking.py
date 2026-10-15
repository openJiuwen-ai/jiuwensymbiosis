# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for observable in-flight work on ``CancelToken``."""

from __future__ import annotations

import threading

import pytest

from jiuwensymbiosis.agent.cancel import CancelToken, RunCancelled, cancellable_call


def test_register_work_returns_idempotent_finish_callback() -> None:
    token = CancelToken()
    finish = token.register_work("camera connect")

    assert token.pending_work == ("camera connect",)
    assert token.wait_for_idle(timeout=0) is False

    finish()
    finish()

    assert token.pending_work == ()
    assert token.wait_for_idle(timeout=0) is True


def test_wait_for_idle_times_out_until_registered_work_finishes() -> None:
    token = CancelToken()
    finish = token.register_work()

    assert token.wait_for_idle(timeout=0.01) is False

    finish()
    assert token.wait_for_idle(timeout=0) is True


def test_abandoned_cancellable_helper_remains_pending_until_it_finishes() -> None:
    token = CancelToken()
    helper_entered = threading.Event()
    release_helper = threading.Event()
    helper_finished = threading.Event()
    caller_finished = threading.Event()
    caller_errors: list[BaseException] = []

    def blocking_operation() -> None:
        helper_entered.set()
        try:
            release_helper.wait()
        finally:
            helper_finished.set()

    def call_and_capture_cancel() -> None:
        try:
            cancellable_call(blocking_operation, token, poll=0.005)
        except BaseException as exc:
            caller_errors.append(exc)
        finally:
            caller_finished.set()

    caller = threading.Thread(target=call_and_capture_cancel)
    caller.start()
    try:
        assert helper_entered.wait(timeout=1)
        assert token.pending_work == ("operation",)

        token.set()
        assert caller_finished.wait(timeout=1)
        assert len(caller_errors) == 1
        assert isinstance(caller_errors[0], RunCancelled)
        assert token.pending_work == ("operation",)
        assert token.wait_for_idle(timeout=0.01) is False

        release_helper.set()
        assert helper_finished.wait(timeout=1)
        assert token.wait_for_idle(timeout=1) is True
        assert token.pending_work == ()
    finally:
        release_helper.set()
        caller.join(timeout=1)


def test_cancellable_call_finishes_registration_on_normal_return_and_error() -> None:
    token = CancelToken()

    assert cancellable_call(lambda: "done", token) == "done"
    assert token.wait_for_idle(timeout=0) is True

    def fail() -> None:
        raise LookupError("helper failed")

    with pytest.raises(LookupError, match="helper failed"):
        cancellable_call(fail, token)
    assert token.pending_work == ()
    assert token.wait_for_idle(timeout=0) is True


def test_cancellable_call_rolls_back_registration_if_thread_start_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    token = CancelToken()

    def fail_start(_thread: threading.Thread) -> None:
        raise RuntimeError("thread start failed")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="thread start failed"):
        cancellable_call(lambda: "unreachable", token)

    assert token.pending_work == ()
    assert token.wait_for_idle(timeout=0) is True
