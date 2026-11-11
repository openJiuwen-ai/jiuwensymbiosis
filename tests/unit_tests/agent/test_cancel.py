# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Unit tests for the run-cancellation primitives (``agent/cancel.py``)."""

from __future__ import annotations

import threading
import time

import pytest

from jiuwensymbiosis.agent.cancel import CancelToken, RunCancelled, cancellable_call, sleep_cancellable


def _trip_after(token: CancelToken, delay: float) -> None:
    threading.Thread(target=lambda: (time.sleep(delay), token.set()), daemon=True).start()


def test_raise_if_set_and_is_set():
    token = CancelToken()
    assert not token.is_set()
    token.raise_if_set()  # no-op when unset
    token.set()
    assert token.is_set()
    with pytest.raises(RunCancelled):
        token.raise_if_set()


def test_set_fires_registered_closers():
    token = CancelToken()
    fired = []
    token.on_cancel(lambda: fired.append("a"))
    token.on_cancel(lambda: fired.append("b"))
    token.set()
    assert fired == ["a", "b"]


def test_on_cancel_after_set_fires_immediately():
    token = CancelToken()
    token.set()
    fired = []
    token.on_cancel(lambda: fired.append("late"))
    assert fired == ["late"]


def test_on_cancel_unregister_prevents_firing():
    token = CancelToken()
    fired = []
    unregister = token.on_cancel(lambda: fired.append("x"))
    unregister()
    token.set()
    assert fired == []


def test_failing_closer_does_not_break_cancellation():
    token = CancelToken()
    fired = []

    def boom():
        raise RuntimeError("closer blew up")

    token.on_cancel(boom)
    token.on_cancel(lambda: fired.append("after"))
    token.set()  # must not raise
    assert token.is_set()
    assert fired == ["after"]


def test_cancellable_call_none_token_is_passthrough():
    assert cancellable_call(lambda: 42, None) == 42


def test_cancellable_call_returns_value_and_reraises():
    token = CancelToken()
    assert cancellable_call(lambda: 7, token) == 7

    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        cancellable_call(boom, token)


def test_cancellable_call_abandons_slow_fn_promptly():
    token = CancelToken()
    _trip_after(token, 0.1)
    start = time.monotonic()
    with pytest.raises(RunCancelled):
        cancellable_call(lambda: time.sleep(5.0), token, poll=0.02)
    assert time.monotonic() - start < 0.5


def test_cancellable_call_prefers_completion_when_both_ready():
    token = CancelToken()
    token.set()  # already set, but fn returns instantly
    assert cancellable_call(lambda: "done", token) == "done"


def test_sleep_cancellable_none_token_sleeps():
    start = time.monotonic()
    sleep_cancellable(0.05, None)
    assert time.monotonic() - start >= 0.04


def test_sleep_cancellable_raises_promptly_on_cancel():
    token = CancelToken()
    _trip_after(token, 0.1)
    start = time.monotonic()
    with pytest.raises(RunCancelled):
        sleep_cancellable(5.0, token, poll=0.02)
    assert time.monotonic() - start < 0.5
