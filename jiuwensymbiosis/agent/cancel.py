# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Run cancellation primitives — a thread-safe token plus helpers to make a
blocking call abandonable.

Framework-only, stdlib-only (no ``jiuwensymbiosis`` / ``openjiuwen`` imports, so
it is a leaf module safe to import from ``agent`` / ``gui`` without a cycle). A
run attaches a ``CancelToken`` to the session; framework enforcement points
either poll it (``raise_if_set``) or wrap a blocking adapter call in
``cancellable_call``. Adapters never see it.

When no token is attached (``token is None``) every helper here is a strict
pass-through to the original synchronous call, so non-GUI runs (CLI, tests) are
byte-for-byte unchanged and ``KeyboardInterrupt`` handling is untouched.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["RunCancelled", "CancelToken", "cancellable_call", "sleep_cancellable"]


class RunCancelled(Exception):
    """Raised on the worker thread when the user cancels the active run.

    Subclasses ``Exception`` (NOT ``RuntimeError`` / ``BaseException``) on
    purpose: ``except RuntimeError`` sites pass it through untouched, while the
    few ``except Exception`` sites that must not swallow it guard with an
    explicit ``except RunCancelled: raise``. Modelling it on ``BaseException``
    would slip past those teardown/emit sites and leak resources.
    """


class CancelToken:
    """A settable cancel flag plus a registry of best-effort resource closers.

    ``set()`` flips the flag and fires every registered closer (e.g. an httpx
    ``client.close`` or a subprocess ``terminate``) so a call blocked in a C
    read can be interrupted at its source, not merely noticed at the next poll.
    Reading (``is_set`` / ``raise_if_set``) is what framework poll-loops use.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._closers: list[Callable[[], None]] = []

    def set(self) -> None:
        """Flip the flag and fire all registered closers (idempotent)."""
        self._event.set()
        with self._lock:
            closers = list(self._closers)
            self._closers.clear()
        for closer in closers:
            self._fire(closer)

    def is_set(self) -> bool:
        return self._event.is_set()

    def raise_if_set(self) -> None:
        """Raise ``RunCancelled`` if cancellation has been requested."""
        if self._event.is_set():
            raise RunCancelled

    def on_cancel(self, closer: Callable[[], None]) -> Callable[[], None]:
        """Register a best-effort closer to interrupt an in-flight blocking call.

        Fires immediately if the token is already set. Returns an unregister
        callable so a caller can drop the closer once its resource is closed
        (e.g. after a normal LLM response), keeping the registry from holding
        stale handles.
        """
        with self._lock:
            already = self._event.is_set()
            if not already:
                self._closers.append(closer)
        if already:
            self._fire(closer)
            return lambda: None

        def _unregister() -> None:
            with self._lock:
                try:
                    self._closers.remove(closer)
                except ValueError:
                    pass

        return _unregister

    @staticmethod
    def _fire(closer: Callable[[], None]) -> None:
        """Run one closer; a failing closer must never break cancellation."""
        try:
            closer()
        except Exception as exc:
            logger.debug("cancel closer %r failed: %s", getattr(closer, "__name__", closer), exc)


def cancellable_call(fn: Callable[[], Any], token: CancelToken | None, *, poll: float = 0.05) -> Any:
    """Run ``fn`` but abandon the wait if ``token`` trips, so the worker yields
    within ``poll`` seconds regardless of whether ``fn`` cooperates.

    ``token is None`` → call ``fn`` directly on the current thread (strict
    pass-through, no helper thread). Otherwise ``fn`` runs in a daemon helper
    thread; on cancel we raise ``RunCancelled`` immediately and leave the helper
    to finish (or error out via a registered closer) in the background.
    """
    if token is None:
        return fn()

    box: dict[str, Any] = {}

    def _worker() -> None:
        try:
            box["out"] = fn()
        except Exception as exc:  # surfaced on the caller thread below
            box["err"] = exc

    thread = threading.Thread(target=_worker, name="jiuwen-cancellable", daemon=True)
    thread.start()
    while True:
        thread.join(poll)
        if not thread.is_alive():
            break
        if token.is_set():
            raise RunCancelled
    if "err" in box:
        raise box["err"]
    return box.get("out")


def sleep_cancellable(seconds: float, token: CancelToken | None, *, poll: float = 0.05) -> None:
    """Sleep ``seconds``, raising ``RunCancelled`` if ``token`` trips; ``None`` → plain ``time.sleep``."""
    if token is None:
        time.sleep(seconds)
        return
    end = time.monotonic() + seconds
    while True:
        token.raise_if_set()
        remaining = end - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(poll, remaining))
