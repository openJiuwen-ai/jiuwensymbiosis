# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""RobotSession — the lifecycle bag the rails and tools share.

A ``RobotSession`` owns:
- the env (hardware driver instance)
- the api (capability-mixin object that calls into env)
- optional sidecar processes (e.g. detection server)
- a ``globals_provider`` for ``InProcessCodeTool``: returns the dict
  injected as code-exec globals.

Lifecycle: ``with session: ...`` connects/disconnects the env and starts/
stops sidecars. Idempotent.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from threading import Thread
from typing import Any

from jiuwensymbiosis.agent.cancel import CancelToken, RunCancelled
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.env.base import BaseRobotEnv

logger = logging.getLogger(__name__)

# Cap on how long the connect reaper waits for an abandoned env.connect to finish
# before giving up (env.connect self-bounds via its own enable timeouts, so this
# is a defensive backstop, not the normal path).
_CONNECT_REAP_TIMEOUT_S = 30.0


def _starter_accepts_token(starter: Callable[..., Any]) -> bool:
    """True if the sidecar starter takes a positional arg (the cancel token)."""
    try:
        params = inspect.signature(starter).parameters
    except (TypeError, ValueError):
        return False
    return any(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL) for p in params.values())


@dataclass
class RobotSession:
    """Container for one robot+api+sidecars unit, with shared globals.

    Attributes:
        env: ``BaseRobotEnv`` instance (already constructed; not yet connected).
        api: ``BaseRobotApi`` instance bound to ``env``.
        name: Used in logging, prompts, and tool prefixes.
        sidecar_starters: Callables returning a context manager / closer.
            Each is entered on ``connect`` and exited on ``disconnect``.
            Use this for the detection subprocess, video recorder, etc.
        extra_globals: Extra names exposed to ``InProcessCodeTool``-executed
            code. The default exposes ``env`` and ``api``; add ``np``,
            ``time``, your own helpers here.
        strict_capabilities: When True, raise ``ValueError`` on connect if the
            api declares capabilities the env does not (a clear config error —
            a Mixin was added without updating the env, or the env's hardware
            capabilities changed). ``env``-only capabilities (hardware has a
            feature the api doesn't surface) always stay a warning, since that
            is a missing tool, not a misconfiguration.
    """

    env: BaseRobotEnv
    api: BaseRobotApi
    name: str = "robot"
    sidecar_starters: list[Callable[[], Any]] = field(default_factory=list)
    extra_globals: dict[str, Any] = field(default_factory=dict)
    strict_capabilities: bool = False

    # Run-scoped cancel token (GUI-only). Set as an attribute by the runner before
    # connect; framework enforcement points read it. None → no cancellation wiring,
    # identical behaviour for CLI / tests.
    cancel_token: CancelToken | None = field(default=None, init=False, repr=False)

    _stack: ExitStack | None = field(default=None, init=False, repr=False)
    _connected: bool = field(default=False, init=False, repr=False)
    # Optional TraceRail (set by build_robot_agent when enable_tracing). Flushed
    # on disconnect as a safety net in case after_invoke didn't fire.
    _trace_rail: Any = field(default=None, init=False, repr=False)

    # ----------------------------------------------------------- context manager
    def __enter__(self) -> RobotSession:
        """Enter context: connect and return self."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Exit context: disconnect."""
        self.disconnect()

    # ----------------------------------------------------------------- lifecycle
    def connect(self) -> None:
        """Connect the env and start all sidecars. Idempotent."""
        if self._connected:
            return
        self._stack = ExitStack()
        # The starter loop is inside the try so a cancel raised between starters
        # (or inside a token-aware sidecar wait) still closes the stack, tearing
        # down any sidecar already started instead of leaking its subprocess.
        try:
            for starter in self.sidecar_starters:
                if self.cancel_token is not None:
                    self.cancel_token.raise_if_set()
                cm = self._enter_starter(starter)
                if hasattr(cm, "__enter__"):
                    self._stack.enter_context(cm)
            self._connect_env()
        except Exception:
            self._stack.close()
            self._stack = None
            raise
        self._connected = True
        logger.info("RobotSession[%s] connected", self.name)

        env_caps = set(self.env.capabilities)
        api_caps = set(self.api.capabilities)
        env_only = env_caps - api_caps
        api_only = api_caps - env_caps
        if env_only:
            logger.warning(
                "RobotSession[%s]: env has capabilities not declared by api: %s. "
                "These capabilities will not generate tools.",
                self.name,
                sorted(env_only),
            )
        if api_only:
            env_cls = type(self.env).__name__
            api_cls = type(self.api).__name__
            fix_hint = f"修复指引：在 {env_cls}.capabilities 里加入这些能力，或从 {api_cls} 移除对应的 Mixin。"
            if self.strict_capabilities:
                # api declares a capability the hardware does not provide — a
                # config error (Mixin added without updating env, or hardware
                # changed). Surface it loudly instead of silently dropping tools.
                self._connected = False
                if self._stack is not None:
                    self._stack.close()
                    self._stack = None
                raise ValueError(
                    f"RobotSession[{self.name}] strict_capabilities: api declares "
                    f"capabilities not in env: {sorted(api_only)}. "
                    f"These capabilities lack hardware support. {fix_hint}"
                )
            logger.warning(
                "RobotSession[%s]: api declares capabilities not in env: %s. "
                "These capabilities lack hardware support. %s",
                self.name,
                sorted(api_only),
                fix_hint,
            )

    def _enter_starter(self, starter: Callable[..., Any]) -> Any:
        """Call a sidecar starter, passing the cancel token if it accepts one.

        Starters are backward-compatible zero-arg callables by default; the shared
        detector starter opts in by accepting an optional token so its model-load
        wait can be interrupted. Arity is detected so custom zero-arg starters keep
        working unchanged.
        """
        if self.cancel_token is not None and _starter_accepts_token(starter):
            return starter(self.cancel_token)
        return starter()

    def _connect_env(self) -> None:
        """Connect the env. With a cancel token, run it in a helper thread so the
        worker can abandon the wait within one poll; a reaper then frees the driver
        (CAN/serial port) so the next run reconnects cleanly. Without a token this
        is a plain ``self.env.connect()``.
        """
        token = self.cancel_token
        if token is None:
            self.env.connect()
            return
        box: dict[str, Any] = {}

        def _work() -> None:
            try:
                self.env.connect()
                box["done"] = True
            except Exception as exc:  # surfaced on the caller thread below
                box["err"] = exc

        thread = Thread(target=_work, name="jiuwen-env-connect", daemon=True)
        thread.start()
        while True:
            thread.join(0.05)
            if not thread.is_alive():
                break
            if token.is_set():
                self._reap_abandoned_connect(thread)
                raise RunCancelled
        if "err" in box:
            raise box["err"]

    def _reap_abandoned_connect(self, thread: Thread) -> None:
        """After a cancelled connect, wait (bounded) for the background env.connect
        to finish, then disconnect — otherwise a driver that finishes connecting in
        the background holds the CAN/serial port into the next run.
        """
        env = self.env
        name = self.name

        def _reaper() -> None:
            thread.join(_CONNECT_REAP_TIMEOUT_S)
            if thread.is_alive():
                logger.warning("RobotSession[%s]: abandoned env.connect still running; port may stay held", name)
                return
            try:
                env.disconnect()
            except Exception as exc:
                logger.warning("RobotSession[%s]: reaper env.disconnect failed: %s", name, exc)

        Thread(target=_reaper, name="jiuwen-env-reap", daemon=True).start()

    def disconnect(self) -> None:
        """Disconnect the env and stop all sidecars. Idempotent."""
        if not self._connected:
            return
        # Full trace teardown: flush any pending trace (safety net; the rail
        # normally finalizes in its after_invoke hook) AND detach the log
        # handler so it isn't left dangling on long-lived loggers. Uses close()
        # rather than finalize() so the handler doesn't leak across builds.
        if self._trace_rail is not None:
            try:
                self._trace_rail.close()
            except (OSError, TypeError, ValueError, AttributeError) as exc:
                logger.warning("RobotSession[%s] trace close failed: %s", self.name, exc)
            self._trace_rail = None
        try:
            self.env.disconnect()
        except Exception as exc:  # noqa: BLE001
            logger.warning("RobotSession[%s] env.disconnect failed: %s", self.name, exc)
        if self._stack is not None:
            self._stack.close()
            self._stack = None
        self._connected = False
        logger.info("RobotSession[%s] disconnected", self.name)

    # ------------------------------------------------------------------- globals
    def attach_trace_rail(self, trace_rail: Any) -> None:
        """Bind a TraceRail so ``disconnect`` flushes + detaches it on teardown.

        Set by ``build_robot_agent`` when tracing is enabled. Safe to overwrite
        a prior rail (the old one is dropped — ``disconnect`` finalizes the
        currently-attached one).
        """
        self._trace_rail = trace_rail

    def globals_provider(self) -> dict[str, Any]:
        """Return the dict that ``InProcessCodeTool`` injects on every run.

        Re-evaluated per call so updates to ``extra_globals`` (rare) propagate.
        """
        import numpy as np

        return {
            "env": self.env,
            "api": self.api,
            "np": np,
            **self.extra_globals,
        }

    # --------------------------------------------------------------- description
    def describe(self) -> dict[str, Any]:
        """JSON-able summary. ``effective_capabilities`` (env ∩ api) gates tools."""
        env_caps = set(self.env.capabilities)
        api_caps = set(self.api.capabilities)
        return {
            "name": self.name,
            "env": getattr(self.env, "name", type(self.env).__name__),
            "env_capabilities": sorted(env_caps),
            "api_capabilities": sorted(api_caps),
            "effective_capabilities": sorted(env_caps & api_caps),
        }
