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
from threading import Lock, Thread
from typing import Any

from jiuwensymbiosis.agent.cancel import CancelToken, RunCancelled
from jiuwensymbiosis.agent.lifecycle import CleanupReport, HardwareCleanupError
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.env.base import DERIVED_CAPABILITIES, BaseRobotEnv

logger = logging.getLogger(__name__)


def _env_capabilities(env: Any) -> set[str]:
    """What the env offers, on the SAME basis the tool gate uses.

    ``build_robot_tools`` reads ``effective_capabilities`` (declared + what the body
    SHIPS); reading plain ``capabilities`` here would report a mismatch the gate does
    not see. Falls back for a duck-typed env, matching ``tools/builder.py``.
    """
    caps = getattr(env, "effective_capabilities", None) or getattr(env, "capabilities", None) or frozenset()
    return set(caps)


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


def _startup_cleanup_confirmed(sidecar: Any) -> bool:
    """Failed entry is reusable only with an explicit report from its owner."""
    report_fn = getattr(sidecar, "cleanup_report", None)
    if not callable(report_fn):
        return False
    try:
        report = report_fn()
    except Exception:
        return False
    return isinstance(report, CleanupReport) and report.released


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
            If entry fails, an optional ``cleanup_report() -> CleanupReport``
            may confirm startup rollback; absent evidence keeps cleanup unknown.
        extra_globals: Extra names exposed to ``InProcessCodeTool``-executed
            code. The default exposes ``env`` and ``api``; add ``np``,
            ``time``, your own helpers here.
        strict_capabilities: When True, raise ``ValueError`` on connect if the
            api declares capabilities the env does not (a clear config error —
            an action was implemented without updating the env, or the env's
            hardware capabilities changed). ``env``-only capabilities (hardware
            has a feature the api doesn't surface) always stay a warning, since
            that is a missing tool, not a misconfiguration. Capabilities in
            ``DERIVED_CAPABILITIES`` are exempt from both: each side derives its
            own half and the intersection already settles it.
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

    # Root for the per-run motion log (commands.log + grasp_debug). Set as an
    # attribute by the runner before connect; None → "./jiuwen_motion_log".
    # See jiuwensymbiosis.utils.logging.begin_run.
    motion_log_dir: str | None = field(default=None, init=False, repr=False)

    _stack: ExitStack | None = field(default=None, init=False, repr=False)
    _connected: bool = field(default=False, init=False, repr=False)
    _env_needs_disconnect: bool = field(default=False, init=False, repr=False)
    _cleanup_errors: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _cleanup_error_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _disconnect_lock: Lock = field(default_factory=Lock, init=False, repr=False)
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
            with self._cleanup_error_lock:
                if not self._cleanup_errors:
                    return
        report = self.cleanup_report()
        if not report.released:
            raise RuntimeError(f"RobotSession[{self.name}] has unresolved cleanup state: {report}")
        from jiuwensymbiosis.utils.logging import begin_run

        # Establish this run's output directory before the env driver attaches a
        # command log (piper) or any grasp-debug dump lands, so all of one run's
        # motion artifacts share one folder.
        begin_run(self.motion_log_dir or "./jiuwen_motion_log")
        self._stack = ExitStack()
        # The starter loop is inside the try so a cancel raised between starters
        # (or inside a token-aware sidecar wait) still closes the stack, tearing
        # down any sidecar already started instead of leaking its subprocess.
        try:
            for index, starter in enumerate(self.sidecar_starters):
                if self.cancel_token is not None:
                    self.cancel_token.raise_if_set()
                cm = None
                try:
                    cm = self._enter_starter(starter)
                    if hasattr(cm, "__enter__"):
                        self._stack.enter_context(cm)
                    elif cm is not None and callable(getattr(cm, "close", None)):
                        self._stack.callback(cm.close)
                    elif callable(cm):
                        self._stack.callback(cm)
                    elif cm is not None:
                        raise TypeError("sidecar starter must return a context manager, closer, or None")
                except BaseException as exc:
                    # __enter__ has no registered ExitStack callback on failure.
                    # An owner may explicitly confirm its own startup rollback;
                    # arbitrary legacy starters remain conservatively unknown.
                    if not _startup_cleanup_confirmed(cm):
                        self._record_cleanup_error(
                            f"sidecar.start.{index}",
                            f"sidecar {index} start failed; cleanup state is unknown: {type(exc).__name__}: {exc}",
                        )
                    raise
            self._env_needs_disconnect = True
            self._connect_env()
        except BaseException as exc:
            # disconnect() defers teardown when a cancelled helper is still in
            # flight; otherwise it cleans partial env and sidecar setup now. The
            # original exception is re-raised after this lifecycle rollback.
            if isinstance(exc, HardwareCleanupError):
                self._record_cleanup_error("driver.cleanup", str(exc))
            self.disconnect()
            raise
        self._connected = True
        logger.info("RobotSession[%s] connected", self.name)

        env_caps = _env_capabilities(self.env)
        api_caps = set(self.api.capabilities)
        env_only = env_caps - api_caps
        # A derived capability is asymmetric BY DESIGN — the api half is "holds a judge",
        # the env half is "ships the model", and the intersection is the answer. Reporting
        # that asymmetry as a config error would flag every body that holds the generic
        # judge without shipping a URDF, which is a supported, deliberate state.
        api_only = api_caps - env_caps - DERIVED_CAPABILITIES
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
            fix_hint = (
                f"修复指引：在 {env_cls}.capabilities 里加入这些能力，"
                f"或从 {api_cls} 移除对应动作的 @implements / capability 声明。"
            )
            if self.strict_capabilities:
                # api declares a capability the hardware does not provide — a config
                # error (an action was implemented without updating the env, or the
                # hardware changed). Surface it loudly instead of silently dropping tools.
                message = (
                    f"RobotSession[{self.name}] strict_capabilities: api declares "
                    f"capabilities not in env: {sorted(api_only)}. "
                    f"These capabilities lack hardware support. {fix_hint}"
                )
                self.disconnect()
                raise ValueError(message)
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
        finish_work = token.register_work("env.connect")

        def _work() -> None:
            try:
                self.env.connect()
                box["done"] = True
            except BaseException as exc:  # surfaced on the caller thread below
                box["err"] = exc
                # The bounded reaper may already have exited. Preserve rollback
                # uncertainty before the helper leaves the work registry.
                if isinstance(exc, HardwareCleanupError):
                    self._record_cleanup_error("driver.cleanup", str(exc))
            finally:
                finish_work()

        thread = Thread(target=_work, name="jiuwen-env-connect", daemon=True)
        try:
            thread.start()
        except BaseException:
            finish_work()
            raise
        while True:
            thread.join(0.05)
            if not thread.is_alive():
                break
            if token.is_set():
                self._reap_abandoned_connect(thread, box)
                raise RunCancelled
        if "err" in box:
            raise box["err"]

    def _reap_abandoned_connect(self, thread: Thread, box: dict[str, Any]) -> None:
        """After a cancelled connect, wait (bounded) for the background env.connect
        to finish, then disconnect — otherwise a driver that finishes connecting in
        the background holds the CAN/serial port into the next run. The reaper itself
        stays in the token's work registry until it has cleaned the session.
        """
        token = self.cancel_token
        if token is None:
            return
        finish_work = token.register_work("env.connect cleanup")

        def _reaper() -> None:
            try:
                thread.join(_CONNECT_REAP_TIMEOUT_S)
                if thread.is_alive():
                    message = (
                        f"RobotSession[{self.name}]: env.connect is still running after "
                        f"{_CONNECT_REAP_TIMEOUT_S:g}s; hardware release is unconfirmed"
                    )
                    self._record_cleanup_error("connect_reaper", message)
                    logger.warning(message)
                    return
                if "err" in box:
                    logger.warning(
                        "RobotSession[%s]: abandoned env.connect later failed: %s",
                        self.name,
                        box["err"],
                    )
                pending = list(token.pending_work)
                try:
                    pending.remove("env.connect cleanup")
                except ValueError:
                    pass
                if pending:
                    message = (
                        f"RobotSession[{self.name}]: cannot reap env.connect while other work "
                        f"is pending: {tuple(pending)}"
                    )
                    self._record_cleanup_error("connect_reaper", message)
                    logger.warning(message)
                    return
                # The helper has joined, so env.connect can no longer race this
                # disconnect. This reaper is itself registered, so bypass the
                # public pending-work guard while retaining its registration.
                self._disconnect_resources(allow_pending=True)
            except BaseException as exc:
                self._record_cleanup_error(
                    "connect_reaper",
                    f"RobotSession[{self.name}] connect reaper failed: {type(exc).__name__}: {exc}",
                )
                logger.exception("RobotSession[%s]: connect reaper failed", self.name)
                if not isinstance(exc, Exception):
                    raise
            finally:
                finish_work()

        try:
            Thread(target=_reaper, name="jiuwen-env-reap", daemon=True).start()
        except BaseException as exc:
            finish_work()
            message = f"RobotSession[{self.name}] could not start connect reaper: {type(exc).__name__}: {exc}"
            self._record_cleanup_error("connect_reaper", message)
            logger.error(message)
            if not isinstance(exc, Exception):
                raise

    def _record_cleanup_error(self, key: str, message: str) -> None:
        with self._cleanup_error_lock:
            self._cleanup_errors[key] = message

    def _clear_cleanup_error(self, key: str) -> None:
        with self._cleanup_error_lock:
            self._cleanup_errors.pop(key, None)

    def cleanup_report(self) -> CleanupReport:
        """Return conservative evidence about pending work and released resources."""
        connected = self._connected or self._env_needs_disconnect or self._stack is not None
        pending_work = () if self.cancel_token is None else self.cancel_token.pending_work
        with self._cleanup_error_lock:
            errors = tuple(self._cleanup_errors[key] for key in sorted(self._cleanup_errors))
        return CleanupReport(pending_work=pending_work, errors=errors, connected=connected)

    def _disconnect_resources(self, *, allow_pending: bool = False) -> None:
        """Attempt teardown; only the joined-connect reaper may bypass pending work."""
        with self._disconnect_lock:
            if not allow_pending and self.cancel_token is not None and self.cancel_token.pending_work:
                return

            # Full trace teardown: flush any pending trace (safety net; the rail
            # normally finalizes in its after_invoke hook) AND detach the log
            # handler. Keep the reference on failure so a later disconnect can retry.
            if self._trace_rail is not None:
                try:
                    self._trace_rail.close()
                except BaseException as exc:
                    message = f"RobotSession[{self.name}] trace close failed: {type(exc).__name__}: {exc}"
                    self._record_cleanup_error("trace.close", message)
                    logger.warning(message)
                    if not isinstance(exc, Exception):
                        raise
                else:
                    self._trace_rail = None
                    self._clear_cleanup_error("trace.close")

            if self._env_needs_disconnect or self._connected:
                try:
                    self.env.disconnect()
                except BaseException as exc:
                    message = f"RobotSession[{self.name}] env.disconnect failed: {type(exc).__name__}: {exc}"
                    self._record_cleanup_error("env.disconnect", message)
                    logger.warning(message)
                    if not isinstance(exc, Exception):
                        raise
                else:
                    self._connected = False
                    self._env_needs_disconnect = False
                    self._clear_cleanup_error("env.disconnect")
                    self._clear_cleanup_error("connect_reaper")

            if self._stack is not None:
                stack = self._stack
                try:
                    stack.close()
                except BaseException as exc:
                    message = f"RobotSession[{self.name}] sidecar cleanup failed: {type(exc).__name__}: {exc}"
                    self._record_cleanup_error("sidecar.cleanup", message)
                    logger.warning(message)
                    if not isinstance(exc, Exception):
                        raise
                else:
                    self._clear_cleanup_error("sidecar.cleanup")
                finally:
                    # ExitStack has consumed its callbacks even if one raised. The
                    # failure record remains because the sidecar's final state is unknown.
                    self._stack = None

    def disconnect(self) -> None:
        """Disconnect env and sidecars; defer teardown while work remains in flight.

        Cleanup failures stay visible through :meth:`cleanup_report`. A later call
        retries env/trace cleanup; an ExitStack sidecar failure remains blocked
        because its final process state cannot be confirmed.
        """
        self._disconnect_resources()
        report = self.cleanup_report()
        if report.released:
            logger.info("RobotSession[%s] disconnected", self.name)
        elif report.pending_work:
            logger.info(
                "RobotSession[%s] disconnect deferred; work is still pending: %s", self.name, report.pending_work
            )
        else:
            logger.warning("RobotSession[%s] cleanup is not confirmed: %s", self.name, report.errors)

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
        env_caps = _env_capabilities(self.env)
        api_caps = set(self.api.capabilities)
        return {
            "name": self.name,
            "env": getattr(self.env, "name", type(self.env).__name__),
            "env_capabilities": sorted(env_caps),
            "api_capabilities": sorted(api_caps),
            "effective_capabilities": sorted(env_caps & api_caps),
        }
