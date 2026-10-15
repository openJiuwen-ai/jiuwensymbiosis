# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Centralised logging configuration for jiuwensymbiosis.

Before this module existed, every package file called
``logging.getLogger(__name__)`` and there was **no** central ``dictConfig`` /
``basicConfig`` — log formatting varied and the only file output was the
Piper-specific ``_attach_cmd_log_handler`` (a hand-rolled module-level singleton).

This module provides:

- :func:`configure_logging` — idempotent namespace setup sharing one
  ``StreamHandler`` and an optional ``RotatingFileHandler`` across core and GUI.
- :func:`get_logger` — thin alias so new code imports from one place; existing
  ``logging.getLogger(__name__)`` calls keep working unchanged.
- :class:`TraceLogHandler` — a ``logging.Handler`` that forwards ``WARNING``+
  records to a bound :class:`~jiuwensymbiosis.agent.trace.TraceRail` so key log
  lines (rail warnings, detector failures, …) land in the execution trace
  without touching business code.
- :func:`begin_run` / :func:`current_run_dir` — the per-run output directory
  (``<root>/<stamp>``) shared by the piper command log and the grasp-debug
  dumps, so one run's on-disk artifacts land under one folder.

Official entry points configure logging before connecting; builders and task
dispatch repeat it safely for Python callers. Upstream logging keeps its own
outputs and format. An explicit upstream level applies to console and files;
omitting it preserves the current upstream policy.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import ExitStack
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Protocol

DEFAULT_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"
_LOG_FILE_NAME = "jiuwensymbiosis.log"
# Sentinel attached to handlers we own, so configure_logging can recognise its
# own handlers on a repeat call and avoid stacking them.
_OWNED_TAG = "_jiuwensymbiosis_owned"
_FRAMEWORK_NAMESPACES = ("jiuwensymbiosis", "jiuwensymbiosis_gui")
_pending_handler_closes: set[logging.Handler] = set()

# Per-run output directory shared by the piper command log and grasp-debug
# dumps. Set at RobotSession.connect(); None until the first run begins.
_run_dir: Path | None = None

LevelLike = int | str


class _TraceSinkLike(Protocol):
    def record_log_event(
        self,
        *,
        logger_name: str,
        level: str,
        msg: str,
        ts: float,
        step: int | None = ...,
    ) -> None:
        """Forward one log record (WARNING+ by default) into the active execution trace."""
        ...


def _to_level(level: LevelLike) -> int:
    if isinstance(level, int):
        return level
    name = str(level).upper()
    try:
        return logging.getLevelNamesMapping()[name]
    except KeyError:
        raise ValueError(f"Unknown logging level: {level!r}") from None


class _Formatter(logging.Formatter):
    """Uniform formatter that uses ``repr`` fallback for non-str messages."""

    def format(self, record: logging.LogRecord) -> str:
        try:
            return super().format(record)
        except (ValueError, TypeError, KeyError):
            return repr(record.msg)


class TraceLogHandler(logging.Handler):
    """Forward ``WARNING``+ log records to a bound trace sink.

    Attached to the loggers named in ``RobotAgentConfig.modules.tracing.capture_loggers``
    (default ``["jiuwensymbiosis"]``) when tracing is enabled. Each emitted
    record becomes a ``log_event`` entry on the current trace step (or the
    trace-level log when no step is active).

    The sink is optional — when ``None`` the handler is a no-op, so it can be
    constructed eagerly without a trace.
    """

    def __init__(self, *, sink: _TraceSinkLike | None, level: int = logging.WARNING) -> None:
        super().__init__(level=level)
        self._sink = sink

    def set_sink(self, sink: _TraceSinkLike | None) -> None:
        """Swap the bound sink (e.g. per invoke)."""
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        sink = self._sink
        if sink is None:
            return
        try:
            msg = record.getMessage()
            sink.record_log_event(
                logger_name=record.name,
                level=record.levelname,
                msg=msg,
                ts=record.created,
            )
        except (AttributeError, TypeError, ValueError):
            # sink is protocol-shaped but misbehaves; logging must never raise.
            pass


def _owned_handlers() -> list[logging.Handler]:
    # The same file handler serves two disjoint trees; deduplicate before close.
    candidates: list[logging.Handler] = []
    for name in _FRAMEWORK_NAMESPACES:
        candidates.extend(logging.getLogger(name).handlers)
    return list(dict.fromkeys(h for h in candidates if getattr(h, _OWNED_TAG, False)))


def _close_retired_handlers(handlers: list[logging.Handler]) -> None:
    """Attempt every close; keep failed, detached handlers for the next call."""
    _pending_handler_closes.update(handlers)
    errors: list[OSError] = []
    for handler in list(_pending_handler_closes):
        try:
            handler.close()
        except OSError as exc:
            exc.add_note("New logging configuration is active; retired handler cleanup will be retried.")
            errors.append(exc)
        else:
            _pending_handler_closes.remove(handler)
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise ExceptionGroup("Failed to close retired logging handlers", errors)


def configure_logging(
    level: LevelLike = "INFO",
    *,
    log_dir: str | Path | None = None,
    openjiuwen_level: LevelLike | None = None,
    fmt: str = DEFAULT_FMT,
) -> None:
    """Configure independent Symbiosis and upstream levels without stacking handlers.

    Args:
        level: Symbiosis logger and owned handler level (name or int).
        log_dir: Directory for jiuwensymbiosis.log (5 MB, 3 backups), or None
            for console only. Changing the directory closes the previous file.
        openjiuwen_level: Upstream console/file severity, independent of level.
            Backend, destinations, format and rotation remain upstream-owned.
            None preserves the current policy, including any prior override;
            it does not restore earlier settings. NOTSET is an explicit level.
        fmt: Symbiosis logging format string.

    Both framework namespace roots stop propagation. Root and foreign handlers
    remain untouched; collectors must attach directly to the desired namespace.
    File preparation errors preserve the prior configuration. Cleanup errors
    propagate after the new configuration is active and are retried next time.
    """
    int_level = _to_level(level)
    upstream_level = _to_level(openjiuwen_level) if openjiuwen_level is not None else None

    formatter = _Formatter(fmt, datefmt=_DATEFMT)
    loggers = [logging.getLogger(name) for name in _FRAMEWORK_NAMESPACES]
    owned = _owned_handlers()
    target = (Path(log_dir).expanduser().resolve() / _LOG_FILE_NAME) if log_dir is not None else None
    active: list[logging.Handler] = []
    with ExitStack() as rollback:
        if target is not None:
            file_handler = next(
                (h for h in owned if isinstance(h, RotatingFileHandler) and Path(h.baseFilename) == target), None
            )
            if file_handler is None:
                target.parent.mkdir(parents=True, exist_ok=True)
                file_handler = RotatingFileHandler(target, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
                rollback.callback(file_handler.close)
            active.append(file_handler)
        stream = next((h for h in owned if not isinstance(h, logging.FileHandler)), None)
        if stream is None:
            stream = logging.StreamHandler(stream=sys.stderr)
            rollback.callback(stream.close)
        active.append(stream)

        # No logger or live handler changes until all filesystem work succeeds.
        if upstream_level is not None:
            from jiuwensymbiosis._openjiuwen_logging import configure_openjiuwen_logging

            configure_openjiuwen_logging(upstream_level, fallback_formatter=formatter)
        for handler in active:
            setattr(handler, _OWNED_TAG, True)
            handler.setFormatter(formatter)
            # Explicit DEBUG children (Piper) still need output-side gating.
            handler.setLevel(int_level)
        for logger in loggers:
            # NOTSET would inherit root; 1 admits every standard severity.
            logger.setLevel(int_level if int_level != logging.NOTSET else 1)
            logger.propagate = False
            for handler in active:
                logger.addHandler(handler)
            for handler in owned:
                if handler not in active:
                    logger.removeHandler(handler)
        rollback.pop_all()
    _close_retired_handlers([h for h in owned if h not in active])


def begin_run(root: str | Path) -> Path:
    """Start a new per-run output directory ``<root>/<stamp>`` and return it.

    Called once per run at ``RobotSession.connect()``. The stamp
    (``YYYY-MM-DD_HH-MM-SS``) matches the legacy motion-log folder name. The
    directory is created lazily by whoever writes into it (grasp-debug dump /
    command-log handler), so a run that writes nothing leaves no empty folder.
    Two runs in the same second stay distinct: the name is disambiguated
    (``<stamp>-2`` …) against both what's on disk and the previous run dir.

    Single-run assumption: the framework runs one task at a time (the GUI
    serialises; the detector sidecar port is a process singleton), so this
    module-level state is safe without locking.
    """
    global _run_dir
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    base = Path(root)
    run_dir = base / stamp
    suffix = 2
    while run_dir.exists() or run_dir == _run_dir:
        run_dir = base / f"{stamp}-{suffix}"
        suffix += 1
    _run_dir = run_dir
    return run_dir


def current_run_dir(default_root: str | Path = "./jiuwen_motion_log") -> Path:
    """Return the current per-run output directory.

    Lazily begins a run under ``default_root`` for callers that never went
    through ``RobotSession.connect()`` (ad-hoc detector scripts, unit tests).
    """
    if _run_dir is None:
        return begin_run(default_root)
    return _run_dir


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger.

    Equivalent to ``logging.getLogger(name or __name__)``; centralised so future
    structured-logging work has one choke point. New code should use this; legacy
    ``logging.getLogger(__name__)`` calls remain valid.
    """
    if name is None:
        import inspect as _inspect

        frame = _inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        mod_name = (
            getattr(caller, "f_globals", {}).get("__name__", "jiuwensymbiosis")
            if caller is not None
            else "jiuwensymbiosis"
        )
        return logging.getLogger(mod_name)
    return logging.getLogger(name)


__all__ = [
    "_OWNED_TAG",
    "DEFAULT_FMT",
    "TraceLogHandler",
    "begin_run",
    "configure_logging",
    "current_run_dir",
    "get_logger",
]
