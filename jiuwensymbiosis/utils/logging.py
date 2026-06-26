# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Centralised logging configuration for jiuwensymbiosis.

Before this module existed, every package file called
``logging.getLogger(__name__)`` and there was **no** central ``dictConfig`` /
``basicConfig`` — log formatting varied and the only file output was the
Piper-specific ``_attach_cmd_log_handler`` (a hand-rolled module-level singleton).

This module provides:

- :func:`configure_logging` — idempotent root-logger setup with one
  ``StreamHandler`` (uniform format) and an optional ``RotatingFileHandler``.
- :func:`get_logger` — thin alias so new code imports from one place; existing
  ``logging.getLogger(__name__)`` calls keep working unchanged.
- :class:`TraceLogHandler` — a ``logging.Handler`` that forwards ``WARNING``+
  records to a bound :class:`~jiuwensymbiosis.agent.trace.TraceRail` so key log
  lines (rail warnings, detector failures, …) land in the execution trace
  without touching business code.

Call ``configure_logging(...)`` once at agent build time (see
``build_robot_agent``). It is idempotent: repeated calls do not stack handlers.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Protocol, Union

DEFAULT_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"
_LOG_FILE_NAME = "jiuwensymbiosis.log"
# Sentinel attached to handlers we own, so configure_logging can recognise its
# own handlers on a repeat call and avoid stacking them.
_OWNED_TAG = "_jiuwensymbiosis_owned"

LevelLike = Union[int, str]


class _TraceSinkLike(Protocol):
    def record_log_event(
        self,
        *,
        logger_name: str,
        level: str,
        msg: str,
        ts: float,
        step: int | None = ...,
    ) -> None: ...


def _to_level(level: LevelLike) -> int:
    if isinstance(level, int):
        return level
    name = str(level).upper()
    return logging.getLevelNamesMapping().get(name, logging.INFO)


class _Formatter(logging.Formatter):
    """Uniform formatter that uses ``repr`` fallback for non-str messages."""

    def format(self, record: logging.LogRecord) -> str:
        try:
            return super().format(record)
        except (ValueError, TypeError, KeyError):
            return repr(record.msg)


class TraceLogHandler(logging.Handler):
    """Forward ``WARNING``+ log records to a bound trace sink.

    Attached to the loggers named in ``RobotAgentConfig.trace_capture_loggers``
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
    root = logging.getLogger()
    return [h for h in root.handlers if getattr(h, _OWNED_TAG, False)]


class _FrameworkFilter(logging.Filter):
    """Accept only records from the ``jiuwensymbiosis.*`` logger namespace.

    Attached to the framework ``RotatingFileHandler`` so ``jiuwensymbiosis.log``
    holds only our own logs, not openjiuwen's stdlib-logging records (which
    bubble up to the root logger we attach to). Matched on the logger name's
    dotted prefix, so ``"jiuwensymbiosis"`` and ``"jiuwensymbiosis.agent"`` both
    pass; ``"openjiuwen.*"`` / ``"common"`` / ``"alembic"`` etc. are dropped.
    """

    _PREFIX = "jiuwensymbiosis"

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name
        return name == self._PREFIX or name.startswith(self._PREFIX + ".")


def configure_logging(
    level: LevelLike = "INFO",
    *,
    log_dir: str | Path | None = None,
    fmt: str = DEFAULT_FMT,
) -> None:
    """Idempotently configure the root logger with a uniform format.

    Args:
        level: Root logger level (name or int).
        log_dir: When given, also attach a ``RotatingFileHandler`` writing to
            ``<log_dir>/jiuwensymbiosis.log`` (5 MB, 3 backups). None = console only.
        fmt: ``logging`` format string.

    Idempotent: repeat calls adjust the level / format but never stack
    duplicate handlers.
    """
    root = logging.getLogger()
    int_level = _to_level(level)
    root.setLevel(int_level)

    formatter = _Formatter(fmt, datefmt=_DATEFMT)

    owned = _owned_handlers()
    if not owned:
        stream = logging.StreamHandler(stream=sys.stderr)
        stream.setFormatter(formatter)
        setattr(stream, _OWNED_TAG, True)
        root.addHandler(stream)
    else:
        for h in owned:
            h.setFormatter(formatter)

    # File handler: attach once if requested, detach if previously attached but
    # no longer requested.
    has_file = any(isinstance(h, RotatingFileHandler) and getattr(h, _OWNED_TAG, False) for h in root.handlers)
    if log_dir and not has_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_h = RotatingFileHandler(
            log_path / _LOG_FILE_NAME,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_h.setFormatter(formatter)
        # Keep our log file clean of openjiuwen's stdlib-logging noise: the
        # file handler accepts only ``jiuwensymbiosis.*`` records. The console
        # handler stays unfiltered so openjiuwen init lines still show while
        # debugging; openjiuwen's *own* (non-stdlib) log backend never reaches
        # the root logger anyway, so it is unaffected.
        file_h.addFilter(_FrameworkFilter())
        setattr(file_h, _OWNED_TAG, True)
        root.addHandler(file_h)
    elif not log_dir and has_file:
        for h in list(root.handlers):
            if isinstance(h, RotatingFileHandler) and getattr(h, _OWNED_TAG, False):
                root.removeHandler(h)
                try:
                    h.close()
                except OSError:
                    pass


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


__all__ = ["_OWNED_TAG", "DEFAULT_FMT", "TraceLogHandler", "configure_logging", "get_logger"]
