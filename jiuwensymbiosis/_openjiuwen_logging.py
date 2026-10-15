# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Configure upstream severity through public APIs, independently of Symbiosis.

Both console and file outputs use the requested level. Backend, destinations,
format and rotation remain upstream-owned. Changing severity may reset upstream
loggers; an unchanged configuration does not reopen their files. No backend
internals or lifecycle monkeypatches are used.
"""

from __future__ import annotations

import copy
import logging
import sys


class _PlainConsoleHandler(logging.StreamHandler):
    """Fallback for plain openjiuwen.* loggers, separate from Symbiosis outputs."""

    def filter(self, record: logging.LogRecord) -> bool:
        source: logging.Logger | None = logging.getLogger(record.name)
        while source is not None:
            if any(
                h is not self and isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                for h in source.handlers
            ):
                # A reachable native/host console already owns this output.
                return False
            if not source.propagate:
                break
            source = source.parent
        return super().filter(record)


def configure_openjiuwen_logging(level: int, *, fallback_formatter: logging.Formatter | None = None) -> None:
    """Apply one upstream level to existing and future console/file outputs.

    Public configuration snapshots preserve backend-specific output settings.
    Per-logger and sink severity overrides are deliberately unified. A later
    explicit upstream reconfiguration takes precedence until this is called
    again; there are no persistent hooks into the upstream lifecycle.
    """
    from openjiuwen.core.common.logging.log_config import configure_log_config, get_log_config_snapshot

    current = get_log_config_snapshot()
    # NOTSET would make stdlib loggers inherit host root severity. Loguru's
    # upstream adapter only accepts named standard levels, so DEBUG is its
    # lowest supported threshold. Both choices admit all standard severities.
    source_level = level if level != logging.NOTSET else 1
    backend_level = logging.DEBUG if level == logging.NOTSET and current.get("backend") == "loguru" else source_level
    updated = copy.deepcopy(current)
    updated["level"] = backend_level
    for logger_config in updated.get("loggers", {}).values():
        logger_config["level"] = backend_level
    if updated.get("backend") == "loguru":
        updated.setdefault("defaults", {})["level"] = backend_level
        for sink_config in updated.get("sinks", {}).values():
            sink_config["level"] = backend_level
    if updated != current:
        configure_log_config(updated)

    # Some upstream modules use stdlib names directly instead of LogManager.
    logger = logging.getLogger("openjiuwen")
    logger.setLevel(source_level)
    handler = next((h for h in logger.handlers if isinstance(h, _PlainConsoleHandler)), None)
    if handler is None and fallback_formatter is not None:
        handler = _PlainConsoleHandler(stream=sys.stderr)
        logger.addHandler(handler)
    if handler is not None:
        handler.setLevel(level)
        if fallback_formatter is not None:
            handler.setFormatter(fallback_formatter)
