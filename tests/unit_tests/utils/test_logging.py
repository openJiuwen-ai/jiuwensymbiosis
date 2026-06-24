# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.utils.logging."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

from jiuwensymbiosis.utils.logging import (
    DEFAULT_FMT,
    TraceLogHandler,
    configure_logging,
    get_logger,
)


class TestConfigureLogging:
    def teardown_method(self):
        """Restore root logger between tests."""
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)
        root.setLevel(logging.NOTSET)

    def test_attaches_stream_handler(self):
        configure_logging(level="INFO")
        root = logging.getLogger()
        assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)

    def test_idempotent_no_duplicate_handlers(self):
        configure_logging(level="INFO")
        n = len(logging.getLogger().handlers)
        configure_logging(level="INFO")
        assert len(logging.getLogger().handlers) == n

    def test_file_handler_written_when_log_dir_given(self, tmp_path):
        configure_logging(level="INFO", log_dir=str(tmp_path))
        log = logging.getLogger("jiuwensymbiosis.test_logging")
        log.warning("hello-from-test")
        log_file = Path(tmp_path) / "jiuwensymbiosis.log"
        assert log_file.exists()
        assert "hello-from-test" in log_file.read_text(encoding="utf-8")

    def test_sets_root_level(self):
        configure_logging(level="WARNING")
        assert logging.getLogger().level == logging.WARNING

    def test_uses_uniform_format(self):
        configure_logging(level="INFO")
        root = logging.getLogger()
        # Filter to OUR owned stream handler (pytest injects its own
        # LogCaptureHandler, which must not be mistaken for ours).
        ours = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
            and getattr(h, "_jiuwensymbiosis_owned", False)
        ]
        assert ours, "configure_logging should attach an owned stream handler"
        fmt = ours[0].formatter._fmt
        assert "%(name)s" in fmt
        assert "%(levelname)s" in fmt


class TestGetLogger:
    def test_returns_logger_by_name(self):
        log = get_logger("jiuwensymbiosis.foo")
        assert isinstance(log, logging.Logger)
        assert log.name == "jiuwensymbiosis.foo"

    def test_default_name_is_module(self):
        log = get_logger()
        assert log.name.startswith("tests.")


class TestTraceLogHandler:
    def teardown_method(self):
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)
        root.setLevel(logging.NOTSET)

    def test_emits_warning_to_sink(self):
        sink = MagicMock()
        handler = TraceLogHandler(sink=sink)
        handler.setLevel(logging.WARNING)
        log = logging.getLogger("jiuwensymbiosis.trace_test")
        log.setLevel(logging.DEBUG)
        log.addHandler(handler)
        log.warning("boom")
        assert sink.record_log_event.called
        kwargs = sink.record_log_event.call_args.kwargs
        assert kwargs["level"] == "WARNING"
        assert "boom" in kwargs["msg"]

    def test_ignores_below_threshold(self):
        sink = MagicMock()
        handler = TraceLogHandler(sink=sink)
        handler.setLevel(logging.WARNING)
        log = logging.getLogger("jiuwensymbiosis.trace_test2")
        log.setLevel(logging.DEBUG)
        log.addHandler(handler)
        log.info("quiet")
        assert not sink.record_log_event.called

    def test_no_sink_does_not_raise(self):
        handler = TraceLogHandler(sink=None)
        handler.setLevel(logging.WARNING)
        record = logging.LogRecord(
            "x", logging.WARNING, __file__, 1, "msg", None, None,
        )
        handler.emit(record)  # must not raise

    def test_bad_sink_does_not_raise(self):
        # A sink whose record_log_event raises must not propagate (logging
        # handlers must never raise). Covers the precise-exception fallback.
        class _BadSink:
            def record_log_event(self, **kwargs):
                raise TypeError("boom")

        handler = TraceLogHandler(sink=_BadSink())
        handler.setLevel(logging.WARNING)
        record = logging.LogRecord(
            "x", logging.WARNING, __file__, 1, "msg", None, None,
        )
        handler.emit(record)  # must not raise

    def test_formatter_repr_fallback_on_bad_record(self):
        from jiuwensymbiosis.utils.logging import _Formatter

        class _BadRecord(logging.LogRecord):
            def getMessage(self):
                raise ValueError("bad")

        fmt = _Formatter("%(message)s")
        record = _BadRecord("x", logging.INFO, __file__, 1, "msg", None, None)
        out = fmt.format(record)
        assert isinstance(out, str)
