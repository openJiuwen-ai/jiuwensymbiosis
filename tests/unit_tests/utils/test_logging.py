# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.utils.logging."""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from jiuwensymbiosis.utils.logging import (
    TraceLogHandler,
    configure_logging,
    get_logger,
)

_STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(-\d+)?$")


pytestmark = pytest.mark.usefixtures("isolated_framework_logging")


class TestConfigureLogging:
    @pytest.mark.parametrize("failure", ["mkdir", "open"])
    @pytest.mark.parametrize("configured", [False, True])
    def test_file_preparation_failure_preserves_all_live_policy(self, tmp_path, monkeypatch, failure, configured):
        from openjiuwen.core.common.logging.log_config import get_log_config_snapshot

        from jiuwensymbiosis.utils import logging as log_module

        if configured:
            configure_logging("DEBUG", log_dir=tmp_path / "old", openjiuwen_level="INFO")
        loggers = [logging.getLogger(n) for n in ("", "jiuwensymbiosis", "jiuwensymbiosis_gui", "openjiuwen")]
        previous = [(log.level, log.propagate, log.handlers[:]) for log in loggers]
        handlers = {h: (h.level, h.formatter, h.filters[:]) for log in loggers for h in log.handlers}
        upstream_config = get_log_config_snapshot()
        failure_fn = Mock(side_effect=PermissionError("test filesystem failure"))
        if failure == "mkdir":
            monkeypatch.setattr(Path, "mkdir", failure_fn)
        else:
            monkeypatch.setattr(log_module.RotatingFileHandler, "_open", failure_fn)
        with pytest.raises(PermissionError, match="filesystem failure"):
            configure_logging("WARNING", log_dir=tmp_path / "new", openjiuwen_level="ERROR")
        assert [(log.level, log.propagate, log.handlers[:]) for log in loggers] == previous
        assert {h: (h.level, h.formatter, h.filters[:]) for h in handlers} == handlers
        assert get_log_config_snapshot() == upstream_config

    def test_close_failure_leaves_new_outputs_consistent_and_retries(self, tmp_path, monkeypatch, capsys):
        from jiuwensymbiosis.utils import logging as log_module

        configure_logging("DEBUG", log_dir=tmp_path / "old")
        old = next(h for h in get_logger("jiuwensymbiosis").handlers if isinstance(h, logging.FileHandler))
        real_close = old.close
        calls = 0

        def fail_once():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("test close failure")
            real_close()

        monkeypatch.setattr(old, "close", fail_once)
        with pytest.raises(OSError, match="close failure"):
            configure_logging("WARNING", log_dir=tmp_path / "new")
        child = get_logger("jiuwensymbiosis.close_failure_child")
        child.setLevel(logging.DEBUG)
        child.debug("private-context")
        child.warning("new-output")
        for output in (capsys.readouterr().err, (tmp_path / "new" / "jiuwensymbiosis.log").read_text()):
            assert "private-context" not in output
            assert output.count("new-output") == 1
        assert "new-output" not in (tmp_path / "old" / "jiuwensymbiosis.log").read_text()
        assert old in log_module._pending_handler_closes
        replacement = next(h for h in get_logger("jiuwensymbiosis").handlers if isinstance(h, logging.FileHandler))
        configure_logging("WARNING", log_dir=None)
        assert calls == 2
        assert old.stream is None
        assert replacement.stream is None
        assert old not in log_module._pending_handler_closes

    def test_child_debug_cannot_bypass_threshold(self, tmp_path, capsys):
        configure_logging("INFO", log_dir=tmp_path)
        child = get_logger("jiuwensymbiosis.explicit_debug_child")
        child.setLevel(logging.DEBUG)
        child.debug("private-context")
        child.info("progress-marker")
        for output in (capsys.readouterr().err, (tmp_path / "jiuwensymbiosis.log").read_text()):
            assert "private-context" not in output
            assert output.count("progress-marker") == 1

    def test_namespaces_and_file_retargeting(self, tmp_path, capsys):
        first, second = tmp_path / "first", tmp_path / "second"
        configure_logging(log_dir=first)
        get_logger("jiuwensymbiosis.agent").info("first-only")
        configure_logging(log_dir=second)
        for name in ("jiuwensymbiosis.cli", "jiuwensymbiosis.calibration.solver", "jiuwensymbiosis_gui.app"):
            get_logger(name).warning("second-only")
        get_logger("unrelated-library").warning("foreign-marker")
        configure_logging(log_dir=None)
        get_logger("jiuwensymbiosis.agent").warning("console-only")
        assert "second-only" not in (first / "jiuwensymbiosis.log").read_text()
        output = (second / "jiuwensymbiosis.log").read_text()
        assert output.count("second-only") == 3
        assert "foreign-marker" not in output
        assert "console-only" not in output
        assert "foreign-marker" not in capsys.readouterr().err

    def test_keeps_host_handler(self):
        host = logging.NullHandler()
        root = logging.getLogger()
        root.addHandler(host)
        try:
            configure_logging()
            configure_logging("DEBUG")
            assert host in root.handlers
            assert host.level == logging.NOTSET
        finally:
            root.removeHandler(host)

    @pytest.mark.parametrize("field", ["level", "openjiuwen_level"])
    def test_rejects_unknown_levels(self, field):
        with pytest.raises(ValueError, match="Unknown logging level"):
            configure_logging(**{field: "typo"})

    def test_attaches_stream_handler(self):
        configure_logging(level="INFO")
        for name in ("jiuwensymbiosis", "jiuwensymbiosis_gui"):
            assert any(isinstance(h, logging.StreamHandler) for h in logging.getLogger(name).handlers)

    def test_idempotent_no_duplicate_handlers(self, tmp_path):
        configure_logging(level="INFO", log_dir=tmp_path)
        core = logging.getLogger("jiuwensymbiosis")
        gui = logging.getLogger("jiuwensymbiosis_gui")
        handlers = [h for h in core.handlers if getattr(h, "_jiuwensymbiosis_owned", False)]
        assert [h for h in gui.handlers if getattr(h, "_jiuwensymbiosis_owned", False)] == handlers
        assert len(handlers) == 2
        configure_logging(level="DEBUG", log_dir=tmp_path)
        for logger in (core, gui):
            assert [h for h in logger.handlers if getattr(h, "_jiuwensymbiosis_owned", False)] == handlers
        assert all(h.level == logging.DEBUG for h in handlers)

    def test_file_handler_written_when_log_dir_given(self, tmp_path):
        configure_logging(level="INFO", log_dir=str(tmp_path))
        log = logging.getLogger("jiuwensymbiosis.test_logging")
        log.warning("hello-from-test")
        log_file = Path(tmp_path) / "jiuwensymbiosis.log"
        assert log_file.exists()
        assert "hello-from-test" in log_file.read_text(encoding="utf-8")

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "NOTSET"])
    def test_symbiosis_level_does_not_change_root_or_upstream_level(self, level):
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        handlers = root.handlers[:]
        states = [(h.level, h.formatter, h.filters[:]) for h in handlers]
        upstream = logging.getLogger("openjiuwen.harness.goal.manager")
        assert upstream.level == logging.NOTSET
        upstream_level = upstream.getEffectiveLevel()
        configure_logging(level)
        assert root.level == logging.INFO
        assert root.handlers == handlers
        assert [(h.level, h.formatter, h.filters[:]) for h in handlers] == states
        assert upstream.level == logging.NOTSET
        assert upstream.getEffectiveLevel() == upstream_level
        assert all(not logging.getLogger(n).propagate for n in ("jiuwensymbiosis", "jiuwensymbiosis_gui"))

    def test_host_root_cannot_suppress_framework_outputs(self, tmp_path, capsys):
        logging.getLogger().setLevel(logging.CRITICAL)
        for level in ("DEBUG", "NOTSET"):
            configure_logging(level, log_dir=tmp_path)
            get_logger("jiuwensymbiosis.root_isolation").debug("core-debug")
            get_logger("jiuwensymbiosis_gui.root_isolation").debug("gui-debug")
        for output in (capsys.readouterr().err, (tmp_path / "jiuwensymbiosis.log").read_text()):
            assert output.count("core-debug") == 2
            assert output.count("gui-debug") == 2
        assert logging.getLogger().level == logging.CRITICAL

    def test_framework_records_do_not_reach_host_root(self):
        host = MagicMock(spec=logging.Handler, level=logging.NOTSET)
        root = logging.getLogger()
        root.addHandler(host)
        configure_logging("INFO")
        get_logger("jiuwensymbiosis.isolation").warning("core-only")
        get_logger("jiuwensymbiosis_gui.isolation").warning("gui-only")
        host.handle.assert_not_called()
        get_logger("unrelated-library").warning("host-only")
        assert host.handle.call_args.args[0].getMessage() == "host-only"

    def test_namespace_collectors_survive_reconfiguration(self, tmp_path):
        sink = MagicMock()
        handler = TraceLogHandler(sink=sink)
        logger = get_logger("jiuwensymbiosis")
        logger.addHandler(handler)
        for level in ("INFO", "DEBUG", "WARNING"):
            configure_logging(level, log_dir=tmp_path)
            get_logger("jiuwensymbiosis.collector_test").warning("captured-warning")
        assert handler in logger.handlers
        assert sink.record_log_event.call_count == 3

    def test_uses_uniform_format(self):
        configure_logging(level="INFO")
        logger = logging.getLogger("jiuwensymbiosis")
        # Filter to OUR owned stream handler (pytest injects its own
        # LogCaptureHandler, which must not be mistaken for ours).
        ours = [
            h
            for h in logger.handlers
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


def test_configure_logging_keeps_symbiosis_and_upstream_levels_independent(tmp_path, capsys):
    from openjiuwen.core.common.logging import LogManager, llm_logger
    from openjiuwen.core.common.logging.log_config import configure_log_config, get_log_config_snapshot

    previous = get_log_config_snapshot()
    try:
        configure_log_config({"level": "INFO", "log_path": str(tmp_path / "upstream"), "output": ["console", "file"]})
        configure_logging("DEBUG", log_dir=tmp_path / "symbiosis", openjiuwen_level="WARNING")
        llm_logger.info("upstream-file-context")
        llm_logger.warning("upstream-console-warning")
        get_logger("jiuwensymbiosis.test").debug("symbiosis-debug")
        captured = capsys.readouterr()
        assert "upstream-file-context" not in captured.out + captured.err
        assert captured.out.count("upstream-console-warning") == 1
        assert "upstream-console-warning" not in captured.err
        assert "symbiosis-debug" in captured.err
        sym_file = (tmp_path / "symbiosis" / "jiuwensymbiosis.log").read_text()
        assert "symbiosis-debug" in sym_file
        assert "upstream-" not in sym_file
        upstream_files = "\n".join(p.read_text() for p in (tmp_path / "upstream").rglob("*.log"))
        assert "upstream-file-context" not in upstream_files
        configure_logging("WARNING", openjiuwen_level="INFO")
        llm_logger.info("enabled-upstream-info")
        LogManager.get_logger("future-module").info("enabled-future-info")
        get_logger("jiuwensymbiosis.test").info("hidden-symbiosis-info")
        captured = capsys.readouterr()
        assert captured.out.count("enabled-upstream-info") == 1
        assert captured.out.count("enabled-future-info") == 1
        assert "hidden-symbiosis-info" not in captured.out + captured.err
        upstream_files = "\n".join(p.read_text() for p in (tmp_path / "upstream").rglob("*.log"))
        assert "enabled-upstream-info" in upstream_files
        assert "enabled-future-info" in upstream_files
    finally:
        configure_log_config(previous)


def test_import_preserves_upstream_default_config(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; "
            "[os.environ.pop(k, None) for k in "
            "('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')]; "
            "from openjiuwen.core.common.logging.log_config import get_log_config_snapshot; "
            "before = get_log_config_snapshot(); "
            "import jiuwensymbiosis; "
            "assert get_log_config_snapshot() == before",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=45,
        check=True,
    )
    assert result.returncode == 0


def test_connection_and_failed_fast_planning_are_logged_before_agent_build(tmp_path, monkeypatch):
    from jiuwensymbiosis.agent import ModelSpec, RobotAgentConfig, run_fast_task
    from tests.helpers import make_mock_session

    config = RobotAgentConfig(model_spec=ModelSpec(), logging={"dir": str(tmp_path)})
    session = make_mock_session(name="logging-test")
    config.prepare_session(session)
    monkeypatch.setattr("jiuwensymbiosis.agent.fast.planner.parse_task", lambda *a, **kw: {"targets": []})

    def fail_plan(*args, **kwargs):
        get_logger("jiuwensymbiosis.agent.fast.planner").info("early-planning-marker")
        raise RuntimeError("offline-planning-failure")

    monkeypatch.setattr("jiuwensymbiosis.agent.fast.plan_task", fail_plan)
    with session:
        result = run_fast_task(session, "hidden-task-query", config)
    assert not result["ok"]
    output = (tmp_path / "jiuwensymbiosis.log").read_text()
    assert "RobotSession[logging-test] connected" in output
    assert "early-planning-marker" in output
    assert "offline-planning-failure" in output
    assert "hidden-task-query" not in output


def test_task_cli_uses_one_output_and_keeps_result_context_in_debug(tmp_path, monkeypatch, capsys):
    from examples import run_task
    from tests.helpers import make_mock_session

    monkeypatch.setattr(
        run_task, "_load_yaml", lambda _: {"agent": {"logging": {"dir": str(tmp_path), "level": "INFO"}}}
    )
    monkeypatch.setattr(run_task, "_build_session", lambda *a: make_mock_session())
    monkeypatch.setattr(
        run_task,
        "run_robot_task",
        lambda *a, **kw: {"output": "final-answer", "context": "private-result-context"},
    )
    assert run_task.main(["--config", "unused.yaml", "--mock", "--query", "private-task-context"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "final-answer"
    for output in (captured.err, (tmp_path / "jiuwensymbiosis.log").read_text()):
        assert output.count("Agent result: succeeded") == 1
        assert "private-task-context" not in output
        assert "private-result-context" not in output


class TestTraceLogHandler:
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
            "x",
            logging.WARNING,
            __file__,
            1,
            "msg",
            None,
            None,
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
            "x",
            logging.WARNING,
            __file__,
            1,
            "msg",
            None,
            None,
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


class TestPerRunDir:
    def teardown_method(self):
        import jiuwensymbiosis.utils.logging as mod

        mod._run_dir = None

    def test_begin_run_stamped_subdir_under_root(self, tmp_path):
        from jiuwensymbiosis.utils.logging import begin_run

        run_dir = begin_run(tmp_path)
        assert run_dir.parent == tmp_path
        assert _STAMP_RE.match(run_dir.name)

    def test_two_runs_same_second_are_distinct(self, tmp_path):
        from jiuwensymbiosis.utils.logging import begin_run

        first = begin_run(tmp_path)
        second = begin_run(tmp_path)
        assert first != second

    def test_current_run_dir_returns_last_begun(self, tmp_path):
        from jiuwensymbiosis.utils.logging import begin_run, current_run_dir

        run_dir = begin_run(tmp_path)
        assert current_run_dir() == run_dir

    def test_current_run_dir_lazy_default(self, tmp_path, monkeypatch):
        import jiuwensymbiosis.utils.logging as mod

        mod._run_dir = None
        monkeypatch.chdir(tmp_path)
        assert mod.current_run_dir().parent.name == "jiuwen_motion_log"

    def test_begin_run_does_not_create_empty_dir(self, tmp_path):
        from jiuwensymbiosis.utils.logging import begin_run

        run_dir = begin_run(tmp_path)
        assert not run_dir.exists()
