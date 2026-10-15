# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Exercise independent logging policies against public upstream APIs."""

from __future__ import annotations

import logging
import subprocess
import sys
import textwrap
from unittest.mock import patch

import pytest
from openjiuwen.core.common.logging import LogManager, llm_logger
from openjiuwen.core.common.logging.log_config import configure_log_config, get_log_config_snapshot

from jiuwensymbiosis._openjiuwen_logging import configure_openjiuwen_logging
from jiuwensymbiosis.utils.logging import configure_logging, get_logger

pytestmark = pytest.mark.usefixtures("isolated_framework_logging")


def _upstream_config(backend, directory, console="stdout"):
    config = {
        "backend": backend,
        "level": "INFO",
        "loggers": {"common": {"level": "DEBUG"}, "strict": {"level": "ERROR"}},
    }
    if backend == "default":
        config.update(log_path=str(directory), output=["console", "file"])
    else:
        config.update(
            defaults={"level": "INFO"},
            sinks={
                "terminal": {"target": console, "level": "ERROR", "format": "{message}"},
                "detail": {"target": str(directory / "detail.log"), "level": "DEBUG", "format": "{message}"},
                "errors": {"target": str(directory / "errors.log"), "level": "ERROR", "format": "{message}"},
            },
            routes={"*": ["terminal", "detail", "errors"]},
        )
        # Older upstream releases enqueue by default; assert output without
        # racing their background writer or depending on backend internals.
        for sink in config["sinks"].values():
            sink["enqueue"] = False
    return config


@pytest.fixture(params=["default", "loguru"])
def upstream(request, tmp_path):
    configure_log_config(_upstream_config(request.param, tmp_path / "upstream"))
    return request.param


def _contents(directory):
    return "\n".join(path.read_text() for path in directory.rglob("*.log"))


def _without_levels(config):
    if isinstance(config, dict):
        return {key: _without_levels(value) for key, value in config.items() if key != "level"}
    return config


@pytest.mark.parametrize("settings", [{}, {"openjiuwen_level": None}])
@pytest.mark.parametrize("previous_override", [False, True])
def test_omitted_or_null_level_preserves_current_upstream_policy(upstream, tmp_path, settings, previous_override):
    from jiuwensymbiosis.agent import RobotAgentConfig

    if previous_override:
        configure_logging(openjiuwen_level="ERROR")
    existing = LogManager.get_logger("common")
    strict = LogManager.get_logger("strict")
    snapshot = get_log_config_snapshot()
    plain = logging.getLogger("openjiuwen")
    state = (plain.level, plain.handlers[:], plain.propagate)
    handler_states = [(h.level, h.formatter, h.filters[:]) for h in plain.handlers]
    with patch(
        "jiuwensymbiosis._openjiuwen_logging.configure_openjiuwen_logging",
        side_effect=AssertionError("omitted/null must not configure upstream"),
    ):
        configure_logging("DEBUG", log_dir=tmp_path / "sym", **settings)
        RobotAgentConfig(logging={"level": "WARNING", "dir": str(tmp_path / "sym"), **settings}).logging.configure()
    assert get_log_config_snapshot() == snapshot
    assert LogManager.get_logger("common") is existing
    assert LogManager.get_logger("strict") is strict
    assert (plain.level, plain.handlers, plain.propagate) == state
    assert [(h.level, h.formatter, h.filters[:]) for h in plain.handlers] == handler_states
    existing.debug("preserved-common-debug")
    strict.warning("preserved-strict-hidden")
    strict.error("preserved-strict-error")
    files = _contents(tmp_path / "upstream")
    assert ("preserved-common-debug" in files) == (not previous_override)
    assert "preserved-strict-hidden" not in files
    assert "preserved-strict-error" in files


@pytest.mark.parametrize("level", [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.NOTSET])
def test_upstream_level_controls_both_outputs_independently(upstream, tmp_path, capsys, level):
    # Materialize a logger before changing policy, just as an embedding host can.
    llm_logger.info("before-policy")
    snapshot = get_log_config_snapshot()
    root = logging.getLogger()
    root.setLevel(logging.CRITICAL)
    root_handlers = root.handlers[:]
    configure_logging("WARNING", log_dir=tmp_path / "sym", openjiuwen_level=level)
    capsys.readouterr()

    common = LogManager.get_logger("common")
    strict = LogManager.get_logger("strict")
    for severity, marker in ((10, "detail-marker"), (20, "info-marker"), (30, "warning-marker"), (40, "error-marker")):
        common.log(severity, marker)
    strict.warning("override-warning")
    llm_logger.info("lazy-info")
    get_logger("jiuwensymbiosis.independence").info("hidden-sym-info")
    get_logger("jiuwensymbiosis.independence").warning("sym-warning")
    get_logger("jiuwensymbiosis_gui.independence").warning("gui-warning")

    captured = capsys.readouterr()
    console = captured.out + captured.err
    files = _contents(tmp_path / "upstream")
    for severity, marker in ((10, "detail-marker"), (20, "info-marker"), (30, "warning-marker"), (40, "error-marker")):
        assert console.count(marker) == int(level <= severity)
        assert (marker in files) == (level <= severity)
    assert console.count("override-warning") == int(level <= logging.WARNING)
    assert ("override-warning" in files) == (level <= logging.WARNING)
    assert console.count("lazy-info") == int(level <= logging.INFO)
    assert ("lazy-info" in files) == (level <= logging.INFO)
    sym_output = (tmp_path / "sym" / "jiuwensymbiosis.log").read_text()
    for output in (console, sym_output):
        assert output.count("sym-warning") == 1
        assert output.count("gui-warning") == 1
        assert "hidden-sym-info" not in output
    assert "-marker" not in sym_output
    assert "sym-warning" not in files and "gui-warning" not in files
    assert _without_levels(get_log_config_snapshot()) == _without_levels(snapshot)
    assert root.level == logging.CRITICAL and root.handlers == root_handlers


def test_repeat_configuration_keeps_existing_loggers_and_outputs(upstream, tmp_path):
    configure_openjiuwen_logging(logging.WARNING)
    logger = LogManager.get_logger("common")
    logger.warning("first-write")
    snapshot = get_log_config_snapshot()
    with patch(
        "openjiuwen.core.common.logging.log_config.configure_log_config",
        side_effect=AssertionError("unchanged policy must not reset upstream"),
    ):
        configure_openjiuwen_logging(logging.WARNING)
        configure_logging("DEBUG", log_dir=tmp_path / "sym")
        configure_logging("ERROR", log_dir=tmp_path / "sym")
    assert LogManager.get_logger("common") is logger
    assert get_log_config_snapshot() == snapshot
    logger.warning("second-write")
    assert "first-write" in _contents(tmp_path / "upstream")
    assert "second-write" in _contents(tmp_path / "upstream")


def test_lazy_creation_reset_and_explicit_reconfiguration(upstream, tmp_path, capsys):
    configure_openjiuwen_logging(logging.ERROR)
    llm_logger.error("initial-error")
    LogManager.reset()
    capsys.readouterr()
    llm_logger.warning("reset-hidden")
    llm_logger.error("reset-visible")
    LogManager.get_logger("future-module").warning("future-hidden")
    captured = capsys.readouterr()
    assert "hidden" not in captured.out + captured.err + _contents(tmp_path / "upstream")
    assert (captured.out + captured.err).count("reset-visible") == 1

    # An explicit host configuration wins until Symbiosis is configured again.
    configure_log_config(_upstream_config(upstream, tmp_path / "host"))
    llm_logger.info("host-info")
    assert "host-info" in _contents(tmp_path / "host")
    configure_openjiuwen_logging(logging.WARNING)
    llm_logger.info("reapplied-hidden")
    assert "reapplied-hidden" not in _contents(tmp_path / "host")


def test_file_only_routes_do_not_gain_console_output(upstream, tmp_path, capsys):
    config = get_log_config_snapshot()
    if upstream == "default":
        config["output"] = ["file"]
    else:
        config["routes"] = {"*": ["detail", "errors"]}
    configure_log_config(config)
    configure_openjiuwen_logging(logging.DEBUG)
    capsys.readouterr()
    LogManager.get_logger("common").debug("file-only-route")
    captured = capsys.readouterr()
    assert "file-only-route" not in captured.out + captured.err
    assert "file-only-route" in _contents(tmp_path / "upstream")


@pytest.mark.parametrize("host_console", [False, True])
def test_plain_loggers_use_upstream_level_without_changing_host_root(tmp_path, host_console):
    script = textwrap.dedent(
        """
        import logging
        import sys
        from jiuwensymbiosis.utils.logging import configure_logging, get_logger

        root = logging.getLogger()
        root.setLevel(logging.CRITICAL)
        host_file = logging.FileHandler('host.log')
        root.addHandler(host_file)
        if sys.argv[1] == 'True':
            root.addHandler(logging.StreamHandler())
        root_handlers = root.handlers[:]
        upstream = logging.getLogger('openjiuwen.harness.goal.manager')
        configure_logging('WARNING', log_dir='sym', openjiuwen_level='INFO')
        upstream.info('upstream-info')
        get_logger('jiuwensymbiosis.test').info('hidden-sym-info')
        get_logger('jiuwensymbiosis.test').warning('sym-warning')
        assert root.level == logging.CRITICAL and root.handlers == root_handlers
        assert upstream.level == logging.NOTSET
        assert upstream.getEffectiveLevel() == logging.INFO

        configure_logging('DEBUG', log_dir='sym', openjiuwen_level='WARNING')
        upstream.info('hidden-upstream-info')
        upstream.warning('upstream-warning')
        own_console = logging.StreamHandler()
        upstream.addHandler(own_console)
        upstream.propagate = False
        upstream.warning('own-console-warning')
        upstream.removeHandler(own_console)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(host_console)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=45,
        check=True,
    )
    output = result.stdout + result.stderr
    for marker in ("upstream-info", "upstream-warning", "own-console-warning", "sym-warning"):
        assert output.count(marker) == 1
    assert "hidden-" not in output
    host_output = (tmp_path / "host.log").read_text()
    assert "upstream-info" in host_output and "upstream-warning" in host_output
    assert "sym-warning" not in host_output and "hidden-" not in host_output
    sym_output = (tmp_path / "sym" / "jiuwensymbiosis.log").read_text()
    assert sym_output.count("sym-warning") == 1
    assert "upstream-" not in sym_output


@pytest.mark.parametrize("backend,console", [("default", "stdout"), ("loguru", "stdout"), ("loguru", "stderr")])
def test_import_preserves_preconfigured_upstream_backend(backend, console, tmp_path):
    config = _upstream_config(backend, tmp_path / "upstream", console)
    script = textwrap.dedent("""\
        import logging
        import os
        for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
            os.environ.pop(key, None)
        # Isolate the adapter from pymilvus's unrelated import-time dictConfig.
        import pymilvus
        from openjiuwen.core.common.logging import LogManager, llm_logger
        from openjiuwen.core.common.logging.log_config import configure_log_config, get_log_config_snapshot
        configure_log_config(CONFIG)
        existing = LogManager.get_logger('common')
        strict = LogManager.get_logger('strict')
        existing.info('before-import')
        snapshot = get_log_config_snapshot()
        plain = logging.getLogger('openjiuwen')
        plain.setLevel(logging.ERROR)
        plain.addHandler(logging.NullHandler())
        plain_state = (plain.level, plain.handlers[:], plain.propagate)
        import jiuwensymbiosis
        from jiuwensymbiosis._openjiuwen_logging import configure_openjiuwen_logging
        assert get_log_config_snapshot() == snapshot
        assert LogManager.get_logger('common') is existing
        assert LogManager.get_logger('strict') is strict
        assert (plain.level, plain.handlers, plain.propagate) == plain_state
        existing.debug('preserved-debug')
        strict.warning('hidden-strict-warning')
        strict.error('visible-error')
        configure_openjiuwen_logging(30)
        llm_logger.info('hidden-explicit-info')
        configure_openjiuwen_logging(20)
        llm_logger.info('visible-info')
    """).replace("CONFIG", repr(config))
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=45, check=False
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    files = _contents(tmp_path)
    assert "hidden-" not in output + files
    assert output.count("visible-error") == 1
    assert output.count("visible-info") == 1
    assert "before-import" in files
    assert "preserved-debug" in files
    assert "visible-error" in files and "visible-info" in files
