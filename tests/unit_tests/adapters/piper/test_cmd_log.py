# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The Piper command log follows the per-run output directory.

``_attach_cmd_log_handler`` is called from the driver ctor on each connect; it
must target the current run dir (shared with grasp-debug) and re-point when a
new run begins so every run's motion lines land in that run's commands.log.
"""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

import pytest

import jiuwensymbiosis.adapters.piper.lowlevel as lowlevel
from jiuwensymbiosis.agent.lifecycle import HardwareCleanupError
from jiuwensymbiosis.utils.logging import begin_run, configure_logging

pytestmark = pytest.mark.usefixtures("isolated_framework_logging")


@pytest.fixture(autouse=True)
def command_log_state():
    import jiuwensymbiosis.utils.logging as mod

    run_dir = mod._run_dir
    yield
    lowlevel._detach_cmd_log_handler()
    lowlevel._CMD_LOG_OWNERS.clear()
    mod._run_dir = run_dir


@pytest.fixture
def driver_factory(monkeypatch):
    """Exercise driver-owned logging without connecting a vendor device."""
    monkeypatch.setitem(sys.modules, "piper_sdk", SimpleNamespace(C_PiperInterface_V2=object))
    monkeypatch.setattr(lowlevel.PiperLowLevel, "_initialize_connected_driver", lambda *_args: None)
    drivers = []

    def build(**kwargs):
        driver = lowlevel.PiperLowLevel(**kwargs)
        drivers.append(driver)
        return driver

    yield build
    for driver in reversed(drivers):
        driver.close()


class TestPiperCmdLogRedirect:
    def test_attach_targets_current_run_dir(self, tmp_path):
        run_dir = begin_run(tmp_path)
        path = lowlevel._attach_cmd_log_handler()
        assert path == run_dir / "commands.log"
        assert lowlevel._CMD_LOG_HANDLER in lowlevel.logger.handlers

    def test_attach_repoints_on_new_run(self, tmp_path):
        begin_run(tmp_path)
        first_path = lowlevel._attach_cmd_log_handler()
        first_handler = lowlevel._CMD_LOG_HANDLER

        second_run = begin_run(tmp_path)
        second_path = lowlevel._attach_cmd_log_handler()

        assert second_path == second_run / "commands.log"
        assert second_path != first_path
        assert first_handler not in lowlevel.logger.handlers  # old handler detached
        assert lowlevel._CMD_LOG_HANDLER in lowlevel.logger.handlers

    def test_attach_idempotent_within_one_run(self, tmp_path):
        begin_run(tmp_path)
        first_path = lowlevel._attach_cmd_log_handler()
        handler = lowlevel._CMD_LOG_HANDLER
        second_path = lowlevel._attach_cmd_log_handler()
        assert first_path == second_path
        assert lowlevel._CMD_LOG_HANDLER is handler  # same handler, not re-created

    def test_disabled_via_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("JIUWEN_PIPER_CMD_LOG", "0")
        begin_run(tmp_path)
        assert lowlevel._attach_cmd_log_handler() is None

    def test_command_file_cannot_override_main_logging(self, tmp_path, capsys):
        root_level = logging.getLogger().level
        configure_logging("INFO", log_dir=tmp_path / "main")
        begin_run(tmp_path / "motion")
        command_path = lowlevel._attach_cmd_log_handler()
        lowlevel.logger.debug("detailed-motion-context")
        lowlevel.logger.info("motion-progress")
        get_main = tmp_path / "main" / "jiuwensymbiosis.log"
        assert logging.getLogger().level == root_level
        assert logging.getLogger("jiuwensymbiosis").level == logging.INFO
        assert "detailed-motion-context" in command_path.read_text()
        for output in (get_main.read_text(), capsys.readouterr().err):
            assert "detailed-motion-context" not in output
            assert output.count("motion-progress") == 1
        configure_logging("WARNING", log_dir=tmp_path / "main")
        lowlevel.logger.info("hidden-info")
        lowlevel.logger.warning("visible-warning")
        assert "hidden-info" not in get_main.read_text()
        assert "visible-warning" in get_main.read_text()


def test_last_driver_close_releases_file_and_restores_logger(tmp_path, driver_factory):
    lowlevel.logger.setLevel(logging.WARNING)
    begin_run(tmp_path)
    first, second = driver_factory(), driver_factory()
    handler = lowlevel._CMD_LOG_HANDLER
    path = lowlevel._CMD_LOG_PATH
    assert lowlevel.logger.level == logging.DEBUG
    first.close()
    assert handler in lowlevel.logger.handlers
    assert not handler.stream.closed
    second.close()
    second.close()
    assert handler not in lowlevel.logger.handlers
    assert handler.stream is None
    assert lowlevel.logger.level == logging.WARNING
    lowlevel.logger.warning("after-close")
    assert "after-close" not in path.read_text()


def test_reconnecting_in_same_run_appends_commands(tmp_path, driver_factory):
    run_dir = begin_run(tmp_path)
    first = driver_factory()
    lowlevel.logger.debug("first-connection-command")
    first.close()
    first_output = (run_dir / "commands.log").read_text()

    second = driver_factory()
    assert second._cmd_log_path == run_dir / "commands.log"
    lowlevel.logger.debug("second-connection-command")
    second.close()
    output = (run_dir / "commands.log").read_text()
    assert output.startswith(first_output)
    assert output.count("first-connection-command") == 1
    assert output.count("second-connection-command") == 1

    next_run = begin_run(tmp_path)
    third = driver_factory()
    lowlevel.logger.debug("next-run-command")
    third.close()
    assert (run_dir / "commands.log").read_text() == output
    assert "connection-command" not in (next_run / "commands.log").read_text()
    assert "next-run-command" in (next_run / "commands.log").read_text()


def test_old_driver_close_does_not_detach_new_run(tmp_path, driver_factory):
    begin_run(tmp_path)
    first = driver_factory()
    old = lowlevel._CMD_LOG_HANDLER
    begin_run(tmp_path)
    second = driver_factory()
    current = lowlevel._CMD_LOG_HANDLER
    first.close()
    assert old.stream is None
    assert current in lowlevel.logger.handlers
    second.close()
    assert current.stream is None


def test_close_preserves_external_logger_level_change(tmp_path, driver_factory):
    lowlevel.logger.setLevel(logging.INFO)
    begin_run(tmp_path)
    driver = driver_factory()
    lowlevel.logger.setLevel(logging.ERROR)
    driver.close()
    assert lowlevel.logger.level == logging.ERROR


@pytest.mark.parametrize("disabled", [True, False])
def test_disabled_or_failed_attachment_stops_previous_run(tmp_path, driver_factory, monkeypatch, disabled):
    lowlevel.logger.setLevel(logging.INFO)
    begin_run(tmp_path)
    driver_factory()
    previous_path, previous_handler = lowlevel._CMD_LOG_PATH, lowlevel._CMD_LOG_HANDLER
    new_dir = begin_run(tmp_path)
    if disabled:
        monkeypatch.setenv("JIUWEN_PIPER_CMD_LOG", "0")
    else:
        new_dir.write_text("cannot create directory here")
    driver_factory()
    assert lowlevel._CMD_LOG_HANDLER is None
    assert lowlevel._CMD_LOG_PATH is None
    assert previous_handler.stream is None
    assert lowlevel.logger.level == logging.INFO
    lowlevel.logger.warning("next-run-marker")
    assert "next-run-marker" not in previous_path.read_text()


def test_failed_constructor_releases_command_log(tmp_path, driver_factory, monkeypatch):
    lowlevel.logger.setLevel(logging.INFO)
    begin_run(tmp_path)

    def fail(*_args):
        raise ValueError("initialization failed")

    monkeypatch.setattr(lowlevel.PiperLowLevel, "_initialize_connected_driver", fail)
    with pytest.raises(ValueError, match="initialization failed"):
        driver_factory()
    assert lowlevel._CMD_LOG_HANDLER is None
    assert not lowlevel._CMD_LOG_OWNERS
    assert lowlevel.logger.level == logging.INFO


def test_close_failure_is_reported_and_retried(tmp_path, driver_factory, monkeypatch):
    lowlevel.logger.setLevel(logging.INFO)
    begin_run(tmp_path)
    driver = driver_factory()
    handler = lowlevel._CMD_LOG_HANDLER
    original_close = handler.close
    calls = 0

    def fail_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("command file close failed")
        original_close()

    monkeypatch.setattr(handler, "close", fail_once)
    with pytest.raises(HardwareCleanupError, match="command file close failed"):
        driver.close()
    assert not driver._closed
    assert driver in lowlevel._CMD_LOG_OWNERS
    assert handler not in lowlevel.logger.handlers
    assert lowlevel.logger.level == logging.INFO
    driver.close()
    assert driver._closed
    assert handler.stream is None
    assert lowlevel._CMD_LOG_HANDLER is None
    assert calls == 2
