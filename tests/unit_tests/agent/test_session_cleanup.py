# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cleanup evidence and retry behavior for ``RobotSession``."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from jiuwensymbiosis.agent.cancel import CancelToken, RunCancelled, cancellable_call
from jiuwensymbiosis.agent.lifecycle import HardwareCleanupError
from jiuwensymbiosis.agent.session import RobotSession


class _Env:
    capabilities = frozenset()
    effective_capabilities = frozenset()

    def __init__(self) -> None:
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.disconnect_failures = 0
        self.connect_error: Exception | None = None
        self.connect_entered: threading.Event | None = None
        self.connect_release: threading.Event | None = None

    def connect(self) -> None:
        self.connect_calls += 1
        if self.connect_entered is not None:
            self.connect_entered.set()
        if self.connect_release is not None:
            self.connect_release.wait()
        if self.connect_error is not None:
            raise self.connect_error
        self.connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        if self.disconnect_failures:
            self.disconnect_failures -= 1
            raise RuntimeError("env disconnect failed")
        self.connected = False


class _Sidecar:
    def __init__(self, *, fail_exit: bool = False) -> None:
        self.fail_exit = fail_exit
        self.enter_calls = 0
        self.exit_calls = 0

    def __enter__(self) -> _Sidecar:
        self.enter_calls += 1
        return self

    def __exit__(self, *_args: Any) -> None:
        self.exit_calls += 1
        if self.fail_exit:
            raise RuntimeError("sidecar exit failed")


def _make_session(
    tmp_path: Path,
    *,
    env: _Env | None = None,
    api_capabilities: frozenset[str] = frozenset(),
    sidecar_starters: list[Any] | None = None,
    strict_capabilities: bool = False,
) -> tuple[RobotSession, _Env]:
    actual_env = env or _Env()
    session = RobotSession(
        env=actual_env,  # type: ignore[arg-type]
        api=SimpleNamespace(capabilities=api_capabilities),  # type: ignore[arg-type]
        name="cleanup-test",
        sidecar_starters=sidecar_starters or [],
        strict_capabilities=strict_capabilities,
    )
    session.motion_log_dir = str(tmp_path / "motion")
    return session, actual_env


def test_cleanup_report_tracks_connected_session_and_idempotent_disconnect(tmp_path: Path) -> None:
    session, env = _make_session(tmp_path)

    assert session.cleanup_report().released is True
    session.connect()
    assert session.connect() is None
    assert env.connect_calls == 1
    assert session.cleanup_report().connected is True
    assert session.cleanup_report().released is False

    session.disconnect()
    session.disconnect()

    report = session.cleanup_report()
    assert report.released is True
    assert report.errors == ()
    assert env.disconnect_calls == 1


def test_env_disconnect_error_is_reported_and_successful_retry_clears_it(tmp_path: Path) -> None:
    env = _Env()
    env.disconnect_failures = 1
    session, _ = _make_session(tmp_path, env=env)
    session.connect()

    session.disconnect()

    failed_report = session.cleanup_report()
    assert failed_report.connected is True
    assert failed_report.released is False
    assert any("env disconnect failed" in error for error in failed_report.errors)

    session.disconnect()

    assert session.cleanup_report().released is True
    assert session.cleanup_report().errors == ()
    assert env.disconnect_calls == 2


def test_failed_connect_is_propagated_and_partial_env_disconnect_is_retryable(tmp_path: Path) -> None:
    env = _Env()
    env.connect_error = RuntimeError("env connect failed")
    env.disconnect_failures = 1
    session, _ = _make_session(tmp_path, env=env)

    with pytest.raises(RuntimeError, match="env connect failed"):
        session.connect()

    report = session.cleanup_report()
    assert report.connected is True
    assert any("env disconnect failed" in error for error in report.errors)

    env.connect_error = None
    session.disconnect()
    assert session.cleanup_report().released is True


def test_strict_capability_failure_disconnects_env_before_raising(tmp_path: Path) -> None:
    session, env = _make_session(
        tmp_path,
        api_capabilities=frozenset({"motion.joint"}),
        strict_capabilities=True,
    )

    with pytest.raises(ValueError, match="strict_capabilities"):
        session.connect()

    assert env.connect_calls == 1
    assert env.disconnect_calls == 1
    assert session.cleanup_report().released is True


def test_driver_constructor_rollback_failure_cannot_be_cleared_by_empty_env(tmp_path: Path) -> None:
    env = _Env()
    env.connect_error = HardwareCleanupError("driver.open", RuntimeError("open"), [RuntimeError("close")])
    session, _ = _make_session(tmp_path, env=env)
    with pytest.raises(HardwareCleanupError):
        session.connect()
    session.disconnect()
    report = session.cleanup_report()
    assert not report.released
    assert any("driver.open" in error for error in report.errors)


def test_sidecar_cleanup_failure_remains_conservatively_blocked(tmp_path: Path) -> None:
    sidecar = _Sidecar(fail_exit=True)
    session, env = _make_session(tmp_path, sidecar_starters=[lambda: sidecar])
    session.connect()

    session.disconnect()

    report = session.cleanup_report()
    assert env.connected is False
    assert report.connected is False
    assert report.released is False
    assert any("sidecar exit failed" in error for error in report.errors)
    assert sidecar.exit_calls == 1

    session.disconnect()
    assert session.cleanup_report().released is False
    assert sidecar.exit_calls == 1
    with pytest.raises(RuntimeError, match="unresolved cleanup state"):
        session.connect()


def test_sidecar_start_failure_is_reported_after_prior_sidecars_are_closed(tmp_path: Path) -> None:
    sidecar = _Sidecar()

    def fail_to_start() -> Any:
        raise RuntimeError("sidecar start failed")

    session, env = _make_session(tmp_path, sidecar_starters=[lambda: sidecar, fail_to_start])

    with pytest.raises(RuntimeError, match="sidecar start failed"):
        session.connect()

    report = session.cleanup_report()
    assert env.connect_calls == 0
    assert sidecar.exit_calls == 1
    assert report.connected is False
    assert report.released is False
    assert any("sidecar 1 start failed" in error for error in report.errors)


@pytest.mark.parametrize("cancel_start", [False, True])
@pytest.mark.parametrize("shutdown", ["terminate", "kill", "failed"])
def test_detector_lifecycle_reports_actual_child_exit(tmp_path, monkeypatch, cancel_start, shutdown):
    from jiuwensymbiosis.perception import detector_sidecar as detector

    class Child:
        pid = 12345
        alive = True

        def poll(self):
            return None if self.alive else 0

        def terminate(self):
            if shutdown != "terminate":
                raise OSError("terminate failed")
            self.alive = False

        def kill(self):
            if shutdown == "failed":
                raise OSError("kill failed")
            self.alive = False

        def wait(self, timeout):
            assert not self.alive
            return 0

    child = Child()
    monkeypatch.setattr(detector, "_port_open", lambda *a, **kw: False)
    monkeypatch.setattr(detector.subprocess, "Popen", lambda *a, **kw: child)

    def ready(*a, **kw):
        if cancel_start:
            raise RunCancelled()
        return True

    monkeypatch.setattr(detector, "_wait_for_port", ready)
    session, env = _make_session(tmp_path, sidecar_starters=[detector.detector_subprocess])
    if cancel_start:
        expected = HardwareCleanupError if shutdown == "failed" else RunCancelled
        with pytest.raises(expected):
            session.connect()
        assert env.connect_calls == 0
    else:
        session.connect()
    session.disconnect()
    assert child.alive is (shutdown == "failed")
    assert session.cleanup_report().released is (shutdown != "failed")
    # Neither a second disconnect nor a vanished ExitStack may erase uncertainty.
    session.disconnect()
    assert session.cleanup_report().released is (shutdown != "failed")


def test_cancelled_connect_reaper_stays_pending_through_cleanup(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()
    env = _Env()
    env.connect_entered = entered
    env.connect_release = release
    sidecar = _Sidecar()
    session, _ = _make_session(tmp_path, env=env, sidecar_starters=[lambda: sidecar])
    token = CancelToken()
    session.cancel_token = token
    caller_done = threading.Event()
    errors: list[BaseException] = []

    def connect() -> None:
        try:
            session.connect()
        except BaseException as exc:
            errors.append(exc)
        finally:
            caller_done.set()

    caller = threading.Thread(target=connect)
    caller.start()
    try:
        assert entered.wait(timeout=1)
        token.set()
        assert caller_done.wait(timeout=1)
        assert len(errors) == 1
        assert isinstance(errors[0], RunCancelled)
        assert {"env.connect", "env.connect cleanup"}.issubset(token.pending_work)

        session.disconnect()
        assert env.disconnect_calls == 0
        assert sidecar.exit_calls == 0
        assert session.cleanup_report().released is False

        release.set()
        assert token.wait_for_idle(timeout=1) is True
        assert env.disconnect_calls == 1
        assert sidecar.exit_calls == 1
        assert session.cleanup_report().released is True
    finally:
        release.set()
        caller.join(timeout=1)


@pytest.mark.parametrize("rollback_failed", [False, True])
def test_reaper_timeout_keeps_blocked_evidence_until_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rollback_failed: bool
) -> None:
    from jiuwensymbiosis.agent import session as session_module

    monkeypatch.setattr(session_module, "_CONNECT_REAP_TIMEOUT_S", 0.01)
    entered = threading.Event()
    release = threading.Event()
    timeout_recorded = threading.Event()
    env = _Env()
    if rollback_failed:
        env.connect_error = HardwareCleanupError("driver constructor", cleanup_errors=(RuntimeError("CAN still open"),))
    env.connect_entered = entered
    env.connect_release = release
    session, _ = _make_session(tmp_path, env=env)
    token = CancelToken()
    session.cancel_token = token
    original_record = session._record_cleanup_error

    def record_and_signal(key: str, message: str) -> None:
        original_record(key, message)
        if key == "connect_reaper" and "still running" in message:
            timeout_recorded.set()

    monkeypatch.setattr(session, "_record_cleanup_error", record_and_signal)
    caller_done = threading.Event()
    errors: list[BaseException] = []

    def connect() -> None:
        try:
            session.connect()
        except BaseException as exc:
            errors.append(exc)
        finally:
            caller_done.set()

    caller = threading.Thread(target=connect)
    caller.start()
    try:
        assert entered.wait(timeout=1)
        token.set()
        assert caller_done.wait(timeout=1)
        assert isinstance(errors[0], RunCancelled)
        assert timeout_recorded.wait(timeout=1)

        report = session.cleanup_report()
        assert "env.connect" in report.pending_work
        assert any("hardware release is unconfirmed" in error for error in report.errors)
        assert report.connected is True
        assert report.released is False
        session.disconnect()
        assert env.disconnect_calls == 0

        release.set()
        assert token.wait_for_idle(timeout=1) is True
        session.disconnect()
        assert env.disconnect_calls == 1
        assert session.cleanup_report().released is not rollback_failed
        if rollback_failed:
            assert any("CAN still open" in error for error in session.cleanup_report().errors)
    finally:
        release.set()
        caller.join(timeout=1)


def test_disconnect_defers_while_cancellable_call_is_still_running(tmp_path: Path) -> None:
    session, env = _make_session(tmp_path)
    session.connect()
    token = CancelToken()
    session.cancel_token = token
    entered = threading.Event()
    release = threading.Event()
    caller_done = threading.Event()
    errors: list[BaseException] = []

    def blocking_operation() -> None:
        entered.set()
        release.wait()

    def invoke() -> None:
        try:
            cancellable_call(blocking_operation, token, poll=0.005)
        except BaseException as exc:
            errors.append(exc)
        finally:
            caller_done.set()

    caller = threading.Thread(target=invoke)
    caller.start()
    try:
        assert entered.wait(timeout=1)
        token.set()
        assert caller_done.wait(timeout=1)
        assert isinstance(errors[0], RunCancelled)

        session.disconnect()
        assert env.disconnect_calls == 0
        assert session.cleanup_report().pending_work == ("operation",)

        release.set()
        assert token.wait_for_idle(timeout=1) is True
        session.disconnect()
        assert env.disconnect_calls == 1
        assert session.cleanup_report().released is True
    finally:
        release.set()
        caller.join(timeout=1)
