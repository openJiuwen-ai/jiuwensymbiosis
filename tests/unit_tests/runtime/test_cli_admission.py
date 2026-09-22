# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Admission boundary tests for official command-line hardware paths."""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from jiuwensymbiosis.agent.lifecycle import CleanupReport
from jiuwensymbiosis.runtime.admission import admitted_session
from jiuwensymbiosis.runtime.resources import ResourceBlockedError, ResourceBusyError, ResourceManager


class _Session:
    def __init__(self, events: list[Any], report: CleanupReport | None = None) -> None:
        self.events = events
        self.report = report or CleanupReport()

    def connect(self) -> None:
        self.events.append("connect")

    def __enter__(self):
        self.connect()
        return self

    def disconnect(self) -> None:
        self.events.append("disconnect")

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    def cleanup_report(self) -> CleanupReport:
        self.events.append("cleanup_report")
        return self.report


class _Binding:
    def __init__(self, events: list[Any], session: _Session, resources: tuple[str, ...] = ("camera:test",)) -> None:
        self.events = events
        self.session = session
        self.resources = resources

    def build_session(self) -> _Session:
        self.events.append("build_session")
        return self.session


class _RecordingManager(ResourceManager):
    def __init__(self, directory: Path, events: list[Any]) -> None:
        super().__init__(directory)
        self.events = events
        self.acquired_resources: list[tuple[str, ...]] = []
        self.operations: list[str] = []

    def acquire(self, resources: tuple[str, ...], *, operation: str = "task"):
        self.events.append("acquire")
        self.acquired_resources.append(tuple(resources))
        self.operations.append(operation)
        return super().acquire(resources, operation=operation)

    def finish(self, lease, report):
        self.events.append(("finish", report))
        return super().finish(lease, report)


def _clear_blocked(manager: ResourceManager) -> None:
    """Give the live test owner clean evidence so process-global admission resets."""
    records = manager.records()
    if records:
        lease = manager._leases[records[0]["operation_id"]]
        manager.finish(lease, CleanupReport())


def test_admitted_session_acquires_before_session_creation_and_finishes_after_cleanup(tmp_path: Path) -> None:
    events: list[Any] = []
    session = _Session(events)
    binding = _Binding(events, session)
    manager = _RecordingManager(tmp_path, events)

    with admitted_session(binding, resource_manager=manager, operation="cli-task") as active:
        events.append("work")
        assert active is session

    assert events == [
        "acquire",
        "build_session",
        "connect",
        "work",
        "disconnect",
        "cleanup_report",
        ("finish", CleanupReport()),
    ]
    assert manager.records() == []


def test_unconfirmed_cleanup_keeps_resources_blocked(tmp_path: Path) -> None:
    events: list[Any] = []
    report = CleanupReport(errors=("torque restore failed",), connected=True)
    session = _Session(events, report)
    binding = _Binding(events, session, resources=("serial:/dev/test-arm",))
    manager = ResourceManager(tmp_path)

    try:
        with pytest.raises(ResourceBlockedError, match="cleanup is not confirmed"):
            with admitted_session(binding, resource_manager=manager, operation="calibration"):
                pass

        records = manager.records()
        assert records[0]["state"] == "blocked"
        assert records[0]["cleanup"]["errors"] == ["torque restore failed"]
        with pytest.raises(ResourceBlockedError):
            manager.acquire(binding.resources)
    finally:
        _clear_blocked(manager)


def test_domain_recovery_failure_blocks_even_when_session_reports_clean(tmp_path: Path) -> None:
    events: list[Any] = []
    session = _Session(events)
    binding = _Binding(events, session, resources=("serial:/dev/guided-arm",))
    manager = ResourceManager(tmp_path)

    class _TorqueRestoreFailure(RuntimeError):
        pass

    try:
        with pytest.raises(ResourceBlockedError, match="cleanup is not confirmed"):
            with admitted_session(
                binding,
                resource_manager=manager,
                operation="calibration-collect",
                unsafe_cleanup_errors=(_TorqueRestoreFailure,),
            ):
                raise _TorqueRestoreFailure("torque was not restored")

        record = manager.records()[0]
        assert record["state"] == "blocked"
        assert "torque was not restored" in record["cleanup"]["errors"][-1]
    finally:
        _clear_blocked(manager)


def test_connection_failure_releases_only_after_confirmed_cleanup(tmp_path: Path) -> None:
    events: list[Any] = []

    class _ConnectFailure(_Session):
        def connect(self) -> None:
            self.events.append("connect")
            raise RuntimeError("connection failed")

    session = _ConnectFailure(events)
    manager = ResourceManager(tmp_path)
    binding = _Binding(events, session)

    with pytest.raises(RuntimeError, match="connection failed"):
        with admitted_session(binding, resource_manager=manager):
            pytest.fail("connect should have failed")

    assert events == ["build_session", "connect", "disconnect", "cleanup_report"]
    assert manager.records() == []


def test_session_build_failure_releases_resources(tmp_path: Path) -> None:
    """A driver rejecting its own configuration must not block the device records.

    ``build_session`` fails before a session object exists, so there is no cleanup
    evidence to read. An ordinary exception means nothing was opened, and the
    reservation is released; only ``HardwareCleanupError`` blocks.
    """
    events: list[Any] = []

    class _BuildFailure(_Binding):
        def build_session(self) -> _Session:
            self.events.append("build_session")
            raise RuntimeError("[Piper] CAN interface 'can_left' does not exist.")

    manager = ResourceManager(tmp_path)
    binding = _BuildFailure(events, _Session(events), resources=("can:can_left", "camera:347622072814"))

    with pytest.raises(RuntimeError, match="does not exist"):
        with admitted_session(binding, resource_manager=manager):
            pytest.fail("a rejected configuration must not connect")

    assert events == ["build_session"]
    assert manager.records() == []


def test_simulated_gui_resource_lease_rejects_cli_admission_before_connect(tmp_path: Path) -> None:
    """A mock GUI owner and official CLI share the same physical-resource gate."""
    events: list[Any] = []
    manager = ResourceManager(tmp_path)
    gui_lease = manager.acquire(("can:can0", "camera:349622073363"), operation="gui-task")
    session = _Session(events)
    binding = _Binding(events, session, resources=("camera:349622073363", "can:can0"))

    try:
        with pytest.raises(ResourceBusyError):
            with admitted_session(binding, resource_manager=manager, operation="cli-task"):
                pytest.fail("conflicting CLI task must not connect")
        assert events == []
    finally:
        manager.finish(gui_lease, CleanupReport())


def test_state_cli_cannot_connect_while_gui_holds_the_device(tmp_path: Path, monkeypatch) -> None:
    """The official live state CLI consults the same gate as the GUI runtime."""
    from jiuwensymbiosis import cli, introspect

    events: list[Any] = []
    manager = ResourceManager(tmp_path)
    gui_lease = manager.acquire(("can:can0",), operation="gui-task")
    binding = _Binding(events, _Session(events), resources=("can:can0",))
    monkeypatch.setattr(introspect, "prepare_binding", lambda _config: binding)
    monkeypatch.setattr(sys, "argv", ["jiuwensymbiosis-state", "--config", "robot.yaml"])

    try:
        with pytest.raises(ResourceBusyError):
            cli.state_main(resource_manager=manager)
        assert events == []
    finally:
        manager.finish(gui_lease, CleanupReport())


@pytest.mark.parametrize("mode", ["collect", "auto", "recovery_failure"])
def test_live_calibration_cli_admits_the_loaded_session_resources(
    tmp_path: Path, monkeypatch, caplog, mode: str
) -> None:
    """Both live calibration modes lease the resources of their supplied session."""
    from scripts.calibrate import hand_eye_calib

    events: list[Any] = []
    config = tmp_path / "calibration.yaml"
    config.write_text("physical_device_id: calibration-fixture\n", encoding="utf-8")
    cfg = SimpleNamespace(
        can_port="can_loaded_fixture",
        camera_serial="camera-loaded-fixture",
        detector=SimpleNamespace(spawn=True, host="127.0.0.1", port=8114),
    )
    session = _Session(events)
    session.env = SimpleNamespace(cfg=cfg)
    device = SimpleNamespace(camera_mount="eye_in_hand", env=session.env)

    class _SessionFactory:
        def resource_keys(self, actual_cfg):
            assert actual_cfg is cfg
            return (f"can:{actual_cfg.can_port}",)

    spec = SimpleNamespace(
        package="jiuwensymbiosis.adapters.fixture",
        session_factory=_SessionFactory(),
        make_calibration_device=lambda env: device,
    )
    monkeypatch.setattr(hand_eye_calib, "clear_proxy_env", lambda: None)
    monkeypatch.setattr(hand_eye_calib, "configure_logging", lambda _debug: None)
    monkeypatch.setattr(hand_eye_calib, "load_adapter_spec_and_session", lambda _path: (spec, session))
    monkeypatch.setattr(
        hand_eye_calib,
        "workflow_dependencies",
        lambda _spec, _args, *, live: SimpleNamespace(options=object(), adapter_package=spec.package),
    )

    if mode in {"collect", "recovery_failure"}:

        def collect(*_args, **_kwargs):
            events.append("work")
            if mode == "recovery_failure":
                raise hand_eye_calib.ManualGuidanceRecoveryError(
                    "Torque was not re-enabled. Support the arm and check the motor bus."
                )

        monkeypatch.setattr(
            hand_eye_calib,
            "collect_waypoints",
            collect,
        )
        argv = ["--collect-poses", str(tmp_path / "poses.npz"), "--config", str(config)]
        operation = "calibration-collect"
    else:

        def _execute(*_args, **_kwargs):
            events.append("work")
            return SimpleNamespace(candidate=False)

        monkeypatch.setattr(
            hand_eye_calib,
            "execute_calibration",
            _execute,
        )
        argv = ["--auto", str(tmp_path / "poses.npz"), "--config", str(config), "--dry-run"]
        operation = "calibration-auto"

    manager = _RecordingManager(tmp_path / mode, events)
    if mode == "recovery_failure":
        try:
            assert hand_eye_calib.main(argv, resource_manager=manager) == hand_eye_calib.EXIT_ERROR
            assert "Support the arm and check the motor bus" in caplog.text
            assert "resources remain blocked" in caplog.text
            assert manager.records()[0]["state"] == "blocked"
        finally:
            _clear_blocked(manager)
        return
    assert hand_eye_calib.main(argv, resource_manager=manager) == hand_eye_calib.EXIT_OK

    assert events == ["acquire", "connect", "work", "disconnect", "cleanup_report", ("finish", CleanupReport())]
    assert manager.operations == [operation]
    assert manager.acquired_resources == [
        ("camera:camera-loaded-fixture", "can:can_loaded_fixture", "physical:calibration-fixture")
    ]
    assert manager.records() == []


def test_calibration_admission_uses_environment_values_captured_by_spec_loader(tmp_path: Path, monkeypatch) -> None:
    """Changing environment after loading a session cannot retarget its camera lease."""
    from scripts.calibrate._cli_common import load_adapter_spec_and_session
    from scripts.calibrate.hand_eye_calib import _prepare_calibration_binding

    config = tmp_path / "piper-calibration.yaml"
    config.write_text(
        "adapter: piper\n"
        "physical_device_id: piper-calibration-fixture\n"
        "calibration:\n"
        "  adapter_module: jiuwensymbiosis.adapters.piper\n"
        "env:\n"
        "  cfg:\n"
        "    low_level:\n"
        "      can_port: can_calibration_fixture\n"
        "      camera_serial: from-config\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CAMERA_SERIAL", "camera-captured-on-load")

    spec, session = load_adapter_spec_and_session(config)
    assert session.env.cfg.camera_serial == "camera-captured-on-load"

    monkeypatch.setenv("CAMERA_SERIAL", "camera-changed-after-load")
    binding = _prepare_calibration_binding(config, spec, session)

    assert "camera:camera-captured-on-load" in binding.resources
    assert "camera:camera-changed-after-load" not in binding.resources
    assert "can:can_calibration_fixture" in binding.resources
    assert "physical:piper-calibration-fixture" in binding.resources


def test_simulated_piper_cli_does_not_admit_physical_resources(monkeypatch) -> None:
    """The documented Piper mock path remains hardware-free and keeps its context lifecycle."""
    script = Path(__file__).resolve().parents[3] / "examples" / "run_task.py"
    spec = spec_from_file_location("_run_task_admission_test", script)
    assert spec is not None and spec.loader is not None
    run_task = module_from_spec(spec)
    spec.loader.exec_module(run_task)

    events: list[Any] = []
    session = _Session(events)
    monkeypatch.setattr(
        run_task,
        "_load_yaml",
        lambda _path: {"adapter": "piper", "agent": {"exec_mode": "stepagent"}},
    )
    monkeypatch.setattr(run_task, "_build_session", lambda _args, _raw: session)
    monkeypatch.setattr(run_task, "run_robot_task", lambda *_args, **_kwargs: {"ok": True})

    class _NoHardwareAdmission:
        def acquire(self, *_args, **_kwargs):
            pytest.fail("MockArmEnv must not claim physical hardware resources")

    assert (
        run_task.main(
            ["--config", "configs/piper/piper.yaml", "--query", "test", "--mock"],
            resource_manager=_NoHardwareAdmission(),
        )
        == 0
    )
    assert events == ["connect", "disconnect"]
