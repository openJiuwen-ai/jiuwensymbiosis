"""The official smoke must acquire the same lease as other runtime callers."""

import json
import sys
from unittest.mock import Mock

from jiuwensymbiosis.agent.lifecycle import CleanupReport
from jiuwensymbiosis.runtime.resources import ResourceManager
from jiuwensymbiosis.simulators.robosim import RobosimEnv
from scripts import robosim_smoke


def test_busy_service_rejects_smoke_before_connection(tmp_path, monkeypatch):
    monkeypatch.setenv("JIUWENSYMBIOSIS_RUNTIME_DIR", str(tmp_path / "runtime"))
    cfg = tmp_path / "robot.yaml"
    cfg.write_text("backend: robosim\nservice_url: http://127.0.0.1:28471\n")
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["smoke", "--config", str(cfg), "--output", str(output)])
    connect = Mock(side_effect=AssertionError("must acquire before connecting"))
    monkeypatch.setattr(RobosimEnv, "connect", connect)
    manager = ResourceManager()
    lease = manager.acquire(("robosim:local:28471",), operation="existing-task")
    try:
        assert robosim_smoke.main() == 1
        connect.assert_not_called()
        report = json.loads(output.read_text())
        assert "ResourceBusyError" in report["error"]
        assert not report["actions"]
    finally:
        manager.finish(lease, CleanupReport())


def _hold_lease(directory, connection):
    manager = ResourceManager(directory)
    lease = manager.acquire(("robosim:local:28471",), operation="other-process")
    try:
        connection.send("ready")
        connection.recv()
    finally:
        manager.finish(lease, CleanupReport())
        connection.close()


def test_other_process_service_lease_rejects_smoke_before_connection(tmp_path, monkeypatch):
    import multiprocessing

    directory = tmp_path / "runtime"
    monkeypatch.setenv("JIUWENSYMBIOSIS_RUNTIME_DIR", str(directory))
    cfg = tmp_path / "robot.yaml"
    cfg.write_text("backend: robosim\nservice_url: http://localhost:28471\n")
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["smoke", "--config", str(cfg), "--output", str(output)])
    connect = Mock(side_effect=AssertionError("must acquire before connecting"))
    monkeypatch.setattr(RobosimEnv, "connect", connect)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_hold_lease, args=(directory, child))
    process.start()
    child.close()
    try:
        assert parent.poll(15), "lease holder did not start"
        assert parent.recv() == "ready"
        assert robosim_smoke.main() == 1
        connect.assert_not_called()
        assert "ResourceBusyError" in json.loads(output.read_text())["error"]
    finally:
        if process.is_alive():
            parent.send("finish")
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(5)
        parent.close()
    assert process.exitcode == 0


def test_smoke_acquires_before_connect_and_releases_after_failure(tmp_path, monkeypatch):
    directory = tmp_path / "runtime"
    monkeypatch.setenv("JIUWENSYMBIOSIS_RUNTIME_DIR", str(directory))
    cfg = tmp_path / "robot.yaml"
    cfg.write_text("backend: robosim\nservice_url: http://127.0.0.1:28471\n")
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["smoke", "--config", str(cfg), "--output", str(output)])
    manager = ResourceManager(directory)

    def connect(env):
        assert manager.records()[0]["operation"] == "robosim-smoke"
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(RobosimEnv, "connect", connect)
    assert robosim_smoke.main() == 1
    assert "backend unavailable" in json.loads(output.read_text())["error"]
    assert manager.records() == []
