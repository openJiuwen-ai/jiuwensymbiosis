"""Admission tests use file locks and subprocesses, never devices."""

import subprocess
import sys

import pytest

from jiuwensymbiosis.agent.lifecycle import CleanupReport
from jiuwensymbiosis.runtime.resources import ResourceBlockedError, ResourceBusyError, ResourceManager


def test_process_slot_spans_managers_and_workspaces(tmp_path):
    first = ResourceManager(tmp_path / "locks")
    second = ResourceManager(tmp_path / "other")
    lease = first.acquire(("camera:123", "can:can0"))
    try:
        with pytest.raises(ResourceBusyError):
            second.acquire(("serial:/dev/ttyACM0",))
    finally:
        first.finish(lease, CleanupReport())
    next_lease = second.acquire(("serial:/dev/ttyACM0",))
    second.finish(next_lease, CleanupReport())


def test_uncertain_cleanup_keeps_admission_closed(tmp_path):
    manager = ResourceManager(tmp_path)
    lease = manager.acquire(("can:can0",))
    try:
        assert not manager.finish(lease, CleanupReport(pending_work=("motion",)))
        assert manager.records()[0]["state"] == "blocked"
        with pytest.raises(ResourceBusyError):
            manager.acquire(("can:can0",))
    finally:
        assert manager.finish(lease, CleanupReport())
    assert not manager.records()
    assert manager.finish(lease, CleanupReport())


def test_dead_process_is_blocked_until_explicit_reconciliation(tmp_path):
    script = """
import sys
from jiuwensymbiosis.runtime.resources import ResourceManager
m = ResourceManager(sys.argv[1])
lease = m.acquire(('can:can0',))
print(lease.operation_id, flush=True)
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, check=True)
    operation_id = result.stdout.strip().splitlines()[-1]
    manager = ResourceManager(tmp_path)
    with pytest.raises(ResourceBlockedError):
        manager.acquire(("can:can0",))
    with pytest.raises(ResourceBlockedError):
        manager.reconcile(operation_id, CleanupReport(errors=("unknown torque",)))
    manager.reconcile(operation_id, CleanupReport())
    lease = manager.acquire(("can:can0",))
    manager.finish(lease, CleanupReport())


def test_conflicting_process_rolls_back_entire_set(tmp_path):
    manager = ResourceManager(tmp_path)
    lease = manager.acquire(("z-camera",))
    script = """
import sys
from jiuwensymbiosis.runtime.resources import ResourceManager, ResourceBusyError
m = ResourceManager(sys.argv[1])
try:
    m.acquire(('a-device', 'z-camera'))
except ResourceBusyError:
    pass
else:
    raise AssertionError('admitted conflicting camera')
lease = m.acquire(('a-device',))
from jiuwensymbiosis.agent.lifecycle import CleanupReport
m.finish(lease, CleanupReport())
"""
    try:
        subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, check=True)
        assert [item["resource"] for item in manager.records()] == ["z-camera"]
    finally:
        manager.finish(lease, CleanupReport())


def test_late_finish_cannot_clear_a_new_generation(tmp_path):
    manager = ResourceManager(tmp_path)
    old = manager.acquire(("can:can0",))
    manager.finish(old, CleanupReport())
    current = manager.acquire(("can:can0",))
    try:
        manager.finish(old, CleanupReport())
        assert manager.records()[0]["generation"] == current.generation
        with pytest.raises(ResourceBusyError):
            manager.reconcile(current.operation_id, CleanupReport())
    finally:
        manager.finish(current, CleanupReport())


@pytest.mark.parametrize("keys", [(), ("",), (None,), ("ok", 1)])
def test_invalid_keys_fail_before_claiming_slot(tmp_path, keys):
    with pytest.raises(ValueError):
        ResourceManager(tmp_path).acquire(keys)


def test_durable_record_failure_rolls_back_before_connect(tmp_path, monkeypatch):
    manager = ResourceManager(tmp_path)
    original = manager._write

    def fail_second(key, record):
        if key == "b":
            raise OSError("disk full")
        original(key, record)

    monkeypatch.setattr(manager, "_write", fail_second)
    with pytest.raises(OSError, match="disk full"):
        manager.acquire(("a", "b"))
    assert not manager.records()
    monkeypatch.setattr(manager, "_write", original)
    lease = manager.acquire(("a", "b"))
    manager.finish(lease, CleanupReport())
