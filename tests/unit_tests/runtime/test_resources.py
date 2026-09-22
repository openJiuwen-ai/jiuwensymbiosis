"""Admission tests use file locks and subprocesses, never devices."""

import subprocess
import sys
from pathlib import Path

import pytest

from jiuwensymbiosis.agent.lifecycle import CleanupReport
from jiuwensymbiosis.runtime.resources import ResourceBlockedError, ResourceBusyError, ResourceManager


def test_process_slot_spans_managers_and_workspaces(tmp_path):
    first = ResourceManager(tmp_path / "locks")
    second = ResourceManager(tmp_path / "other")
    lease = first.acquire(("camera:123", "can:can0"))
    try:
        assert all(not handle.closed for handle in lease.handles)
        with pytest.raises(ResourceBusyError):
            second.acquire(("serial:/dev/ttyACM0",))
    finally:
        first.finish(lease, CleanupReport())
    assert all(handle.closed for handle in lease.handles)
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


@pytest.mark.parametrize(
    ("error_type", "cleanup_failure"),
    [(OSError, "unlink"), (OSError, "close"), (KeyboardInterrupt, "both")],
)
def test_acquire_rollback_continues_after_cleanup_failure(tmp_path, monkeypatch, error_type, cleanup_failure):
    manager = ResourceManager(tmp_path)
    original_error = error_type("record write failed")
    original_write = manager._write
    original_open = Path.open
    original_unlink = Path.unlink
    handles = {}
    close_callbacks = []

    def fail_third_write(key, record):
        if key == "c":
            raise original_error
        original_write(key, record)

    def fail_first_unlink(path, *args, **kwargs):
        if path == manager._path("a", ".json") and cleanup_failure in {"unlink", "both"}:
            raise PermissionError("record deletion failed")
        return original_unlink(path, *args, **kwargs)

    def fail_close():
        raise OSError("lock close failed")

    def track_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path.suffix == ".lock":
            handles[path] = handle
            close_callbacks.append(handle.close)
            if path == manager._path("a", ".lock") and cleanup_failure in {"close", "both"}:
                patcher.setattr(handle, "close", fail_close)
        return handle

    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(manager, "_write", fail_third_write)
            patcher.setattr(Path, "open", track_open)
            patcher.setattr(Path, "unlink", fail_first_unlink)
            with pytest.raises(error_type) as failed:
                manager.acquire(("a", "b", "c"))

        assert failed.value is original_error
        assert handles[manager._path("b", ".lock")].closed
        assert handles[manager._path("c", ".lock")].closed
        if cleanup_failure in {"unlink", "both"}:
            assert [record["resource"] for record in manager.records()] == ["a"]
            assert any("record deletion failed" in note for note in failed.value.__notes__)
        else:
            assert not manager.records()
        if cleanup_failure in {"close", "both"}:
            assert any("lock close failed" in note for note in failed.value.__notes__)
        else:
            assert handles[manager._path("a", ".lock")].closed
            with pytest.raises(ResourceBlockedError):
                manager.acquire(("a",))

        # Retaining the original error/handles prevents GC from hiding a leak.
        # A new lease proves both the OS locks and the process slot are available.
        lease = manager.acquire(("b", "c"))
        manager.finish(lease, CleanupReport())
    finally:
        for close in close_callbacks:
            close()


@pytest.mark.parametrize("failure", ["unlink", "close"])
def test_finish_retries_only_unreleased_resources(tmp_path, monkeypatch, failure):
    manager = ResourceManager(tmp_path)
    lease = manager.acquire(("a", "b", "c"))
    original_records = {record["resource"]: record for record in manager.records()}
    original_unlink = Path.unlink

    def fail_second_unlink(path, *args, **kwargs):
        if path == manager._path("b", ".json"):
            raise PermissionError("record deletion failed")
        return original_unlink(path, *args, **kwargs)

    def fail_close():
        raise OSError("lock close failed")

    try:
        with monkeypatch.context() as patcher:
            if failure == "unlink":
                patcher.setattr(Path, "unlink", fail_second_unlink)
            else:
                patcher.setattr(lease.handles[1], "close", fail_close)
            with pytest.raises(OSError, match="failed"):
                manager.finish(lease, CleanupReport())

        assert not lease.released
        assert lease.handles[0].closed and lease.handles[2].closed
        assert [record["resource"] for record in manager.records()] == ["b"]
        with pytest.raises(ResourceBlockedError):
            manager.acquire(("unrelated",))

        # Another process may reserve an already released resource while this
        # lease still owns b. Retrying must neither validate nor delete its record.
        replacement = {**original_records["a"], "operation_id": "new-owner", "generation": "new-generation"}
        manager._write("a", replacement)
        assert manager.finish(lease, CleanupReport())
        assert all(handle.closed for handle in lease.handles)
        assert manager.records() == [replacement]
        assert manager.finish(lease, CleanupReport())
        next_lease = manager.acquire(("b", "c"))
        manager.finish(next_lease, CleanupReport())
    finally:
        if not lease.released:
            # Also clean up against the pre-fix implementation when reproducing
            # its partial-record failure, so it cannot poison later tests.
            for key, record in original_records.items():
                manager._write(key, record)
            manager.finish(lease, CleanupReport())
