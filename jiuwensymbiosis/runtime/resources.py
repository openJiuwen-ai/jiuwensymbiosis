"""Atomic admission for cooperating local hardware entry points.

OS locks serialize processes; durable records retain uncertainty across crashes.
Lock files are never removed because replacing an inode would split the lock.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO

from jiuwensymbiosis.agent.lifecycle import CleanupReport

__all__ = ["ResourceManager", "ResourceLease", "ResourceBusyError", "ResourceBlockedError"]

_PROCESS_LOCK = threading.RLock()
_PROCESS_OWNER: str | None = None
_PROCESS_BLOCKED = False
_PROCESS_PID = os.getpid()


class ResourceBusyError(RuntimeError):
    """A cooperating operation is already holding a required resource."""


class ResourceBlockedError(ResourceBusyError):
    """Previous cleanup has not been confirmed."""


@dataclass
class ResourceLease:
    operation_id: str
    resources: tuple[str, ...]
    operation: str
    pid: int
    generation: str
    released: bool = False
    report: CleanupReport = field(default_factory=lambda: CleanupReport(connected=True))
    handles: list[IO[str]] = field(default_factory=list, repr=False)
    released_resources: set[str] = field(default_factory=set, init=False, repr=False)


class ResourceManager:
    """Share one process slot and acquire a complete set of device locks."""

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = (
            Path(
                directory
                or os.environ.get("JIUWENSYMBIOSIS_RUNTIME_DIR")
                or Path.home() / ".jiuwensymbiosis" / "runtime"
            )
            .expanduser()
            .resolve()
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        self._leases: dict[str, ResourceLease] = {}

    def _path(self, resource: str, suffix: str) -> Path:
        return self.directory / (hashlib.sha256(resource.encode()).hexdigest() + suffix)

    def _record(self, resource: str) -> dict | None:
        path = self._path(resource, ".json")
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text())
            if not isinstance(record, dict) or not record.get("operation_id"):
                raise ValueError("invalid record")
            return record
        except (OSError, ValueError) as exc:
            raise ResourceBlockedError(f"占用记录无法核对: {resource}") from exc

    def _write(self, resource: str, record: dict) -> None:
        destination = self._path(resource, ".json")
        temp = destination.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            with temp.open("x", encoding="utf-8") as file:
                json.dump(record, file, ensure_ascii=False)
                file.flush()
                os.fsync(file.fileno())
            temp.replace(destination)
        finally:
            temp.unlink(missing_ok=True)

    def acquire(self, resources: tuple[str, ...], *, operation: str = "task") -> ResourceLease:
        """Reserve every resource or roll back, without starting any hardware."""
        global _PROCESS_OWNER, _PROCESS_PID, _PROCESS_BLOCKED
        if not resources or any(not isinstance(key, str) or not key.strip() for key in resources):
            raise ValueError("resource keys must be non-empty strings")
        keys = tuple(sorted(set(resources)))
        with _PROCESS_LOCK:
            if _PROCESS_PID != os.getpid():
                _PROCESS_PID, _PROCESS_OWNER, _PROCESS_BLOCKED = os.getpid(), None, False
            if _PROCESS_OWNER is not None:
                if _PROCESS_BLOCKED:
                    raise ResourceBlockedError("当前进程的硬件清理尚未确认")
                raise ResourceBusyError("当前进程已有任务或维护操作占用硬件")
            lease = ResourceLease(uuid.uuid4().hex, keys, operation, os.getpid(), uuid.uuid4().hex)
            written: list[str] = []
            try:
                for key in keys:
                    handle = self._path(key, ".lock").open("a+")
                    lease.handles.append(handle)
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as exc:
                        raise ResourceBusyError(f"资源正在使用: {key}") from exc
                    if self._record(key) is not None:
                        raise ResourceBlockedError(f"资源上次收尾未确认: {key}")
                for key in keys:
                    self._write(
                        key,
                        {
                            "resource": key,
                            "operation_id": lease.operation_id,
                            "generation": lease.generation,
                            "pid": lease.pid,
                            "operation": operation,
                            "state": "reserved",
                        },
                    )
                    written.append(key)
            except BaseException as error:
                # This acquisition boundary has not connected hardware. Roll back even
                # for an interrupt. A cleanup failure must not skip other resources
                # or replace the original exception (including cancellation).
                for key in written:
                    try:
                        self._path(key, ".json").unlink(missing_ok=True)
                    except BaseException as cleanup_error:
                        error.add_note(
                            f"Resource rollback could not remove record for {key!r}: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
                for held_handle in lease.handles:
                    try:
                        held_handle.close()
                    except BaseException as cleanup_error:
                        error.add_note(
                            f"Resource rollback could not close lock {held_handle.name!r}: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
                raise
            _PROCESS_OWNER = lease.operation_id
            self._leases[lease.operation_id] = lease
            return lease

    def finish(self, lease: ResourceLease, report: CleanupReport) -> bool:
        """Release confirmed resources; retry unfinished cleanup without touching new owners."""
        global _PROCESS_OWNER, _PROCESS_BLOCKED
        with _PROCESS_LOCK:
            if lease.released:
                return True
            if self._leases.get(lease.operation_id) is not lease or lease.pid != os.getpid():
                raise ValueError("lease does not belong to this manager/process")
            remaining = [
                (key, handle)
                for key, handle in zip(lease.resources, lease.handles, strict=True)
                if key not in lease.released_resources
            ]
            for key, _handle in remaining:
                record = self._record(key)
                if (
                    not record
                    or record.get("operation_id") != lease.operation_id
                    or record.get("generation") != lease.generation
                ):
                    raise ResourceBlockedError(f"占用代次不一致: {key}")
            lease.report = report
            if not report.released:
                _PROCESS_BLOCKED = True
                for key, _handle in remaining:
                    record = self._record(key)
                    # Validated moments ago while holding the same owner lock; a
                    # vanished record still fails closed rather than silently
                    # writing a blocked state for an unknown generation.
                    if record is None:
                        raise ResourceBlockedError(f"占用记录在标记阻断前消失: {key}")
                    self._write(key, {**record, "state": "blocked", "cleanup": asdict(report)})
                return False
            failures: list[tuple[str, BaseException]] = []
            for key, handle in remaining:
                try:
                    # Keep the record until close succeeds. If record deletion
                    # fails, it still blocks admission and can be validated on retry.
                    if not handle.closed:
                        handle.close()
                    self._path(key, ".json").unlink(missing_ok=True)
                except BaseException as exc:
                    # Complete the other resources even during an interrupt, then
                    # propagate the first failure with all cleanup diagnostics.
                    failures.append((key, exc))
                else:
                    # A later finish must not inspect or delete a new owner's record.
                    lease.released_resources.add(key)
            if failures:
                _PROCESS_BLOCKED = True
                error = failures[0][1]
                for key, failure in failures:
                    error.add_note(f"Resource release failed for {key!r}: {type(failure).__name__}: {failure}")
                raise error
            lease.released = True
            self._leases.pop(lease.operation_id)
            if _PROCESS_OWNER == lease.operation_id:
                _PROCESS_OWNER = None
                _PROCESS_BLOCKED = False
            return True

    def records(self) -> list[dict]:
        """Inspect durable resource records without opening device connections."""
        records = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                record = json.loads(path.read_text())
                if not isinstance(record, dict):
                    raise ValueError("record must be an object")
                records.append(record)
            except (OSError, ValueError):
                records.append({"state": "invalid", "path": str(path)})
        return records

    def reconcile(self, operation_id: str, report: CleanupReport) -> None:
        """Clear a dead owner's records only with explicit external cleanup evidence.

        Live owners must call finish() using their lease instead. This is a trusted
        operator API, never an automatic startup recovery or a browser callback.
        """
        if not report.released:
            raise ResourceBlockedError("清理仍未确认，不能解除占用")
        records = [record for record in self.records() if record.get("operation_id") == operation_id]
        if not records:
            raise KeyError(operation_id)
        handles = []
        with _PROCESS_LOCK:
            try:
                for record in records:
                    try:
                        os.kill(int(record["pid"]), 0)
                    except ProcessLookupError:
                        pass
                    else:
                        raise ResourceBusyError("原持有进程仍存活，必须由其报告收尾")
                    handle = self._path(record["resource"], ".lock").open("a+")
                    handles.append(handle)
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as exc:
                        raise ResourceBusyError("核对期间资源已被占用") from exc
                    current = self._record(record["resource"])
                    if current != record:
                        raise ResourceBlockedError("占用记录在核对期间发生变化")
                for record in records:
                    self._path(record["resource"], ".json").unlink(missing_ok=True)
            finally:
                for handle in handles:
                    handle.close()
