# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared admission, cleanup reporting, and bounded events for maintenance tools."""

from __future__ import annotations

import queue
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jiuwensymbiosis.agent.lifecycle import CleanupReport

__all__ = [
    "BoundedEventQueue",
    "MaintenanceOwner",
    "append_cleanup_errors",
    "cleanup_report_for",
    "cleanup_report_message",
    "finish_admission",
]


class BoundedEventQueue:
    """A non-blocking FIFO that bounds preview/log memory and reports dropped events.

    It implements the ``put`` / ``get_nowait`` subset used by workbench engines
    and their log handlers. On overflow it drops the oldest event. The next read
    first yields ``("gap", {"dropped": n})`` so consumers can detect the loss.
    """

    def __init__(self, maxsize: int = 1000) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be positive")
        self.maxsize = maxsize
        self._items: deque[Any] = deque()
        self._dropped = 0
        self._lock = threading.Lock()

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        """Add an event without waiting; ``block`` / ``timeout`` are queue-compatible."""
        del block, timeout
        with self._lock:
            if len(self._items) >= self.maxsize:
                self._items.popleft()
                self._dropped += 1
            self._items.append(item)

    def get_nowait(self) -> Any:
        """Return a pending gap report before the retained events, or raise Empty."""
        with self._lock:
            if self._dropped:
                dropped = self._dropped
                self._dropped = 0
                return "gap", {"dropped": dropped}
            if not self._items:
                raise queue.Empty
            return self._items.popleft()

    def qsize(self) -> int:
        """Return the number of retained events, excluding a pending synthetic gap."""
        with self._lock:
            return len(self._items)

    def empty(self) -> bool:
        """Whether there is neither a retained event nor a pending gap report."""
        with self._lock:
            return not self._items and not self._dropped


@dataclass(frozen=True)
class MaintenanceOwner:
    """Bind a maintenance run to its selected config source and Runtime lease.

    Engines pass the final config snapshot they will execute. Preparation freezes
    environment-derived defaults and adapter-owned resource identities before a
    thread starts. Maintenance bindings omit detector sidecars because these
    tools connect their camera/robot session directly.
    """

    runtime: Any
    config_source: Path
    workspace: Path
    adapter: str | None

    def __init__(
        self, runtime: Any, config_source: str | Path, workspace: str | Path, *, adapter: str | None = None
    ) -> None:
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "config_source", Path(config_source).expanduser().resolve(strict=False))
        object.__setattr__(self, "workspace", Path(workspace).expanduser().resolve(strict=False))
        object.__setattr__(self, "adapter", adapter)

    def acquire(
        self,
        config_snapshot: Mapping[str, Any],
        operation: str,
        *,
        stop=None,
    ) -> tuple[Any, Any]:
        """Prepare actual execution config and reserve its complete device set."""
        from jiuwensymbiosis.runtime.bindings import prepare_binding

        config_snapshot = dict(config_snapshot)
        if self.adapter is not None:
            if config_snapshot.get("adapter", self.adapter) != self.adapter:
                raise ValueError("配置中的 adapter 与当前选择的本体不一致，请重新选择本体和配置。")
            config_snapshot.setdefault("adapter", self.adapter)
        binding = prepare_binding(
            self.config_source,
            config_snapshot=config_snapshot,
            workspace=self.workspace,
            include_sidecars=False,
        )
        lease = self.runtime.acquire_maintenance(binding, operation, stop=stop)
        return binding, lease

    def finish(self, lease: Any, report: CleanupReport) -> bool:
        """Release the reservation only when the cleanup report is clear."""
        return bool(self.runtime.finish_maintenance(lease, report))


def cleanup_report_for(session: Any) -> CleanupReport:
    """Retry session teardown and return conservative, public cleanup evidence.

    A session must expose its public cleanup report before a resource is declared
    released. A disconnect exception or invalid/missing report is converted to
    blocked evidence instead of being mistaken for successful release.
    """
    if session is None:
        return CleanupReport()

    errors: list[str] = []
    disconnect = getattr(session, "disconnect", None)
    if callable(disconnect):
        try:
            disconnect()
        except BaseException as exc:
            errors.append(f"session disconnect failed: {type(exc).__name__}: {exc}")

    report_fn = getattr(session, "cleanup_report", None)
    if not callable(report_fn):
        report = CleanupReport(
            errors=("session has no public cleanup_report; hardware release cannot be confirmed",),
            connected=True,
        )
    else:
        try:
            report = report_fn()
        except BaseException as exc:
            report = CleanupReport(errors=(f"cleanup report failed: {type(exc).__name__}: {exc}",), connected=True)
        if not isinstance(report, CleanupReport):
            report = CleanupReport(errors=("session returned an invalid cleanup report",), connected=True)

    return append_cleanup_errors(report, errors)


def append_cleanup_errors(report: CleanupReport, errors: tuple[str, ...] | list[str]) -> CleanupReport:
    """Return a copy of ``report`` carrying additional uncertainty evidence."""
    if not errors:
        return report
    return CleanupReport(
        pending_work=report.pending_work,
        errors=(*report.errors, *errors),
        connected=report.connected,
    )


def cleanup_report_message(report: CleanupReport) -> str:
    """Format the fields that keep a maintenance reservation from being released."""
    details = list(report.errors)
    if report.pending_work:
        details.append(f"pending work: {', '.join(report.pending_work)}")
    if report.connected:
        details.append("session still connected")
    return "; ".join(details) or "cleanup is not confirmed"


def finish_admission(owner: MaintenanceOwner | None, lease: Any, report: CleanupReport) -> tuple[bool, CleanupReport]:
    """Apply cleanup evidence to the optional Runtime lease without losing errors."""
    if owner is None or lease is None:
        return report.released, report
    try:
        released = owner.finish(lease, report)
        if not released and report.released:
            report = append_cleanup_errors(report, ["runtime did not confirm maintenance resource release"])
        return released, report
    except BaseException as exc:
        report = append_cleanup_errors(report, [f"resource release record failed: {type(exc).__name__}: {exc}"])
        return False, report
