# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Resource admission for official local hardware entry points."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from jiuwensymbiosis.agent.lifecycle import CleanupReport, HardwareCleanupError
from jiuwensymbiosis.runtime.resources import ResourceBlockedError, ResourceManager

__all__ = ["admitted_session"]


def _disconnect_report(session: Any) -> tuple[CleanupReport, BaseException | None]:
    """Disconnect and collect conservative evidence, retaining unknown state."""
    failures: list[BaseException] = []
    disconnect = getattr(session, "disconnect", None)
    if not callable(disconnect):
        failures.append(TypeError("session does not expose disconnect()"))
    else:
        try:
            disconnect()
        except BaseException as exc:
            failures.append(exc)

    report_fn = getattr(session, "cleanup_report", None)
    if not callable(report_fn):
        report = CleanupReport(errors=("session does not expose cleanup_report()",), connected=True)
    else:
        try:
            report = report_fn()
        except BaseException as exc:
            failures.append(exc)
            report = CleanupReport(connected=True)

    if not isinstance(report, CleanupReport):
        failures.append(TypeError("session.cleanup_report() did not return CleanupReport"))
        report = CleanupReport(connected=True)

    if failures:
        failure_messages = tuple(f"{type(exc).__name__}: {exc}" for exc in failures)
        report = CleanupReport(
            pending_work=report.pending_work,
            errors=(*report.errors, *failure_messages),
            connected=True,
        )
    return report, failures[0] if failures else None


@contextmanager
def admitted_session(
    binding: Any,
    *,
    session: Any | None = None,
    resource_manager: ResourceManager | None = None,
    operation: str = "cli-task",
    unsafe_cleanup_errors: tuple[type[BaseException], ...] = (),
) -> Iterator[Any]:
    """Acquire a binding's resources before connecting and release only if clean.

    ``session`` can supply an existing, unconnected adapter session when an
    official workflow must preserve its own session factory (calibration does
    this). With no supplied session, ``binding.build_session()`` runs after
    admission. Offline callers should build from a binding directly and skip
    this context manager.

    Disconnect and cleanup evidence are always checked before finishing the
    lease. Missing, invalid, or failed cleanup evidence is recorded as blocked;
    it is never treated as a successful release. Entry points may pass
    ``unsafe_cleanup_errors`` for domain exceptions that prove recovery failed
    even when the session itself cannot observe that failure.
    """
    manager = resource_manager or ResourceManager()
    lease = manager.acquire(binding.resources, operation=operation)
    active_session = session
    operation_error: BaseException | None = None
    unsafe_cleanup_error: BaseException | None = None
    try:
        if active_session is None:
            active_session = binding.build_session()
        active_session.connect()
        yield active_session
    except BaseException as exc:
        operation_error = exc
        if isinstance(exc, (HardwareCleanupError, *unsafe_cleanup_errors)):
            unsafe_cleanup_error = exc
        raise
    finally:
        report = CleanupReport()
        cleanup_error: BaseException | None = None
        if active_session is not None:
            report, cleanup_error = _disconnect_report(active_session)
        if unsafe_cleanup_error is not None:
            report = CleanupReport(
                pending_work=report.pending_work,
                errors=(
                    *report.errors,
                    f"unsafe recovery failed: {type(unsafe_cleanup_error).__name__}: {unsafe_cleanup_error}",
                ),
                connected=True,
            )
        released = manager.finish(lease, report)
        if not released:
            cause = cleanup_error or operation_error
            error = ResourceBlockedError(
                f"resource cleanup is not confirmed; resources remain blocked: {lease.resources}; "
                + "; ".join(report.errors or report.pending_work or ("session is still connected",))
            )
            if cause is not None:
                raise error from cause
            raise error
