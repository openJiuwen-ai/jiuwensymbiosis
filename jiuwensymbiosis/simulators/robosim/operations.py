"""Process-local command ownership shared by clients of one loopback service.

An ambiguous write survives client closure and chat reset. Only observing a new
service instance clears that uncertainty; a successful health query is not proof
that a previously submitted robot command stopped. This is not a distributed lock.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit

from jiuwensymbiosis.errors import JiuwenSymbiosisError


class RobosimError(JiuwenSymbiosisError, RuntimeError):
    """Preserve remote failure codes and whether a write's outcome is unknown."""

    def __init__(self, message: str, *, code: str = "robosim_error", outcome_unknown: bool = False) -> None:
        super().__init__(message, code=code)
        self.outcome_unknown = outcome_unknown
        self.skip_recovery = outcome_unknown


@dataclass(frozen=True)
class CommandRecord:
    """Local command status, distinct from physical task success."""

    request_id: str
    action: str
    status: str
    error_code: str = ""


class CommandCoordinator:
    """Serialize nested synchronous commands and retain uncertain outcomes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: int | None = None
        self._depth = 0
        self._record: CommandRecord | None = None
        self._service_id: str | None = None

    def snapshot(self) -> dict:
        """Return a copy suitable for diagnostics without releasing ownership."""
        with self._lock:
            return {
                "busy": self._owner is not None,
                "write_blocked": self._record is not None and self._record.status == "outcome_unknown",
                "command": asdict(self._record) if self._record else None,
            }

    def observe_service(self, instance_id: str) -> None:
        """Reconcile a restart only when no local command is still executing."""
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError("A nonempty service instance ID is required")
        with self._lock:
            if self._service_id is not None and self._service_id != instance_id:
                if self._owner is not None:
                    raise RobosimError(
                        "Service restarted during a command", code="robosim_service_changed", outcome_unknown=True
                    )
                self._record = None
            self._service_id = instance_id

    @contextmanager
    def command(self, action: str) -> Iterator[None]:
        """Hold a write lease across dispatch and readback; never swallow cancellation."""
        owner = threading.get_ident()
        with self._lock:
            if self._owner not in {None, owner}:
                raise RobosimError("Another RoboSim command is running", code="robosim_busy")
            if self._record is not None and self._record.status == "outcome_unknown":
                raise RobosimError(
                    "Previous command outcome is unknown; inspect state and restart the owned backend before writing",
                    code="robosim_outcome_unknown",
                    outcome_unknown=True,
                )
            outer = self._owner is None
            if outer:
                self._record = CommandRecord(uuid.uuid4().hex, action, "running")
                self._owner = owner
            self._depth += 1
        try:
            yield
        except BaseException as exc:
            # This is an execution boundary: interruption cannot imply motion stopped.
            unknown = getattr(exc, "outcome_unknown", False) or not isinstance(exc, Exception)
            if outer or unknown:
                self._finish("outcome_unknown" if unknown else "failed", getattr(exc, "code", type(exc).__name__))
            raise
        else:
            if outer:
                self._finish("completed")
        finally:
            with self._lock:
                self._depth -= 1
                if self._depth == 0:
                    self._owner = None

    def _finish(self, status: str, code: str = "") -> None:
        with self._lock:
            if self._record is not None and self._record.status != "outcome_unknown":
                self._record = CommandRecord(self._record.request_id, self._record.action, status, code)


_coordinators: dict[int, CommandCoordinator] = {}
_registry_lock = threading.Lock()


def coordinator_for(service_url: str) -> CommandCoordinator:
    """Share one lease across localhost aliases, arms, clients and GUI conversations."""
    port = urlsplit(service_url).port or 80
    with _registry_lock:
        return _coordinators.setdefault(port, CommandCoordinator())
