"""Durable device-action calls within an already admitted Runtime job.

This module owns invocation records, not instrument policy, hardware drivers,
threads or a second task state machine. Domain callbacks interpret evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from jiuwensymbiosis.agent.cancel import CancelToken, RunCancelled
from jiuwensymbiosis.runtime.resources import ResourceBlockedError, ResourceLease
from jiuwensymbiosis.runtime.store import RequestConflictError

if TYPE_CHECKING:
    from jiuwensymbiosis.runtime.jobs import Runtime

__all__ = ["InvocationContext", "InvocationOutcome", "InvocationFailed", "InvocationUnknown"]


class InvocationFailed(RuntimeError):
    """The execution owner supplied terminal failure evidence."""


class InvocationUnknown(RuntimeError):
    """No safe terminal evidence exists; the resource must remain blocked."""


@dataclass(frozen=True)
class InvocationOutcome:
    """Domain-validated execution facts, with an explicit result scope.

    ``response`` means the protocol response completed. It does not imply
    physical completion. ``physical`` additionally requires device evidence.
    A remote handle is preserved independently from the model-facing result.
    """

    status: str
    result: Any = None
    scope: str = "response"
    completion_evidence: dict[str, Any] = field(default_factory=dict)
    remote_handle: Any = None


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass
class InvocationContext:
    """An execution capability issued only after Runtime resource admission."""

    runtime: Runtime
    job_id: str
    binding_id: str
    lease: ResourceLease
    token: CancelToken
    principal: str = "local"
    _uncertain: set[str] = field(default_factory=set, init=False, repr=False)

    @property
    def pending_work(self) -> tuple[str, ...]:
        """Unresolved records prevent lease release even after socket cleanup."""
        return tuple(sorted(self._uncertain))

    def _write(self, invocation_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.runtime.store.update_invocation(invocation_id, changes)
        except Exception as exc:
            self._uncertain.add(invocation_id)
            self.token.set()
            self.runtime.storage_failed(exc)
            raise

    def _unknown(self, invocation_id: str, reason: str) -> None:
        self._uncertain.add(invocation_id)
        self.token.set()
        self._write(invocation_id, {"status": "unknown", "uncertainty": reason})

    def _result(self, record: dict[str, Any]) -> Any:
        if record["status"] == "succeeded":
            return record.get("result")
        if record["status"] == "failed":
            raise InvocationFailed(f"device action failed: {record['invocation_id']}")
        self._uncertain.add(record["invocation_id"])
        self.token.set()
        raise InvocationUnknown(f"device action unresolved: {record['invocation_id']}")

    def _check_running(self) -> None:
        if self.runtime.persistence_failed:
            self.token.set()
            raise InvocationUnknown("Runtime persistence is unavailable")
        if self.runtime.closing:
            self.token.set()
        self.token.raise_if_set()

    def invoke(
        self,
        *,
        resource_key: str,
        action_name: str,
        contract_version: str,
        parameters: dict[str, Any],
        request_key: str,
        validate: Callable[[], None],
        dispatch: Callable[[str, str], InvocationOutcome],
        poll: Callable[[dict[str, Any]], InvocationOutcome] | None = None,
        stop: Callable[[dict[str, Any]], None] | None = None,
        device_binding: dict[str, Any] | None = None,
    ) -> Any:
        """Validate, durably record intent, dispatch once, and await domain evidence.

        Policy and authorization are rechecked even for a cached request. A
        transport exception never retries dispatch. Cancellation can request a
        stop, but only a subsequent domain outcome proves that work ended.
        """
        if self.lease.released or resource_key not in self.lease.resources:
            raise PermissionError("device action has no matching Runtime admission")
        if self._uncertain:
            self.token.set()
            raise InvocationUnknown("unresolved device work has stopped this job")
        self._check_running()
        validate()
        # State queries and policy checks may block while the user cancels or
        # the owner closes. Never create a new dispatch intent after that stop.
        self._check_running()
        parameters_fingerprint = _fingerprint(parameters)
        binding = device_binding or {"resource_key": resource_key}
        fingerprint = _fingerprint([resource_key, binding, action_name, contract_version, parameters, self.principal])
        try:
            unresolved = self.runtime.store.unresolved_invocations(resource_key)
        except Exception as exc:
            self._uncertain.add("unreadable-invocations")
            self.token.set()
            self.runtime.storage_failed(exc)
            raise
        if unresolved:
            self._uncertain.update(record["invocation_id"] for record in unresolved)
            self.token.set()
            existing = self.runtime.store.find_invocation(resource_key, request_key, fingerprint)
            if existing is not None:
                try:
                    self.runtime.store.link_invocation(self.job_id, existing["invocation_id"])
                except Exception as exc:
                    self.token.set()
                    self.runtime.storage_failed(exc)
                    raise
                return self._result(existing)
            raise InvocationUnknown("an unresolved invocation blocks further device dispatch in this job")
        try:
            self.runtime.resources.bind_authority(self.lease, resource_key, self.runtime.store.path)
            record, created = self.runtime.store.begin_invocation(
                self.job_id,
                resource_key,
                request_key,
                fingerprint,
                {
                    "binding_id": self.binding_id,
                    "device_binding": binding,
                    "action_name": action_name,
                    "contract_version": contract_version,
                    "parameters_fingerprint": parameters_fingerprint,
                    "owner_pid": os.getpid(),
                    "principal": self.principal,
                    "lease_operation_id": self.lease.operation_id,
                    "scope": None,
                    "result": None,
                    "remote_handle": None,
                    "completion_evidence": {},
                },
            )
        except (RequestConflictError, ResourceBlockedError):
            raise
        except Exception as exc:
            self._uncertain.add("unrecorded-intent")
            self.token.set()
            self.runtime.storage_failed(exc)
            raise
        invocation_id = record["invocation_id"]
        if not created:
            if record["status"] not in {"succeeded", "failed"}:
                self._uncertain.add(invocation_id)
            return self._result(record)
        self._uncertain.add(invocation_id)
        try:
            self._check_running()
        except RunCancelled:
            # The intent exists, but this owner knows dispatch was never called.
            # Persist that fact before permitting cleanup; a failed commit keeps
            # the intent unresolved and therefore retains the resource block.
            self._write(
                invocation_id,
                {
                    "status": "failed",
                    "scope": "response",
                    "completion_evidence": {"dispatch_started": False, "cancelled_before_dispatch": True},
                },
            )
            self._uncertain.discard(invocation_id)
            raise
        try:
            outcome = dispatch(invocation_id, request_key)
        except Exception as exc:
            self._unknown(invocation_id, f"dispatch failed: {type(exc).__name__}")
            raise InvocationUnknown(f"dispatch outcome unknown: {invocation_id}") from exc
        stop_requested = False
        while True:
            if not isinstance(outcome, InvocationOutcome) or outcome.status not in {
                "succeeded",
                "failed",
                "pending",
                "unknown",
            }:
                self._unknown(invocation_id, "invalid domain outcome")
                raise InvocationUnknown("invalid domain outcome")
            if outcome.scope not in {"response", "physical"}:
                self._unknown(invocation_id, "invalid result scope")
                raise InvocationUnknown("invalid result scope")
            if outcome.status == "failed" or (outcome.status == "succeeded" and outcome.scope == "physical"):
                if not outcome.completion_evidence:
                    self._unknown(invocation_id, "terminal outcome lacks completion evidence")
                    raise InvocationUnknown("terminal outcome lacks completion evidence")
            if outcome.status == "unknown":
                self.token.set()
            facts = asdict(outcome)
            if facts["remote_handle"] is None:
                facts["remote_handle"] = record.get("remote_handle")
            record = self._write(invocation_id, facts)
            if outcome.status in {"succeeded", "failed"}:
                self._uncertain.discard(invocation_id)
                return self._result(record)
            if outcome.status == "unknown":
                raise InvocationUnknown(f"device action unresolved: {invocation_id}")
            if self.token.is_set() and not stop_requested:
                stop_requested = True
                if stop is not None:
                    self._write(invocation_id, {"stop_requested": True})
                    try:
                        stop(record)
                    except Exception as exc:
                        self._unknown(invocation_id, f"stop outcome unknown: {type(exc).__name__}")
                        raise InvocationUnknown("stop request did not prove physical stop") from exc
                elif poll is None or self.runtime.closing:
                    self._unknown(invocation_id, "owner cannot monitor pending work after cancellation")
                    raise InvocationUnknown("pending work has no confirmed stop")
            if poll is None:
                self._unknown(invocation_id, "pending work has no authoritative completion query")
                raise InvocationUnknown("pending work has no authoritative completion query")
            try:
                time.sleep(0.05)
                outcome = poll(record)
            except Exception as exc:
                self._unknown(invocation_id, f"completion query failed: {type(exc).__name__}")
                raise InvocationUnknown("completion query failed; dispatch was not retried") from exc
            if self.runtime.closing and isinstance(outcome, InvocationOutcome) and outcome.status == "pending":
                self._unknown(invocation_id, "execution host closed before terminal evidence")
                raise InvocationUnknown("execution host closed before terminal evidence")
