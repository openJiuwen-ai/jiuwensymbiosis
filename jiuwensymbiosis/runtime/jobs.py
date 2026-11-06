"""Application-owned task coordination, independent of any GUI framework."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jiuwensymbiosis.agent.cancel import CancelToken
from jiuwensymbiosis.agent.lifecycle import CleanupReport
from jiuwensymbiosis.runtime.artifacts import ArtifactStore
from jiuwensymbiosis.runtime.bindings import default_workspace
from jiuwensymbiosis.runtime.resources import ResourceBusyError, ResourceLease, ResourceManager
from jiuwensymbiosis.runtime.store import JobStore

if TYPE_CHECKING:
    from jiuwensymbiosis.runtime.bindings import BindingSnapshot


class RuntimeClosedError(RuntimeError):
    """The application has started closing and cannot accept new work."""


class Runtime:
    """One workspace's jobs, with shared local-machine resource admission.

    Returned snapshots and event batches are JSON values. A browser disconnect
    never owns or cancels a task. Applications call close() on host shutdown.
    """

    def __init__(
        self,
        workspace: str | Path | None = None,
        *,
        resource_directory: str | Path | None = None,
        cleanup_timeout: float = 2.0,
    ) -> None:
        self.workspace = Path(workspace or default_workspace()).expanduser().resolve()
        self.directory = self.workspace / "runtime"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.resources = ResourceManager(resource_directory)
        self.store = JobStore(self.directory / "jobs.sqlite3")
        self.artifacts = ArtifactStore(self.directory / "frames")
        self.cleanup_timeout = float(cleanup_timeout)
        self._bindings: dict[str, BindingSnapshot] = {}
        self._operations: dict[str, tuple[threading.Thread, CancelToken, ResourceLease]] = {}
        self._maintenance: dict[str, tuple[ResourceLease, Callable[[], Any] | None]] = {}
        self._lock = threading.RLock()
        self._closing = False
        self._storage_error: str | None = None
        self._recover_orphans()

    @property
    def closing(self) -> bool:
        """Whether the host has begun shutdown."""
        return self._closing

    @property
    def persistence_failed(self) -> bool:
        """Prevent hardware dispatch after an execution fact could not be saved."""
        return self._storage_error is not None

    def _recover_orphans(self):
        for invocation in self.store.list_invocations():
            if invocation["status"] not in {"intent", "pending"}:
                continue
            try:
                pid = int(invocation.get("owner_pid", 0))
                if pid <= 0:
                    raise ProcessLookupError("missing owner")
                os.kill(pid, 0)
            except ProcessLookupError:
                self.store.update_invocation(
                    invocation["invocation_id"],
                    {"status": "unknown", "uncertainty": "execution owner exited before terminal evidence"},
                )
                self.store.update(
                    invocation["job_id"],
                    {"phase": "blocked", "cleanup": {"released": False, "errors": ["unresolved device invocation"]}},
                    event_kind="blocked",
                    event_data={"invocation_id": invocation["invocation_id"]},
                )
        for job in self.store.unfinished():
            try:
                owner_pid = int(job.get("owner_pid", 0))
                if owner_pid <= 0:
                    raise ProcessLookupError("missing owner")
                os.kill(owner_pid, 0)
            except ProcessLookupError:
                self.store.update(
                    job["job_id"],
                    {
                        "phase": "blocked",
                        "cleanup": {
                            "released": False,
                            "errors": ["previous process exited without cleanup confirmation"],
                        },
                    },
                    event_kind="blocked",
                    event_data={"reason": "owner_exited"},
                )

    def prepare_binding(
        self,
        config_source: str | Path,
        *,
        config_snapshot: Mapping[str, Any] | None = None,
        workspace: str | Path | None = None,
    ) -> BindingSnapshot:
        from jiuwensymbiosis.runtime.bindings import prepare_binding

        chosen = Path(workspace).expanduser().resolve() if workspace is not None else self.workspace
        if chosen != self.workspace:
            raise ValueError("create a Runtime for the requested workspace")
        binding = prepare_binding(config_source, config_snapshot=config_snapshot, workspace=chosen)
        with self._lock:
            self._bindings[binding.binding_id] = binding
        return binding

    def get_binding(self, binding_id: str) -> BindingSnapshot:
        """Read a frozen binding for trusted tool discovery without connecting."""
        return self._bindings[binding_id]

    def submit_task(
        self,
        binding_id: str,
        request_id: str,
        query: str,
        *,
        agent_options: dict[str, Any] | None = None,
        principal: str = "local",
    ) -> dict[str, Any]:
        """Submit natural-language work through the selected domain's agent."""
        return self._submit(binding_id, request_id, query, agent_options, None, principal)

    def submit_actions(
        self,
        binding_id: str,
        request_id: str,
        steps: list[dict[str, Any]],
        *,
        principal: str = "local",
    ) -> dict[str, Any]:
        """Submit finite actions to the same worker, lease and JobStore as agents.

        The host owns the worker until completion; returning a job handle does
        not promise progress after that host exits. No LLM or MCP is required.
        """
        if not isinstance(steps, list) or not steps or any(not isinstance(step, dict) for step in steps):
            raise ValueError("steps must be a non-empty list of action objects")
        frozen_steps = json.loads(json.dumps(steps, allow_nan=False))
        return self._submit(binding_id, request_id, "execute admitted actions", None, frozen_steps, principal)

    def _submit(
        self,
        binding_id: str,
        request_id: str,
        query: str,
        agent_options: dict[str, Any] | None,
        steps: list[dict[str, Any]] | None,
        principal: str,
    ) -> dict[str, Any]:
        from jiuwensymbiosis.runtime.worker import JobRequest, build_agent_config, run_job

        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be non-empty")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be non-empty")
        if not isinstance(principal, str) or not principal.strip():
            raise ValueError("principal must be non-empty")
        # Strict JSON excludes callable rails, driver objects and non-finite data.
        options = json.loads(json.dumps(agent_options or {}, allow_nan=False))
        with self._lock, (self.directory / "submit.lock").open("a+") as submission:
            fcntl.flock(submission, fcntl.LOCK_EX)
            self._prune_finished_operations()
            binding = self._bindings[binding_id]
            if steps is None:
                build_agent_config(binding, options)  # validate before admission/acceptance
            fingerprint = hashlib.sha256(
                json.dumps(
                    [binding.fingerprint, query, options, steps, principal], sort_keys=True, allow_nan=False
                ).encode()
            ).hexdigest()
            existing = self.store.find_request(request_id, fingerprint)
            if existing is not None:
                return self.get_job(existing["job_id"])
            if self._closing:
                raise RuntimeClosedError("runtime is closing")
            if self._storage_error:
                raise RuntimeError(f"runtime persistence failed: {self._storage_error}")
            lease = self.resources.acquire(binding.resources)
            try:
                for resource in binding.resources:
                    self.resources.bind_authority(lease, resource, self.store.path)
            except BaseException:
                # No session exists and no command has been issued yet.
                self.resources.finish(lease, CleanupReport())
                raise
            job_id = uuid.uuid4().hex
            token = CancelToken()
            initial = {
                "job_id": job_id,
                "binding_id": binding_id,
                "principal": principal,
                "phase": "reserved",
                "result": None,
                "cancel_requested": False,
                "owner_pid": os.getpid(),
                "operation_id": lease.operation_id,
                "cleanup": {"released": False},
                "workspace": str(self.workspace),
            }
            try:
                snapshot, created = self.store.create(request_id, fingerprint, initial)
            except BaseException:
                self.resources.finish(lease, CleanupReport())
                raise
            if not created:
                self.resources.finish(lease, CleanupReport())
                return snapshot
            thread = threading.Thread(
                target=run_job,
                args=(JobRequest(self, job_id, binding, query, options, lease, token, request_id, steps, principal),),
                name=f"jiuwen-task-{job_id[:8]}",
                daemon=True,
            )
            self._operations[job_id] = (thread, token, lease)
            try:
                thread.start()
            except BaseException as exc:
                self._operations.pop(job_id)
                self.resources.finish(lease, CleanupReport())
                self.store.update(
                    job_id,
                    {"phase": "failed", "error": str(exc), "cleanup": {"released": True}},
                    event_kind="failed",
                    event_data={"reason": "thread_start_failed"},
                )
                raise
            return snapshot

    def get_job(self, job_id: str) -> dict[str, Any]:
        snapshot = self.store.get(job_id)
        cleanup = snapshot.get("cleanup", {})
        records = self.resources.records()
        owned = [record for record in records if record.get("operation_id") == snapshot.get("operation_id")]
        if cleanup.get("release_ready"):
            # SQLite and file leases cannot commit atomically. Persist release
            # evidence first; report actual release from the authoritative lease
            # records rather than a final post-release SQLite write.
            held = bool(owned) or any(record.get("state") == "invalid" for record in records)
            snapshot["cleanup"] = {**cleanup, "released": not held}
        control = self._operations.get(job_id)
        persistence_holds_lease = self.persistence_failed and control is not None and not control[2].released
        if any(record.get("state") == "blocked" for record in owned) or persistence_holds_lease:
            snapshot["phase"] = "blocked"
            snapshot["cleanup"] = {**snapshot.get("cleanup", {}), "released": False}
        return snapshot

    def get_invocation(self, invocation_id: str) -> dict[str, Any]:
        return self.store.get_invocation(invocation_id)

    def list_invocations(self, job_id: str) -> list[dict[str, Any]]:
        self.store.get(job_id)
        return self.store.list_invocations(job_id)

    def reconcile_invocation(
        self,
        invocation_id: str,
        *,
        status: str,
        completion_evidence: dict[str, Any],
        cleanup_report: CleanupReport,
        result: Any = None,
    ) -> dict[str, Any]:
        """Trusted operator reconciliation; never an exported/model-callable tool.

        External physical and connection cleanup evidence is required. Resolving
        one uncertain action does not claim that its abandoned user plan ran to
        completion. The plan remains incomplete and is never replayed here.
        """
        if not isinstance(cleanup_report, CleanupReport) or not cleanup_report.released:
            raise ValueError("confirmed device and connection cleanup is required")
        if not isinstance(completion_evidence, dict) or not completion_evidence:
            raise ValueError("completion evidence is required")
        with self._lock:
            record = self.store.get_invocation(invocation_id)
            if record["status"] != "unknown":
                raise ValueError("only an unknown invocation can be reconciled")
            job_id = record["job_id"]
            control = self._operations.get(job_id)
            if control is not None and control[0].is_alive():
                raise ResourceBusyError("execution owner is still monitoring this invocation")
            if control is None:
                # A second Runtime in the same live process cannot bypass the
                # actual owner's controls merely by opening its database.
                owner_pid = int(record.get("owner_pid", 0))
                try:
                    if owner_pid <= 0:
                        raise ProcessLookupError("missing owner")
                    os.kill(owner_pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise ResourceBusyError("reconcile through the live execution owner")
            updated = self.store.update_invocation(
                invocation_id,
                {"status": status, "completion_evidence": completion_evidence, "result": result},
                reconciled=True,
            )
            unresolved = [
                item for item in self.store.list_invocations(job_id) if item["status"] not in {"succeeded", "failed"}
            ]
            if not unresolved:
                self.store.update(
                    job_id,
                    {"phase": "incomplete", "reconciled": True},
                    event_kind="reconciled",
                    event_data={"invocation_id": invocation_id},
                )
                if control is not None:
                    self.record_cleanup(job_id, control[2], cleanup_report)
                else:
                    cleanup = {**asdict(cleanup_report), "released": False, "release_ready": True}
                    self.store.update(job_id, {"cleanup": cleanup}, event_kind="cleanup", event_data=cleanup)
                    self.resources.reconcile(record["lease_operation_id"], cleanup_report)
            return updated

    def _prune_finished_operations(self) -> None:
        """Drop controls after both thread exit and confirmed resource release.

        Sweep on host activity instead of adding a reaper thread. Durable job
        history stays in JobStore; blocked or still-finishing owners stay here.
        """
        with self._lock:
            for job_id, (thread, _token, lease) in tuple(self._operations.items()):
                if not thread.is_alive() and lease.released:
                    self._operations.pop(job_id)

    def wait_for_job(self, job_id: str, timeout: float | None = None) -> dict[str, Any]:
        """Trusted host/testing wait; timeout never implies resource release."""
        with self._lock:
            operation = self._operations.get(job_id)
        if operation is not None:
            operation[0].join(timeout)
        self._prune_finished_operations()
        return self.get_job(job_id)

    def read_events(self, job_id: str, after_seq: int = 0, limit: int = 100) -> dict[str, Any]:
        return self.store.read_events(job_id, after_seq, limit)

    def latest_frame(self, job_id: str) -> dict[str, Any] | None:
        self.store.get(job_id)
        return self.artifacts.latest(job_id)

    def read_artifact(self, reference: dict[str, Any]) -> bytes:
        self.store.get(reference.get("job_id", ""))
        return self.artifacts.read(reference)

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self.store.get(job_id)
            operation = self._operations.get(job_id)
            if operation is None or not operation[0].is_alive():
                return {"accepted": False, "phase": job["phase"]}
            # Set the control signal first: storage trouble cannot suppress stop.
            operation[1].set()
            if not job.get("cancel_requested"):
                self.store.update(job_id, {"cancel_requested": True}, event_kind="cancel_requested", event_data={})
            return {"accepted": True, "phase": job["phase"]}

    def get_runtime_state(self) -> dict[str, Any]:
        self._prune_finished_operations()
        records = self.resources.records()
        for record in records:
            if record.get("state") == "reserved":
                try:
                    pid = int(record.get("pid", 0))
                    if pid <= 0:
                        raise ProcessLookupError("missing owner")
                    os.kill(pid, 0)
                except ProcessLookupError:
                    record["state"] = "blocked"
                    record["reason"] = "owner exited without cleanup confirmation"
        return {
            "closing": self._closing,
            "busy": bool(records),
            "blocked": bool(self._storage_error)
            or any(item.get("state") in {"blocked", "invalid"} for item in records),
            "operations": records,
            "storage_error": self._storage_error,
        }

    def storage_failed(self, error: BaseException) -> None:
        """Stop active work and reject new submissions after fact persistence fails."""
        with self._lock:
            self._storage_error = f"{type(error).__name__}: {error}"
            tokens = [token for _thread, token, _lease in self._operations.values()]
        for token in tokens:
            token.set()

    def acquire_maintenance(
        self,
        binding: BindingSnapshot,
        operation: str,
        *,
        stop: Callable[[], Any] | None = None,
    ) -> ResourceLease:
        """Trusted Python workflow admission; never expose callbacks to a browser."""
        with self._lock:
            if self._closing:
                raise RuntimeClosedError("runtime is closing")
            lease = self.resources.acquire(binding.resources, operation=operation)
            self._maintenance[lease.operation_id] = (lease, stop)
            return lease

    def finish_maintenance(self, lease: ResourceLease, report: CleanupReport) -> bool:
        with self._lock:
            released = self.resources.finish(lease, report)
            if released:
                self._maintenance.pop(lease.operation_id, None)
            return released

    def close(self, timeout: float = 2.0) -> dict[str, Any]:
        with self._lock:
            self._closing = True
            operations = list(self._operations.values())
            maintenance = list(self._maintenance.values())
        for _thread, token, _lease in operations:
            token.set()
        errors = []
        for _lease, stop in maintenance:
            if stop is not None:
                try:
                    stop()
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
        import time

        deadline = time.monotonic() + max(0, timeout)
        for thread, _token, _lease in operations:
            if thread.ident is not None:
                thread.join(max(0, deadline - time.monotonic()))
        state = self.get_runtime_state()
        state["stop_errors"] = errors
        state["closed"] = (
            not errors
            and all(lease.released for _, _, lease in operations)
            and all(lease.released for lease, _ in maintenance)
            and not any(thread.is_alive() for thread, _, _ in operations)
        )
        # Keep the read side available for final snapshots / repeated close().
        # SQLite owns no hardware; callers may dispose the Runtime after closure.
        return state

    def record_cleanup(self, job_id: str, lease: ResourceLease, report: CleanupReport) -> bool:
        """Commit cleanup evidence before allowing another hardware owner.

        A crash between this transaction and lease release retains the durable
        lease record. No failed SQLite write can accidentally release hardware.
        """
        if self.persistence_failed:
            report = CleanupReport(
                pending_work=report.pending_work,
                connected=report.connected,
                errors=(*report.errors, "Runtime persistence failed"),
            )
        cleanup = {**asdict(report), "released": False, "release_ready": report.released}
        self.store.update(job_id, {"cleanup": cleanup}, event_kind="cleanup", event_data=cleanup)
        return self.resources.finish(lease, report)
