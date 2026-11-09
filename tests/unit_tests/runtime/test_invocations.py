"""Action facts use the real Runtime store and admission, without device hardware."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from jiuwensymbiosis.agent.cancel import CancelToken, RunCancelled
from jiuwensymbiosis.agent.lifecycle import CleanupReport
from jiuwensymbiosis.runtime import RequestConflictError, ResourceBlockedError, Runtime
from jiuwensymbiosis.runtime.invocations import InvocationContext, InvocationOutcome, InvocationUnknown
from jiuwensymbiosis.runtime.store import JobStore
from jiuwensymbiosis.tools.bindings import ActionBinding


@pytest.fixture
def admitted(tmp_path):
    runtime = Runtime(tmp_path / "workspace", resource_directory=tmp_path / "locks")
    lease = runtime.resources.acquire(("physical:synthetic-temperature",))
    runtime.store.create(
        "job-request", "job-fingerprint", {"job_id": "job", "phase": "running", "owner_pid": os.getpid()}
    )
    context = InvocationContext(runtime, "job", "binding", lease, CancelToken())
    yield runtime, context
    # Trusted test cleanup after assertions, never an execution-path fallback.
    runtime.resources.finish(lease, CleanupReport())
    runtime.store.close()


def invoke(context, dispatch, **overrides):
    args = {
        "resource_key": "physical:synthetic-temperature",
        "action_name": "read_temperature",
        "contract_version": "1",
        "parameters": {},
        "request_key": "stable-action-request",
        "validate": lambda: None,
        "dispatch": dispatch,
    }
    args.update(overrides)
    return context.invoke(**args)


def test_intent_is_committed_before_dispatch_and_numeric_result_is_preserved(admitted):
    runtime, context = admitted
    calls = []

    def dispatch(invocation_id, request_key):
        reader = JobStore(runtime.store.path)
        try:
            saved = reader.get_invocation(invocation_id)
            assert saved["status"] == "intent"
            assert saved["request_key"] == request_key
            assert saved["parameters_fingerprint"]
            assert saved["binding_id"] == "binding"
            assert reader.read_events("job")["events"][-1]["kind"] == "invocation_intent"
        finally:
            reader.close()
        calls.append(invocation_id)
        return InvocationOutcome("succeeded", {"temperature_c": 20}, "response")

    assert invoke(context, dispatch) == {"temperature_c": 20}
    assert invoke(context, dispatch) == {"temperature_c": 20}
    assert len(calls) == 1
    assert not context.pending_work
    record = runtime.get_invocation(calls[0])
    assert record["scope"] == "response"
    assert record["status"] == "succeeded"
    with pytest.raises(RequestConflictError):
        invoke(context, dispatch, parameters={"temperature_c": 21})


def test_action_dedupe_is_independent_of_job_id(admitted):
    runtime, context = admitted
    calls = []

    def dispatch(*args):
        calls.append(args)
        return InvocationOutcome("succeeded", {"temperature_c": 20})

    invoke(context, dispatch)
    runtime.store.create("second-job", "other", {"job_id": "second", "phase": "running"})
    second = InvocationContext(runtime, "second", "same-device-new-binding-id", context.lease, CancelToken())
    assert invoke(second, dispatch) == {"temperature_c": 20}
    assert len(calls) == 1


def test_policy_is_rechecked_on_retries_before_dispatch_or_replay(admitted):
    runtime, context = admitted
    invoke(context, lambda *a: InvocationOutcome("succeeded", {}))

    def denied():
        raise PermissionError("permission revoked")

    with pytest.raises(PermissionError):
        invoke(context, lambda *a: pytest.fail("dispatch"), validate=denied)
    assert len(runtime.list_invocations("job")) == 1


def test_transport_error_records_unknown_and_never_resends(admitted):
    runtime, context = admitted
    calls = []

    def lost_response(*args):
        calls.append(args)
        raise TimeoutError("response lost after device accepted command")

    with pytest.raises(InvocationUnknown):
        invoke(context, lost_response)
    with pytest.raises(InvocationUnknown):
        invoke(context, lost_response)
    assert len(calls) == 1
    record = runtime.list_invocations("job")[0]
    assert record["status"] == "unknown"
    assert context.token.is_set()
    assert context.pending_work == (record["invocation_id"],)
    assert not runtime.resources.finish(context.lease, CleanupReport(pending_work=context.pending_work))
    assert runtime.get_runtime_state()["blocked"]


def test_intent_write_failure_prevents_dispatch_and_blocks_release(admitted, monkeypatch):
    runtime, context = admitted

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(runtime.store, "begin_invocation", fail)
    with pytest.raises(OSError):
        invoke(context, lambda *a: pytest.fail("write without durable intent"))
    assert context.pending_work
    assert runtime.persistence_failed
    assert context.token.is_set()
    assert not runtime.record_cleanup("job", context.lease, CleanupReport())


def test_completion_write_failure_keeps_uncertainty(admitted, monkeypatch):
    runtime, context = admitted
    calls = []

    def fail(*args, **kwargs):
        raise OSError("completion commit failed")

    monkeypatch.setattr(runtime.store, "update_invocation", fail)
    with pytest.raises(OSError):
        invoke(context, lambda *a: (calls.append(a), InvocationOutcome("succeeded", {}))[1])
    assert len(calls) == 1
    assert runtime.list_invocations("job")[0]["status"] == "intent"
    assert runtime.persistence_failed and context.pending_work
    assert context.token.is_set()


def test_cancel_requests_stop_once_and_waits_for_authoritative_completion(admitted):
    runtime, context = admitted
    stops, polls = [], []

    def dispatch(*args):
        context.token.set()
        return InvocationOutcome("pending", {"accepted": True}, "physical", remote_handle={"task_id": "remote-123"})

    def poll(record):
        assert runtime.get_invocation(record["invocation_id"])["remote_handle"] == {"task_id": "remote-123"}
        polls.append(record["invocation_id"])
        if len(polls) == 1:
            return InvocationOutcome("pending", scope="physical", remote_handle=record["remote_handle"])
        return InvocationOutcome("succeeded", {"stopped": True}, "physical", {"owner": "synthetic", "idle": True})

    assert invoke(context, dispatch, poll=poll, stop=lambda record: stops.append(record)) == {"stopped": True}
    assert len(stops) == 1 and len(polls) == 2
    assert not context.pending_work


def test_host_shutdown_records_unknown_instead_of_claiming_durable_work_continues(admitted):
    runtime, context = admitted

    def dispatch(*args):
        runtime.close(timeout=0)
        context.token.set()
        return InvocationOutcome("pending", remote_handle="owner-task")

    with pytest.raises(InvocationUnknown):
        invoke(context, dispatch, poll=lambda record: InvocationOutcome("pending"))
    assert runtime.list_invocations("job")[0]["status"] == "unknown"
    assert context.pending_work


def test_physical_completion_requires_evidence(admitted):
    runtime, context = admitted
    with pytest.raises(InvocationUnknown, match="evidence"):
        invoke(context, lambda *a: InvocationOutcome("succeeded", {}, "physical"))
    assert runtime.list_invocations("job")[0]["status"] == "unknown"


def test_crashed_process_retains_lease_and_intent_becomes_unknown(tmp_path):
    workspace, locks = tmp_path / "workspace", tmp_path / "locks"
    code = textwrap.dedent("""
        import os, sys
        from jiuwensymbiosis.runtime import Runtime
        from jiuwensymbiosis.runtime.invocations import InvocationContext
        from jiuwensymbiosis.agent.cancel import CancelToken
        runtime = Runtime(sys.argv[1], resource_directory=sys.argv[2])
        lease = runtime.resources.acquire(("physical:crash-device",))
        runtime.store.create("crash", "fp", {"job_id": "crash-job", "phase": "running", "owner_pid": os.getpid()})
        ctx = InvocationContext(runtime, "crash-job", "binding", lease, CancelToken())
        ctx.invoke(resource_key="physical:crash-device", action_name="write", contract_version="1", parameters={},
                   request_key="once", validate=lambda: None, dispatch=lambda *args: os._exit(9))
    """)
    completed = subprocess.run([sys.executable, "-c", code, str(workspace), str(locks)], check=False)
    assert completed.returncode == 9
    reopened = Runtime(workspace, resource_directory=locks)
    try:
        assert reopened.get_job("crash-job")["phase"] == "blocked"
        assert reopened.list_invocations("crash-job")[0]["status"] == "unknown"
        with pytest.raises(ResourceBlockedError):
            reopened.resources.acquire(("physical:crash-device",))
        assert reopened.get_runtime_state()["blocked"]
    finally:
        reopened.store.close()


@dataclass
class RuntimeBinding:
    workspace: Path
    session: object
    binding_id: str = "synthetic"
    fingerprint: str = "synthetic"
    adapter: str = "instrument"
    resources: tuple[str, ...] = ("physical:synthetic-temperature",)

    def config_data(self):
        return {}

    def build_session(self):
        return self.session


class SyntheticSession:
    capabilities = frozenset({"instrument.measurement"})

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0
        self.connected = False
        self.validate = lambda: None

    def bind_execution(self, context):
        self.context = context

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def cleanup_report(self):
        return CleanupReport(connected=self.connected)

    def action_bindings(self):
        contract = SimpleNamespace(
            name="read_temperature",
            description="synthetic measurement",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {"temperature_c": {"type": "number"}}},
            required_capabilities=frozenset({"instrument.measurement"}),
            planner_visible=True,
        )

        def call(params, request_key, principal):
            def dispatch(*args):
                self.calls += 1
                if self.fail:
                    raise TimeoutError("unknown physical outcome")
                return InvocationOutcome("succeeded", {"temperature_c": 20})

            return invoke(self.context, dispatch, request_key=request_key, parameters=params, validate=self.validate)

        return (
            ActionBinding(
                contract,
                "synthetic-temperature",
                ("physical:synthetic-temperature",),
                "instrument",
                call,
                lambda principal: principal == "local",
            ),
        )


@pytest.mark.parametrize("stop", ["cancel", "close"])
@pytest.mark.parametrize("checkpoint", ["validation", "persisted_intent"])
def test_stop_during_pre_dispatch_work_never_sends_command(tmp_path, monkeypatch, stop, checkpoint):
    runtime = Runtime(tmp_path / "workspace", resource_directory=tmp_path / "locks")
    session = SyntheticSession()
    binding = RuntimeBinding(runtime.workspace, session)
    runtime._bindings[binding.binding_id] = binding
    entered, resume = threading.Event(), threading.Event()

    def pause():
        entered.set()
        if not resume.wait(3):
            raise TimeoutError("test did not release pre-dispatch work")

    if checkpoint == "validation":
        session.validate = pause
    else:
        begin = runtime.store.begin_invocation

        def persist_then_pause(*args, **kwargs):
            record = begin(*args, **kwargs)
            pause()
            return record

        monkeypatch.setattr(runtime.store, "begin_invocation", persist_then_pause)
    try:
        job = runtime.submit_actions(binding.binding_id, "stop-before-dispatch", [{"op": "read_temperature"}])
        assert entered.wait(3)
        if stop == "cancel":
            assert runtime.cancel(job["job_id"])["accepted"]
        else:
            assert runtime.close(timeout=0)["closing"]
        resume.set()
        final = runtime.wait_for_job(job["job_id"], 3)
        assert session.calls == 0
        assert final["phase"] == "cancelled"
        assert final["cleanup"]["released"] is True
        records = runtime.list_invocations(job["job_id"])
        if checkpoint == "validation":
            assert records == []
        else:
            assert len(records) == 1 and records[0]["status"] == "failed"
            assert records[0]["completion_evidence"] == {"dispatch_started": False, "cancelled_before_dispatch": True}
    finally:
        resume.set()
        runtime.close()
        runtime.store.close()


def test_cancelled_intent_write_failure_retains_blocking_without_dispatch(admitted, monkeypatch):
    runtime, context = admitted
    begin = runtime.store.begin_invocation

    def persist_then_cancel(*args, **kwargs):
        record = begin(*args, **kwargs)
        context.token.set()
        return record

    def fail(*args, **kwargs):
        raise OSError("cannot save that dispatch never started")

    monkeypatch.setattr(runtime.store, "begin_invocation", persist_then_cancel)
    monkeypatch.setattr(runtime.store, "update_invocation", fail)
    with pytest.raises(OSError):
        invoke(context, lambda *a: pytest.fail("cancelled intent must not dispatch"))
    assert runtime.list_invocations("job")[0]["status"] == "intent"
    assert context.pending_work and context.token.is_set()
    assert runtime.persistence_failed
    assert not runtime.record_cleanup("job", context.lease, CleanupReport())


@pytest.mark.parametrize("checkpoint", ["validation", "persisted_intent"])
def test_storage_failure_stops_an_active_job_before_dispatch(tmp_path, monkeypatch, checkpoint):
    runtime = Runtime(tmp_path / "workspace", resource_directory=tmp_path / "locks")
    session = SyntheticSession()
    binding = RuntimeBinding(runtime.workspace, session)
    runtime._bindings[binding.binding_id] = binding

    def lose_persistence():
        runtime.storage_failed(OSError("execution evidence storage unavailable"))
        assert session.context.token.is_set()

    if checkpoint == "validation":
        session.validate = lose_persistence
    else:
        begin = runtime.store.begin_invocation

        def persist_then_fail(*args, **kwargs):
            record = begin(*args, **kwargs)
            lose_persistence()
            return record

        monkeypatch.setattr(runtime.store, "begin_invocation", persist_then_fail)
    try:
        job = runtime.submit_actions(binding.binding_id, "storage-stop", [{"op": "read_temperature"}])
        final = runtime.wait_for_job(job["job_id"], 3)
        assert session.calls == 0
        assert final["phase"] == "blocked"
        assert final["cleanup"]["released"] is False
        assert runtime.resources.records()
        records = runtime.list_invocations(job["job_id"])
        assert [record["status"] for record in records] == ([] if checkpoint == "validation" else ["intent"])
    finally:
        for _, _, lease in tuple(runtime._operations.values()):
            runtime.resources.finish(lease, CleanupReport())
        runtime.close()
        runtime.store.close()


def test_unreadable_action_history_stops_the_shared_task(admitted, monkeypatch):
    runtime, context = admitted

    def fail(*args):
        raise OSError("cannot determine pending actions")

    monkeypatch.setattr(runtime.store, "unresolved_invocations", fail)
    with pytest.raises(OSError):
        invoke(context, lambda *a: pytest.fail("dispatch without readable action history"))
    assert context.token.is_set() and context.pending_work
    assert runtime.persistence_failed
    assert runtime.list_invocations("job") == []


def test_unknown_domain_result_stops_all_work_using_the_shared_token(admitted):
    runtime, context = admitted
    with pytest.raises(InvocationUnknown):
        invoke(context, lambda *a: InvocationOutcome("unknown", {"received": "ambiguous"}))
    assert runtime.list_invocations("job")[0]["result"] == {"received": "ambiguous"}
    with pytest.raises(RunCancelled):
        context.token.raise_if_set()
    assert context.pending_work


@pytest.mark.parametrize("fail", [False, True])
def test_submit_actions_uses_existing_worker_and_preserves_unknown_blocking(tmp_path, fail):
    runtime = Runtime(tmp_path / "workspace", resource_directory=tmp_path / "locks")
    session = SyntheticSession(fail)
    binding = RuntimeBinding(runtime.workspace, session)
    runtime._bindings[binding.binding_id] = binding
    try:
        job = runtime.submit_actions(binding.binding_id, "direct-action", [{"op": "read_temperature", "params": {}}])
        final = runtime.wait_for_job(job["job_id"], 3)
        assert session.calls == 1
        assert final["phase"] == ("blocked" if fail else "succeeded")
        assert final["cleanup"]["released"] is not fail
        assert runtime.list_invocations(job["job_id"])[0]["status"] == ("unknown" if fail else "succeeded")
        assert not hasattr(session, "env") and not hasattr(session, "api")
        assert (
            runtime.submit_actions(binding.binding_id, "direct-action", [{"op": "read_temperature", "params": {}}])[
                "job_id"
            ]
            == job["job_id"]
        )
        assert session.calls == 1
    finally:
        for _, _, lease in tuple(runtime._operations.values()):
            runtime.resources.finish(lease, CleanupReport())
        runtime.close()
        runtime.store.close()


def test_unknown_blocks_new_keys_and_other_jobs_even_after_external_lease_cleanup(admitted):
    runtime, context = admitted
    with pytest.raises(InvocationUnknown):
        invoke(context, lambda *a: InvocationOutcome("unknown"))
    with pytest.raises(InvocationUnknown):
        invoke(context, lambda *a: pytest.fail("new command after unknown"), request_key="new-key")
    # Even a trusted file-lock cleanup alone cannot erase an unresolved action.
    runtime.resources.finish(context.lease, CleanupReport())
    lease = runtime.resources.acquire(context.lease.resources)
    runtime.store.create("other-job", "other", {"job_id": "other", "phase": "running"})
    second = InvocationContext(runtime, "other", "binding", lease, CancelToken())
    try:
        with pytest.raises(InvocationUnknown):
            invoke(second, lambda *a: pytest.fail("dispatch after only lock reconciliation"), request_key="different")
        assert second.pending_work and second.token.is_set()
    finally:
        runtime.resources.finish(lease, CleanupReport())


def test_another_workspace_cannot_silently_create_a_second_physical_history(tmp_path):
    locks = tmp_path / "locks"
    first = Runtime(tmp_path / "first", resource_directory=locks)
    second = Runtime(tmp_path / "second", resource_directory=locks)
    first_session, second_session = SyntheticSession(), SyntheticSession()
    first_binding = RuntimeBinding(first.workspace, first_session)
    second_binding = RuntimeBinding(second.workspace, second_session)
    first._bindings[first_binding.binding_id] = first_binding
    second._bindings[second_binding.binding_id] = second_binding
    try:
        job = first.submit_actions(first_binding.binding_id, "first", [{"op": "read_temperature", "params": {}}])
        assert first.wait_for_job(job["job_id"], 3)["phase"] == "succeeded"
        with pytest.raises(ResourceBlockedError, match="another Runtime JobStore"):
            second.submit_actions(second_binding.binding_id, "second", [{"op": "read_temperature", "params": {}}])
        assert second_session.calls == 0 and not second_session.connected
        assert not second.resources.records()
    finally:
        first.close()
        second.close()
        first.store.close()
        second.store.close()


def test_cleanup_fact_write_failure_retains_actual_resource_lease(admitted, monkeypatch):
    runtime, context = admitted

    def fail(*args, **kwargs):
        raise OSError("cleanup evidence failed to commit")

    monkeypatch.setattr(runtime.store, "update", fail)
    with pytest.raises(OSError):
        runtime.record_cleanup("job", context.lease, CleanupReport())
    assert not context.lease.released
    assert runtime.resources.records()


def test_trusted_reconciliation_resolves_device_but_does_not_complete_abandoned_plan(tmp_path):
    runtime = Runtime(tmp_path / "workspace", resource_directory=tmp_path / "locks")
    session = SyntheticSession(fail=True)
    binding = RuntimeBinding(runtime.workspace, session)
    runtime._bindings[binding.binding_id] = binding
    try:
        job = runtime.submit_actions(binding.binding_id, "uncertain", [{"op": "read_temperature", "params": {}}])
        assert runtime.wait_for_job(job["job_id"], 3)["phase"] == "blocked"
        invocation = runtime.list_invocations(job["job_id"])[0]
        with pytest.raises(ValueError, match="cleanup"):
            runtime.reconcile_invocation(
                invocation["invocation_id"],
                status="failed",
                completion_evidence={"observer": "test"},
                cleanup_report=CleanupReport(connected=True),
            )
        resolved = runtime.reconcile_invocation(
            invocation["invocation_id"],
            status="failed",
            completion_evidence={"observer": "test operator", "physical_idle": True, "connection_closed": True},
            cleanup_report=CleanupReport(),
        )
        assert resolved["status"] == "failed"
        final = runtime.get_job(job["job_id"])
        assert final["phase"] == "incomplete" and final["cleanup"]["released"]
        assert session.calls == 1
    finally:
        for _, _, lease in tuple(runtime._operations.values()):
            runtime.resources.finish(lease, CleanupReport())
        runtime.close()
        runtime.store.close()


def test_worker_cleanup_write_failure_cannot_report_a_successful_job(tmp_path, monkeypatch):
    runtime = Runtime(tmp_path / "workspace", resource_directory=tmp_path / "locks")
    binding = RuntimeBinding(runtime.workspace, SyntheticSession())
    runtime._bindings[binding.binding_id] = binding
    update = runtime.store.update

    def failed_cleanup(job_id, changes, **kwargs):
        if kwargs.get("event_kind") == "cleanup":
            raise OSError("cleanup evidence commit failed")
        return update(job_id, changes, **kwargs)

    monkeypatch.setattr(runtime.store, "update", failed_cleanup)
    try:
        job = runtime.submit_actions(binding.binding_id, "failure", [{"op": "read_temperature", "params": {}}])
        final = runtime.wait_for_job(job["job_id"], 3)
        assert final["phase"] == "blocked"
        assert final["cleanup"]["released"] is False
        assert runtime.resources.records()
    finally:
        for _, _, lease in tuple(runtime._operations.values()):
            runtime.resources.finish(lease, CleanupReport())
        runtime.close()
        runtime.store.close()
