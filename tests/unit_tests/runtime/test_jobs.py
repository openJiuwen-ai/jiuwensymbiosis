"""Runtime uses its real store/admission with fake sessions and task execution."""

import gc
import logging
import weakref
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jiuwensymbiosis.agent.cancel import cancellable_call
from jiuwensymbiosis.agent.lifecycle import CleanupReport, HardwareCleanupError
from jiuwensymbiosis.runtime import RequestConflictError, ResourceBusyError, Runtime, RuntimeClosedError, worker


@dataclass
class Binding:
    workspace: Path
    session: object
    binding_id: str = "frozen-binding"
    fingerprint: str = "frozen-fingerprint"
    adapter: str = "stub"
    resources: tuple = ("device:test",)

    def config_data(self):
        return {"model": {"api_key": "sensitive-key"}, "agent": {"exec_mode": "stepagent"}}

    def build_session(self):
        return self.session


@pytest.fixture
def runtime(tmp_path):
    runtime = Runtime(tmp_path / "workspace", resource_directory=tmp_path / "locks", cleanup_timeout=0.01)
    session = Mock()
    session.env.get_observation.return_value = SimpleNamespace(joints=None, rgb=None)
    session.cleanup_report.side_effect = lambda: CleanupReport(pending_work=session.cancel_token.pending_work)
    binding = Binding(runtime.workspace, session)
    runtime._bindings[binding.binding_id] = binding
    yield runtime, binding
    runtime.close(timeout=3)
    runtime.store.close()


def wait_job(runtime, job):
    final = runtime.wait_for_job(job["job_id"], 3)
    assert final["phase"] not in {"reserved", "connecting", "running"}
    return final


def test_completed_task_controls_are_collectable_but_history_survives(runtime, monkeypatch):
    runtime, binding = runtime
    tokens = []

    def run(_session, *_args, cancel_token, **_kwargs):
        tokens.append(weakref.ref(cancel_token))
        return {"result_type": "success", "output": "done"}

    monkeypatch.setattr(worker, "run_robot_task", run)
    jobs = []
    for index in range(3):
        job = runtime.submit_task(binding.binding_id, f"request-{index}", "hello")
        jobs.append(job)
        assert runtime.wait_for_job(job["job_id"], 3)["cleanup"]["released"]
    # The fixture's reused fake session owns the latest token independently.
    binding.session.cancel_token = None
    runtime.get_runtime_state()
    gc.collect()
    assert all(token() is None for token in tokens)
    for index, job in enumerate(jobs):
        assert runtime.get_job(job["job_id"])["phase"] == "succeeded"
        assert runtime.read_events(job["job_id"])["events"]
        assert runtime.submit_task(binding.binding_id, f"request-{index}", "hello")["job_id"] == job["job_id"]


def test_headless_submit_deduplicates_and_events_have_separate_cursors(runtime, monkeypatch):
    runtime, binding = runtime
    calls = Mock(return_value={"result_type": "success", "output": "done sensitive-key"})
    monkeypatch.setattr(worker, "run_robot_task", calls)
    job = runtime.submit_task(binding.binding_id, "request1", "hello")
    final = wait_job(runtime, job)
    assert final["phase"] == "succeeded"
    assert final["cleanup"]["released"]
    assert runtime.submit_task(binding.binding_id, "request1", "hello")["job_id"] == job["job_id"]
    calls.assert_called_once()
    with pytest.raises(RequestConflictError):
        runtime.submit_task(binding.binding_id, "request1", "different")
    first = runtime.read_events(job["job_id"], limit=2)
    assert first == runtime.read_events(job["job_id"], limit=2)
    assert "sensitive-key" not in str(runtime.get_job(job["job_id"]))
    assert runtime.read_events(job["job_id"], first["next_seq"])["events"]


def test_busy_retry_is_deduplicated_before_new_admission(runtime, monkeypatch):
    runtime, binding = runtime
    entered, release = Event(), Event()

    def run(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return {"output": "done"}

    monkeypatch.setattr(worker, "run_robot_task", run)
    job = runtime.submit_task(binding.binding_id, "request1", "hello")
    try:
        assert entered.wait(3)
        assert runtime.submit_task(binding.binding_id, "request1", "hello")["job_id"] == job["job_id"]
        with pytest.raises(ResourceBusyError):
            runtime.submit_task(binding.binding_id, "request2", "hello")
    finally:
        release.set()
        wait_job(runtime, job)


def test_cancelled_helper_holds_resources_until_actual_completion(runtime, monkeypatch):
    runtime, binding = runtime
    entered, release = Event(), Event()

    def slow_operation():
        entered.set()
        release.wait(3)

    def run(session, *args, cancel_token, **kwargs):
        return cancellable_call(slow_operation, cancel_token)

    monkeypatch.setattr(worker, "run_robot_task", run)
    job = runtime.submit_task(binding.binding_id, "request1", "hello")
    try:
        assert entered.wait(3)
        assert runtime.cancel(job["job_id"])["accepted"]
        result = runtime.close(timeout=0.2)
        assert not result["closed"]
        assert result["busy"]
    finally:
        release.set()
        final = wait_job(runtime, job)
    assert final["phase"] == "cancelled"
    assert final["cleanup"]["released"]
    assert runtime.close()["closed"]
    with pytest.raises(RuntimeClosedError):
        runtime.submit_task(binding.binding_id, "new", "hello")


def test_failed_persistence_never_starts_hardware_and_rolls_back(runtime, monkeypatch):
    runtime, binding = runtime
    monkeypatch.setattr(runtime.store, "create", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        runtime.submit_task(binding.binding_id, "req", "hello")
    binding.session.connect.assert_not_called()
    assert not runtime.resources.records()


def test_thread_start_failure_is_persisted_without_hardware(runtime, monkeypatch):
    runtime, binding = runtime
    monkeypatch.setattr("threading.Thread.start", Mock(side_effect=RuntimeError("cannot start")))
    with pytest.raises(RuntimeError, match="cannot start"):
        runtime.submit_task(binding.binding_id, "req", "hello")
    assert not runtime.resources.records()
    binding.session.connect.assert_not_called()


def test_cleanup_failure_is_not_successfully_released(runtime, monkeypatch):
    runtime, binding = runtime
    binding.session.cleanup_report.side_effect = None
    binding.session.cleanup_report.return_value = CleanupReport(errors=("driver close failed",), connected=True)
    monkeypatch.setattr(worker, "run_robot_task", lambda *a, **kw: {"output": "done"})
    job = runtime.submit_task(binding.binding_id, "req", "hello")
    final = wait_job(runtime, job)
    assert final["phase"] == "blocked"
    assert not final["cleanup"]["released"]
    assert runtime.get_runtime_state()["blocked"]
    # Test fixture supplies explicit successful cleanup after verifying blocking.
    lease = runtime._operations[job["job_id"]][2]
    runtime.resources.finish(lease, CleanupReport())


@pytest.mark.parametrize(
    "joint_driver,joints,expected", [(True, [1.0, 2.0], True), (False, [1.0], False), (True, None, False)]
)
def test_start_pose_is_tied_to_binding_and_driver_contract(runtime, monkeypatch, joint_driver, joints, expected):
    runtime, binding = runtime
    binding.session.env.low_level = (
        SimpleNamespace(get_angles=lambda: joints, move_joint_blocking=lambda q: None)
        if joint_driver
        else SimpleNamespace()
    )
    binding.session.env.get_observation.return_value = SimpleNamespace(joints=joints, rgb=None)
    monkeypatch.setattr(worker, "run_robot_task", lambda *a, **kw: {"output": "done"})
    job = runtime.submit_task(binding.binding_id, "pose", "hello")
    final = wait_job(runtime, job)
    assert ("start_pose" in final) == expected
    if expected:
        assert final["start_pose"]["binding_id"] == binding.binding_id
        assert final["start_pose"]["joints"] == joints


def test_reopening_marks_orphan_job_blocked_without_replaying(tmp_path):
    first = Runtime(tmp_path / "workspace", resource_directory=tmp_path / "locks")
    first.store.create("request", "fingerprint", {"job_id": "orphan", "phase": "running", "owner_pid": 0})
    first.store.close()
    reopened = Runtime(tmp_path / "workspace", resource_directory=tmp_path / "locks")
    try:
        job = reopened.get_job("orphan")
        assert job["phase"] == "blocked"
        assert not job["cleanup"]["released"]
        assert not reopened._operations
    finally:
        reopened.store.close()


def test_partial_construction_rollback_failure_stays_blocked(runtime, monkeypatch):
    runtime, binding = runtime
    monkeypatch.setattr(binding, "build_session", Mock(side_effect=HardwareCleanupError("driver constructor")))
    job = runtime.submit_task(binding.binding_id, "constructor", "hello")
    try:
        final = wait_job(runtime, job)
        assert final["phase"] == "blocked"
        assert final["cleanup"]["errors"]
        assert not final["cleanup"]["released"]
    finally:
        runtime.resources.finish(runtime._operations[job["job_id"]][2], CleanupReport())


def test_next_job_cannot_enter_until_finalization_and_logs_are_detached(runtime, monkeypatch):
    runtime, binding = runtime
    finalize_entered, finish_finalization = Event(), Event()
    lease_released, finish_old_worker = Event(), Event()
    original_update = runtime.store.update
    original_cleanup = runtime.record_cleanup
    held_job = []

    def update(job_id, changes, **kwargs):
        if kwargs.get("event_kind") == "run_finished" and not held_job:
            held_job.append(job_id)
            finalize_entered.set()
            assert finish_finalization.wait(5)
        return original_update(job_id, changes, **kwargs)

    def cleanup(job_id, lease, report):
        result = original_cleanup(job_id, lease, report)
        if held_job == [job_id]:
            lease_released.set()
            assert finish_old_worker.wait(5)
        return result

    def run(_session, query, *_args, **_kwargs):
        logging.getLogger("jiuwensymbiosis.job_handoff").warning("message for %s", query)
        return {"output": query}

    monkeypatch.setattr(runtime.store, "update", update)
    monkeypatch.setattr(runtime, "record_cleanup", cleanup)
    monkeypatch.setattr(worker, "run_robot_task", run)
    first = runtime.submit_task(binding.binding_id, "first-request", "first-job")
    try:
        assert finalize_entered.wait(3)
        with pytest.raises(ResourceBusyError):
            runtime.submit_task(binding.binding_id, "second-request", "second-job")
        finish_finalization.set()
        assert lease_released.wait(3)
        second = runtime.submit_task(binding.binding_id, "second-request", "second-job")
        wait_job(runtime, second)
        # Released hardware is insufficient to discard a still-finishing thread.
        assert not runtime.close(timeout=0)["closed"]
    finally:
        finish_finalization.set()
        finish_old_worker.set()
        wait_job(runtime, first)
    first_events = str(runtime.read_events(first["job_id"], limit=1000))
    assert "message for first-job" in first_events
    assert "second-job" not in first_events
