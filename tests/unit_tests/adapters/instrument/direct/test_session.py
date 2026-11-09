"""Real Runtime admission + instrument policy, with owned software adapters."""

from __future__ import annotations

import json
import time
from copy import deepcopy

import pytest

from jiuwensymbiosis.adapters.instrument.common.naming import tool_name
from jiuwensymbiosis.adapters.instrument.direct.providers import MappedAdapterProvider
from jiuwensymbiosis.agent.lifecycle import CleanupReport
from jiuwensymbiosis.runtime.jobs import Runtime
from tests.unit_tests.adapters.instrument.direct.test_contracts_mapping import config, http_controller


class Hardware:
    def __init__(self):
        self.calls = []
        self.state = "idle"
        self.closed = False

    def snapshot(self):
        return {"status_available": True, "device_state": self.state, "source": "hardware"}

    def start(self, action, parameters):
        self.calls.append((action, deepcopy(parameters)))
        return {
            "status": "succeeded",
            "executed": True,
            "data": {"level": 3},
            "evidence": [{"field": "protocol_response_verified", "value": True, "source": "hardware"}],
        }

    def can_cancel(self, action):
        return False

    def close(self):
        self.closed = True


@pytest.fixture
def host(tmp_path, monkeypatch):
    hardware = Hardware()
    monkeypatch.setattr(MappedAdapterProvider, "create_adapter", lambda *args: hardware)
    installation = config()
    installation["devices"][0]["physical_device_id"] = "test-controller-serial-001"
    installation["devices"][0]["limits"] = {"level": {"min": -3, "max": 5, "step": 1, "unit": "documented units"}}
    names = [tool_name("controller", action) for action in installation["devices"][0]["operation_ids"]]
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "adapter": "instrument",
                "instrument": {
                    "access": "direct",
                    "installation": installation,
                    "permissions": {"local": names, "reader": [names[1]]},
                },
            }
        )
    )
    runtime = Runtime(tmp_path / "work", resource_directory=tmp_path / "locks")
    binding = runtime.prepare_binding(path)
    try:
        yield runtime, binding, hardware, names, path
    finally:
        for inv in runtime.store.list_invocations():
            runtime.wait_for_job(inv["job_id"], 2)
            if inv["status"] == "unknown":
                runtime.reconcile_invocation(
                    inv["invocation_id"],
                    status="failed",
                    completion_evidence={"software_fixture_idle": True},
                    cleanup_report=CleanupReport(),
                )
        runtime.close()


def submit(host, steps, *, key="test-job", principal="local"):
    runtime, binding, *_ = host
    job = runtime.submit_actions(binding.binding_id, key, steps, principal=principal)
    return runtime.wait_for_job(job["job_id"], 5)


@pytest.mark.parametrize("parameter", [True, 3.0, "3", 6, 2.5])
def test_schema_and_deployment_limits_reject_before_write(host, parameter):
    runtime, _, hardware, names, _ = host
    job = submit(host, [{"op": names[0], "params": {"level": parameter}}])
    assert not hardware.calls
    assert job["phase"] != "succeeded"
    assert runtime.list_invocations(job["job_id"]) == []
    assert job["cleanup"]["released"] is True


def test_device_state_and_permission_both_checked_before_io(host):
    runtime, _, hardware, names, _ = host
    hardware.state = "busy"
    denied = submit(host, [{"op": names[0], "params": {"level": 2}}])
    assert not hardware.calls and denied["phase"] != "succeeded"
    hardware.state = "idle"
    denied = submit(host, [{"op": names[0], "params": {"level": 2}}], key="reader-write", principal="reader")
    assert not hardware.calls and denied["phase"] != "succeeded"
    allowed = submit(host, [{"op": names[1], "params": {}}], key="reader-read", principal="reader")
    assert allowed["phase"] == "succeeded" and len(hardware.calls) == 1
    assert runtime.list_invocations(allowed["job_id"])[0]["principal"] == "reader"


def test_frozen_binding_action_dedupe_and_conflicting_payload(host):
    runtime, binding, hardware, names, path = host
    path.write_text("invalid changed deployment")
    steps = [{"op": names[0], "params": {"level": 2}, "request_key": "stable-device-command"}]
    first = submit(host, steps)
    assert first["phase"] == "succeeded"
    assert binding.resources == ("physical:test-controller-serial-001",)
    second = submit(host, steps, key="different-job")
    assert second["phase"] == "succeeded" and len(hardware.calls) == 1
    conflict = submit(host, [{**steps[0], "params": {"level": 3}}], key="conflicting-job")
    assert conflict["phase"] != "succeeded" and len(hardware.calls) == 1
    inv = runtime.list_invocations(first["job_id"])[0]
    assert inv["scope"] == "response" and inv["completion_evidence"]["physical_completion_verified"] is False
    assert inv["device_binding"]["physical_device_id"] == "test-controller-serial-001"


@pytest.mark.parametrize("stage", ["guard", "baseline"])
def test_mapping_preflight_failure_is_proven_unsent_and_releases_both_pending_sets(tmp_path, monkeypatch, stage):
    from jiuwensymbiosis.adapters.instrument.direct.session import InstrumentSession

    sessions = []
    bind = InstrumentSession.bind_execution

    def capture(session, context):
        sessions.append(session)
        bind(session, context)

    monkeypatch.setattr(InstrumentSession, "bind_execution", capture)
    with http_controller() as (origin, calls, state):
        state["response"] = {"ok": True, "state": "IDLE"}
        installation = config(origin)
        installation["devices"][0].update(
            physical_device_id="synthetic-preflight-device",
            protocol={
                "status_action": "controller_read",
                "state_path": ["state"],
                "state_values": {"IDLE": "idle"},
                "idle_checks": [{"path": ["ok"], "equals": True}],
            },
        )
        mapping = installation["extensions"][0]["actions"][0]["mapping"]
        if stage == "guard":
            mapping["guards"] = [{"observe": "controller_read", "checks": [{"path": ["ok"], "equals": False}]}]
        else:
            mapping["completion"] = {
                "observe": "controller_read",
                "changed": ["missing_baseline"],
                "checks": [{"path": ["ok"], "equals": True}],
            }
        name = tool_name("controller", "controller_set")
        path = tmp_path / "guard.json"
        path.write_text(
            json.dumps(
                {
                    "adapter": "instrument",
                    "instrument": {"access": "direct", "installation": installation, "permissions": {"local": [name]}},
                }
            )
        )
        runtime = Runtime(tmp_path / "work", resource_directory=tmp_path / "locks")
        try:
            binding = runtime.prepare_binding(path)
            job = runtime.submit_actions(binding.binding_id, "preflight", [{"op": name, "params": {"level": 2}}])
            final = runtime.wait_for_job(job["job_id"], 5)
            assert final["phase"] == "incomplete", final
            assert final["cleanup"]["released"]
            assert not [call for call in calls if call[0] == "POST"]
            (record,) = runtime.list_invocations(job["job_id"])
            assert record["status"] == "failed"
            assert record["completion_evidence"]["dispatch_started"] is False
            assert sessions[0].cleanup_report().pending_work == ()
            assert sessions[0]._context.pending_work == ()
            lease = runtime.resources.acquire(binding.resources)
            assert runtime.resources.finish(lease, CleanupReport())
        finally:
            runtime.close()


def test_plain_driver_error_cannot_claim_unsent_even_with_default_uncertain_false(host, monkeypatch):
    from jiuwensymbiosis.adapters.instrument.direct.errors import DriverError

    runtime, _, hardware, names, _ = host

    def failed_after_write(action, parameters):
        hardware.calls.append((action, parameters))
        raise DriverError("LOST_RECEIPT", "write happened but the receipt was lost")

    monkeypatch.setattr(hardware, "start", failed_after_write)
    final = submit(host, [{"op": names[0], "params": {"level": 2}}])
    assert len(hardware.calls) == 1
    assert final["phase"] == "blocked" and not final["cleanup"]["released"]
    assert runtime.list_invocations(final["job_id"])[0]["status"] == "unknown"


def test_unmanaged_session_cannot_open_adapter(host):
    _, binding, hardware, _, _ = host
    session = binding.build_session()
    with pytest.raises(RuntimeError, match="Runtime"):
        session.connect()
    assert not hardware.calls and not hardware.closed


def test_write_ack_keeps_owner_while_device_still_busy(host, monkeypatch):
    runtime, binding, hardware, names, _ = host
    start = hardware.start

    def accepted_but_busy(action, params):
        value = start(action, params)
        hardware.state = "busy"
        return value

    monkeypatch.setattr(hardware, "start", accepted_but_busy)
    job = runtime.submit_actions(binding.binding_id, "long-command", [{"op": names[0], "params": {"level": 2}}])
    deadline = time.monotonic() + 3
    while not hardware.calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert hardware.calls
    pending = runtime.wait_for_job(job["job_id"], 0.1)
    assert not pending["cleanup"]["released"] and not hardware.closed
    assert runtime.list_invocations(job["job_id"])[0]["status"] == "pending"
    hardware.state = "idle"
    finished = runtime.wait_for_job(job["job_id"], 3)
    assert finished["phase"] == "succeeded" and len(hardware.calls) == 1
    invocation = runtime.list_invocations(job["job_id"])[0]
    assert invocation["result"] == {"level": 3}
    assert invocation["completion_evidence"]["post_write_observation"]["device_state"] == "idle"


def test_write_ack_with_unknown_post_state_blocks_next_action(host, monkeypatch):
    runtime, _, hardware, names, _ = host
    start = hardware.start

    def accepted_with_lost_state(action, params):
        value = start(action, params)
        hardware.state = "unknown"
        return value

    monkeypatch.setattr(hardware, "start", accepted_with_lost_state)
    job = submit(host, [{"op": names[0], "params": {"level": 2}}, {"op": names[1], "params": {}}])
    assert job["phase"] == "blocked" and len(hardware.calls) == 1
    assert runtime.list_invocations(job["job_id"])[0]["status"] == "unknown"


def test_same_named_state_on_another_instrument_does_not_admit_write(host, monkeypatch):
    runtime, _, first, _, path = host
    installation = config()
    one = installation["devices"][0]
    one["physical_device_id"] = "first-device"
    two = deepcopy(one)
    two.update(device_id="another", physical_device_id="second-device")
    installation["devices"].append(two)
    second = Hardware()
    second.state = "busy"
    monkeypatch.setattr(
        MappedAdapterProvider,
        "create_adapter",
        lambda _self, profile, registry: first if profile["device_id"] == "controller" else second,
    )
    name = tool_name("another", "controller_set")
    path.write_text(
        json.dumps(
            {
                "adapter": "instrument",
                "instrument": {"access": "direct", "installation": installation, "permissions": {"local": [name]}},
            }
        )
    )
    binding = runtime.prepare_binding(path)
    job = runtime.submit_actions(binding.binding_id, "second-device", [{"op": name, "params": {"level": 2}}])
    assert runtime.wait_for_job(job["job_id"], 5)["phase"] != "succeeded"
    assert not first.calls and not second.calls


@pytest.mark.parametrize("interlock", [True, False, None, 1, "true"])
def test_mapped_conditions_require_current_boolean_evidence(tmp_path, interlock):
    with http_controller() as (origin, calls, state):
        installation = config(origin)
        profile = installation["devices"][0]
        profile["physical_device_id"] = "mapped-conditions-fixture"
        profile["required_confirmations"] = ["interlock_closed", "emergency_stop_clear", "device_ready"]
        profile["protocol"] = {
            "status_action": "controller_read",
            "state_path": ["state"],
            "state_values": {"idle": "idle"},
            "idle_checks": [{"path": ["state"], "equals": "idle"}],
        }
        state["response"] = {"ok": True, "state": "idle", "emergency_stop_clear": True, "device_ready": True}
        if interlock is not None:
            state["response"]["interlock_closed"] = interlock
        name = tool_name(profile["device_id"], "controller_set")
        source = tmp_path / "instrument.json"
        source.write_text(
            json.dumps(
                {
                    "adapter": "instrument",
                    "instrument": {"access": "direct", "installation": installation, "permissions": {"local": [name]}},
                }
            )
        )
        runtime = Runtime(tmp_path / "work", resource_directory=tmp_path / "locks")
        try:
            binding = runtime.prepare_binding(source)
            job = runtime.submit_actions(binding.binding_id, "checked-write", [{"op": name, "params": {"level": 2}}])
            result = runtime.wait_for_job(job["job_id"], 5)
            writes = [call for call in calls if call[0] == "POST"]
            if interlock is True:
                assert result["phase"] == "succeeded"
                assert writes == [("POST", "/set", {"level": 2})]
            else:
                assert result["phase"] == "incomplete"
                assert not writes
                assert not runtime.list_invocations(job["job_id"])
            assert result["cleanup"]["released"] is True
        finally:
            runtime.close()
