"""Reopen actual f0bf5a0 JobStore snapshots with the new deployment envelope."""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

from jiuwensymbiosis.adapters.instrument.common.naming import tool_name
from jiuwensymbiosis.adapters.instrument.direct.providers import MappedAdapterProvider
from jiuwensymbiosis.agent.lifecycle import CleanupReport
from jiuwensymbiosis.runtime import ResourceBlockedError, Runtime
from tests.unit_tests.adapters.instrument.direct.test_session import Hardware


@pytest.fixture(params=[("direct", "succeeded"), ("direct", "unknown"), ("mcp", "succeeded"), ("mcp", "unknown")])
def legacy_host(request, tmp_path, monkeypatch):
    access, mode = request.param
    source = Path(__file__).resolve().parents[1] / "adapters/instrument" / access / "fixtures" / f"legacy-{mode}.json"
    # Only filesystem roots are substituted, never action identities or hashes.
    fixture = json.loads(source.read_text().replace("__LEGACY_ROOT__", str(tmp_path)))
    assert fixture["generated_by"] == "f0bf5a0f0124f469c825d73435af84a8cdb84c08"
    directory = tmp_path / "work/runtime"
    directory.mkdir(parents=True)
    with sqlite3.connect(directory / "jobs.sqlite3") as db:
        db.executescript(fixture["sql"])
    locks = tmp_path / "locks"
    locks.mkdir()
    for name, data in fixture["locks"].items():
        (locks / name).write_text(data)
    hardware = Hardware()
    monkeypatch.setattr(MappedAdapterProvider, "create_adapter", lambda *_: hardware)
    # The archived fixture owner exited after generation. Avoid PID reuse in
    # future test runs changing the meaning of that recorded historical fact.
    import os

    original_kill = os.kill
    runtime = Runtime(tmp_path / "work", resource_directory=locks)
    record = runtime.get_invocation(fixture["invocation_id"])
    old_pid = record["owner_pid"]

    def exited_owner(pid, signal):
        if pid == old_pid:
            raise ProcessLookupError("archived fixture owner exited")
        return original_kill(pid, signal)

    monkeypatch.setattr(os, "kill", exited_owner)
    old = fixture["old_config"]
    fixture["access"] = access
    if access == "direct":
        config = {
            "adapter": "instrument",
            "instrument": {
                "access": "direct",
                "installation": old["instruments"],
                "permissions": old["instrument_permissions"],
            },
        }
    else:
        from jiuwensymbiosis.adapters.instrument.mcp.client import MCPConnection
        from tests.unit_tests.adapters.instrument.mcp.test_client import SERVER

        config = {"adapter": "instrument", "instrument": {"access": "mcp", **old["instrument_mcp"]}}
        # The archived command belongs to its generation host; the new deployment
        # uses this environment without rewriting historical identities or hashes.
        config["instrument"]["command"] = sys.executable
        config["instrument"]["args"] = [str(SERVER), "stdio", str(tmp_path / "commands.jsonl")]
        monkeypatch.setattr(MCPConnection, "call_tool", lambda *_args, **_kw: pytest.fail("old action was resent"))
    path = tmp_path / "new.json"
    path.write_text(json.dumps(config))
    binding = runtime.prepare_binding(path)
    try:
        yield fixture, runtime, binding, hardware
    finally:
        runtime.close()
        for record in runtime.store.list_invocations():
            if record["status"] == "unknown":
                runtime.reconcile_invocation(
                    record["invocation_id"],
                    status="failed",
                    completion_evidence={"software_fixture_exited": True},
                    cleanup_report=CleanupReport(),
                )
        runtime.store.close()


def test_old_call_is_readable_and_cannot_be_replayed_by_a_new_binding(legacy_host):
    fixture, runtime, binding, hardware = legacy_host
    old = runtime.get_invocation(fixture["invocation_id"])
    assert old["binding_id"] != binding.binding_id
    assert old["status"] == fixture["mode"]
    assert runtime.get_job(fixture["job_id"])["job_id"] == fixture["job_id"]
    assert runtime.read_events(fixture["job_id"])["events"]
    step = {
        "op": tool_name("controller", "controller_set") if fixture["access"] == "direct" else old["action_name"],
        "params": {"level": 2} if fixture["access"] == "direct" else fixture.get("parameters", {}),
        "request_key": "legacy-action-key",
    }
    if fixture["mode"] == "unknown":
        with pytest.raises(ResourceBlockedError):
            runtime.submit_actions(binding.binding_id, "new-envelope", [step])
        assert runtime.get_invocation(old["invocation_id"])["status"] == "unknown"
        runtime.reconcile_invocation(
            old["invocation_id"],
            status="failed",
            completion_evidence={"software_fixture_exited": True},
            cleanup_report=CleanupReport(),
        )
    job = runtime.submit_actions(binding.binding_id, "new-envelope", [step])
    result = runtime.wait_for_job(job["job_id"], 5)
    assert result["phase"] == ("succeeded" if fixture["mode"] == "succeeded" else "incomplete"), result
    assert runtime.list_invocations(job["job_id"])[0]["invocation_id"] == old["invocation_id"]
    assert not hardware.calls
    changed = {"level": 3} if fixture["access"] == "direct" else {"temperature_c": 21}
    conflict = runtime.submit_actions(binding.binding_id, "changed-parameters", [{**step, "params": changed}])
    final = runtime.wait_for_job(conflict["job_id"], 5)
    assert final["phase"] == ("failed" if fixture["access"] == "mcp" and fixture["mode"] == "unknown" else "incomplete")
    assert not hardware.calls
