"""MCP instrument sessions use the real Runtime store and its admission."""

import json
from dataclasses import asdict, replace

import pytest

from jiuwensymbiosis.adapters.instrument.common.naming import tool_name
from jiuwensymbiosis.adapters.instrument.mcp.config import RemoteSemantics, RemoteToolPolicy
from jiuwensymbiosis.agent.lifecycle import CleanupReport
from jiuwensymbiosis.runtime import Runtime
from tests.unit_tests.adapters.instrument.mcp.test_client import remote_config


def binding_for(runtime, cfg, tmp_path):
    data = asdict(cfg)
    data["headers_env"] = dict(data["headers_env"])
    path = tmp_path / "remote.json"
    path.write_text(json.dumps({"adapter": "instrument", "instrument": {"access": "mcp", **data}}))
    return runtime.prepare_binding(path)


@pytest.mark.parametrize("transport", ["stdio", "streamable_http"])
def test_remote_numeric_result_is_kept_with_protocol_scope(tmp_path, transport):
    with remote_config(tmp_path, transport) as (cfg, _):
        cfg = replace(cfg, tools=(*cfg.tools, RemoteToolPolicy("echo_temperature", read_only=True)))
        runtime = Runtime(tmp_path / "work", resource_directory=tmp_path / "locks")
        try:
            binding = binding_for(runtime, cfg, tmp_path)
            job = runtime.submit_actions(
                binding.binding_id,
                "read-task",
                [
                    {"op": tool_name(cfg.physical_device_id, "read_temperature"), "params": {}, "bind": "measurement"},
                    {
                        "op": tool_name(cfg.physical_device_id, "echo_temperature"),
                        "params": {"temperature_c": "measurement.structuredContent.temperature_c"},
                    },
                ],
            )
            final = runtime.wait_for_job(job["job_id"], 10)
            assert final["phase"] == "succeeded", final
            invocation, echoed = runtime.list_invocations(job["job_id"])
            assert echoed["result"]["structuredContent"] == {"temperature_c": 20}
            assert invocation["result"]["structuredContent"] == {"temperature_c": 20}
            assert invocation["scope"] == "response"
            assert invocation["device_binding"] == {
                "physical_device_id": cfg.physical_device_id,
                "authority": cfg.authority,
            }
            assert "physical" not in invocation["completion_evidence"]
            assert runtime.get_runtime_state()["busy"] is False
        finally:
            runtime.close()
            runtime.store.close()


def test_remote_handle_and_physical_evidence_are_owned_by_remote_server(tmp_path):
    with remote_config(tmp_path, "stdio", tool="start") as (cfg, counter):
        semantics = RemoteSemantics(
            request_key_parameter="request_key",
            handle_field="job_id",
            status_field="phase",
            query_tool="query",
            stop_tool="stop",
            completion_scope="physical",
            evidence_field="evidence",
            evidence_equals={"sensor_stopped": True},
        )
        cfg = replace(cfg, tools=(replace(cfg.tools[0], semantics=semantics),))
        runtime = Runtime(tmp_path / "work", resource_directory=tmp_path / "locks")
        try:
            binding = binding_for(runtime, cfg, tmp_path)
            steps = [
                {"op": tool_name(cfg.physical_device_id, "start"), "params": {}, "request_key": "stable-device-request"}
            ]
            job = runtime.submit_actions(binding.binding_id, "long-job", steps)
            final = runtime.wait_for_job(job["job_id"], 10)
            assert final["phase"] == "succeeded", final
            (invocation,) = runtime.list_invocations(job["job_id"])
            assert invocation["remote_handle"] == "remote-job-7"
            assert invocation["scope"] == "physical"
            assert invocation["completion_evidence"]["remote_evidence"] == {"sensor_stopped": True}
            assert json.loads(counter.read_text())["arguments"]["request_key"] == "stable-device-request"
            duplicate = runtime.submit_actions(binding.binding_id, "new-job", steps)
            final_duplicate = runtime.wait_for_job(duplicate["job_id"], 10)
            assert final_duplicate["phase"] == "succeeded", final_duplicate
            assert len(counter.read_text().splitlines()) == 1
            revised = replace(
                cfg,
                tools=(
                    replace(
                        cfg.tools[0], semantics=replace(semantics, completion_scope="response", evidence_field=None)
                    ),
                ),
            )
            revised_binding = binding_for(runtime, revised, tmp_path)
            changed = runtime.submit_actions(revised_binding.binding_id, "changed-contract", steps)
            assert runtime.wait_for_job(changed["job_id"], 10)["phase"] != "succeeded"
            assert len(counter.read_text().splitlines()) == 1
        finally:
            runtime.close()
            runtime.store.close()


def test_remote_timeout_persists_unknown_without_resending(tmp_path):
    with remote_config(tmp_path, "stdio", tool="write_timeout", timeout_s=2.0) as (cfg, counter):
        runtime = Runtime(tmp_path / "work", resource_directory=tmp_path / "locks")
        job = None
        try:
            binding = binding_for(runtime, cfg, tmp_path)
            steps = [
                {"op": tool_name(cfg.physical_device_id, "write_timeout"), "params": {}, "request_key": "same-write"}
            ]
            job = runtime.submit_actions(binding.binding_id, "uncertain", steps)
            final = runtime.wait_for_job(job["job_id"], 10)
            assert final["phase"] == "blocked", final
            (invocation,) = runtime.list_invocations(job["job_id"])
            assert invocation["status"] == "unknown"
            assert not final["cleanup"]["released"]
            assert len(counter.read_text().splitlines()) == 1
            duplicate = runtime.submit_actions(binding.binding_id, "uncertain", steps)
            assert duplicate["job_id"] == job["job_id"]
            assert len(counter.read_text().splitlines()) == 1
        finally:
            if job is not None:
                for invocation in runtime.list_invocations(job["job_id"]):
                    if invocation["status"] == "unknown":
                        # This test owns the software-only subprocess; disconnect
                        # terminated it. This is not a hardware reconciliation.
                        runtime.reconcile_invocation(
                            invocation["invocation_id"],
                            status="failed",
                            completion_evidence={"software_fixture_exited": True},
                            cleanup_report=CleanupReport(),
                        )
            runtime.close()
            runtime.store.close()


def test_remote_widened_input_schema_rejected_before_dispatch(tmp_path):
    with remote_config(tmp_path, "stdio", tool="start") as (cfg, counter):
        cfg = replace(cfg, tools=(replace(cfg.tools[0], input_schema_sha256="f" * 64),))
        runtime = Runtime(tmp_path / "work", resource_directory=tmp_path / "locks")
        try:
            binding = binding_for(runtime, cfg, tmp_path)
            submitted = runtime.submit_actions(
                binding.binding_id,
                "unreviewed-schema",
                [{"op": tool_name(cfg.physical_device_id, "start"), "params": {}}],
            )
            job = runtime.wait_for_job(submitted["job_id"], 10)
            assert job["phase"] != "succeeded"
            assert not counter.exists()
            assert runtime.list_invocations(job["job_id"]) == []
            assert job["cleanup"]["released"]
        finally:
            runtime.close()
            runtime.store.close()


def test_write_requires_review_and_remote_schemas_cannot_fetch_urls():
    from jiuwensymbiosis.adapters.instrument.mcp.session import _validate_remote_schema

    with pytest.raises(ValueError, match="deployer-reviewed"):
        RemoteToolPolicy("unreviewed_write")
    for schema in ({"$ref": "file:///etc/passwd"}, {"$id": "https://example.invalid/schema", "$ref": "#/$defs/value"}):
        with pytest.raises(ValueError, match="local definitions"):
            _validate_remote_schema(schema)


@pytest.mark.parametrize("evidence", [{"sensor_stopped": False}, {"sensor_stopped": 1}, {"other": True}, {}, None])
@pytest.mark.parametrize("is_error", [False, True])
def test_physical_completion_requires_exact_reviewed_evidence(evidence, is_error):
    from jiuwensymbiosis.adapters.instrument.mcp.config import InstrumentMCPConfig
    from jiuwensymbiosis.adapters.instrument.mcp.session import InstrumentMCPSession

    semantics = RemoteSemantics(
        completion_scope="physical", evidence_field="evidence", evidence_equals={"sensor_stopped": True}
    )
    policy = RemoteToolPolicy("read", read_only=True, semantics=semantics)
    config = InstrumentMCPConfig("serial42", "hostA", "streamable_http", (policy,), url="http://localhost:1/mcp")
    session = InstrumentMCPSession(config)
    raw = {"content": [], "isError": is_error, "structuredContent": {"phase": "succeeded", "evidence": evidence}}
    outcome = session._outcome(raw, policy, "owner-job")
    assert outcome.status == "unknown"
    assert outcome.scope == "response"
    assert outcome.remote_handle == "owner-job"
    assert outcome.result == raw


def test_physical_predicate_is_required_and_frozen():
    with pytest.raises(ValueError, match="evidence_equals"):
        RemoteSemantics(completion_scope="physical", evidence_field="evidence")
    source = {"sensor_stopped": True}
    semantics = RemoteSemantics(completion_scope="physical", evidence_field="evidence", evidence_equals=source)
    source["sensor_stopped"] = False
    assert semantics.evidence_equals == (("sensor_stopped", True),)


@pytest.mark.parametrize("transport", ["stdio", "streamable_http"])
@pytest.mark.parametrize(
    "phase,is_error,stopped,expected",
    [
        ("failed", False, False, "unknown"),
        ("cancelled", False, False, "unknown"),
        ("running", True, False, "unknown"),
        ("failed", True, False, "unknown"),
        ("succeeded", True, True, "unknown"),
        ("failed", False, True, "failed"),
        ("cancelled", False, True, "failed"),
        ("failed", True, True, "failed"),
    ],
)
def test_remote_failure_and_error_require_reviewed_physical_stop_evidence(
    tmp_path, transport, phase, is_error, stopped, expected
):
    with remote_config(tmp_path, transport, tool="terminal_result") as (cfg, counter):
        semantics = RemoteSemantics(
            handle_field="job_id",
            status_field="phase",
            query_tool="query",
            completion_scope="physical",
            evidence_field="evidence",
            evidence_equals={"sensor_stopped": True},
        )
        cfg = replace(cfg, tools=(replace(cfg.tools[0], semantics=semantics),))
        runtime = Runtime(tmp_path / "work", resource_directory=tmp_path / "locks")
        try:
            binding = binding_for(runtime, cfg, tmp_path)
            job = runtime.submit_actions(
                binding.binding_id,
                "remote-terminal",
                [
                    {
                        "op": tool_name(cfg.physical_device_id, "terminal_result"),
                        "params": {"phase": phase, "stopped": stopped, "is_error": is_error},
                    }
                ],
            )
            final = runtime.wait_for_job(job["job_id"], 10)
            (record,) = runtime.list_invocations(job["job_id"])
            assert record["status"] == expected, final
            assert record["remote_handle"] == "remote-job-7"
            assert record["result"]["isError"] is is_error
            assert record["result"]["structuredContent"]["evidence"] == {"sensor_stopped": stopped}
            assert final["cleanup"]["released"] is (expected == "failed")
            assert bool(runtime.resources.records()) is (expected == "unknown")
            assert len(counter.read_text().splitlines()) == 1
            if expected == "failed":
                assert record["scope"] == "physical"
                assert record["completion_evidence"]["remote_evidence"] == {"sensor_stopped": True}
            else:
                assert final["phase"] == "blocked"
        finally:
            for record in runtime.store.list_invocations():
                if record["status"] == "unknown":
                    runtime.reconcile_invocation(
                        record["invocation_id"],
                        status="failed",
                        completion_evidence={"owned_software_fixture_has_no_physical_work": True},
                        cleanup_report=CleanupReport(),
                    )
            runtime.close()
            runtime.store.close()


@pytest.mark.parametrize("transport", ["stdio", "streamable_http"])
@pytest.mark.parametrize("stage", ["dispatch", "query"])
def test_remote_output_schema_failure_keeps_full_result_and_handle_for_reconciliation(tmp_path, transport, stage):
    tool = "malformed_result" if stage == "dispatch" else "start"
    with remote_config(tmp_path, transport, tool=tool) as (cfg, counter):
        semantics = RemoteSemantics(
            request_key_parameter="request_key" if stage == "query" else None,
            handle_field="job_id",
            status_field="phase",
            query_tool="query_bad",
            completion_scope="physical",
            evidence_field="evidence",
            evidence_equals={"sensor_stopped": True},
        )
        cfg = replace(cfg, tools=(replace(cfg.tools[0], semantics=semantics),))
        runtime = Runtime(tmp_path / "work", resource_directory=tmp_path / "locks")
        try:
            binding = binding_for(runtime, cfg, tmp_path)
            job = runtime.submit_actions(
                binding.binding_id, "bad-output", [{"op": tool_name(cfg.physical_device_id, tool), "params": {}}]
            )
            final = runtime.wait_for_job(job["job_id"], 10)
            (record,) = runtime.list_invocations(job["job_id"])
            assert final["phase"] == "blocked", final
            assert not final["cleanup"]["released"] and runtime.resources.records()
            assert record["status"] == "unknown"
            assert record["remote_handle"] == "remote-job-7"
            raw = record["result"]
            assert raw["structuredContent"] == {
                "job_id": "remote-job-7",
                "phase": 123,
                "evidence": {"sensor_stopped": False},
            }
            assert [item["text"] for item in raw["content"]] == ["receipt", "details"]
            assert raw["_meta"] == {"remote_fact": "kept"} and raw["isError"] is False
            assert record["completion_evidence"]["output_schema_valid"] is False
            calls = [json.loads(line)["name"] for line in counter.read_text().splitlines()]
            assert calls == (["malformed_result"] if stage == "dispatch" else ["start", "query_bad"])
        finally:
            for record in runtime.store.list_invocations():
                if record["status"] == "unknown":
                    runtime.reconcile_invocation(
                        record["invocation_id"],
                        status="failed",
                        completion_evidence={"owned_software_fixture_has_no_physical_work": True},
                        cleanup_report=CleanupReport(),
                    )
            runtime.close()
            runtime.store.close()
