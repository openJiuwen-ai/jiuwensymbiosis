"""Real configuration, Runtime and SDK paths with synthetic hardware/models."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from jiuwensymbiosis.adapters.instrument.common.naming import tool_name
from jiuwensymbiosis.adapters.instrument.direct.providers import MappedAdapterProvider
from jiuwensymbiosis.agent.fast.runner import run_bound_sequence
from jiuwensymbiosis.runtime import ResourceManager, Runtime
from tests.unit_tests.adapters.instrument.direct.test_contracts_mapping import config
from tests.unit_tests.adapters.instrument.direct.test_session import Hardware


def deployment():
    installation = config()
    installation["devices"][0]["physical_device_id"] = "synthetic-meter-001"
    names = [tool_name("controller", name) for name in installation["devices"][0]["operation_ids"]]
    return {
        "adapter": "instrument",
        "instrument": {"access": "direct", "installation": installation, "permissions": {"local": names}},
    }, names


def test_managed_cli_waits_for_real_runtime_and_keeps_model_and_workspace_selection(tmp_path, monkeypatch, capsys):
    from examples import run_task
    from jiuwensymbiosis.agent import run

    data, names = deployment()
    data["name"] = "synthetic-instruments"
    data["agent"] = {"workspace": str(tmp_path / "configured-workspace")}
    source = tmp_path / "instruments.json"
    source.write_text(json.dumps(data))
    hardware = Hardware()
    monkeypatch.setattr(MappedAdapterProvider, "create_adapter", lambda *args: hardware)
    seen = []

    def run_task_without_llm(session, query, cfg, *, conversation_id, cancel_token, principal):
        seen.append((query, asdict(cfg.model_spec), cfg.workspace))
        assert not hasattr(session, "env")
        return run_bound_sequence(
            session, [{"op": names[1], "params": {}}], request_key_prefix=conversation_id, principal=principal
        )

    monkeypatch.setattr(run, "run_session_task", run_task_without_llm)
    code = run_task.main(
        [
            "--config",
            str(source),
            "--query",
            "read the synthetic measurement",
            "--model",
            "test-model",
            "--server-url",
            "http://127.0.0.1:1/v1",
            "--api-key",
            "synthetic-test-credential",
        ],
        resource_manager=ResourceManager(tmp_path / "locks"),
    )
    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["phase"] == "succeeded" and output["cleanup"]["released"]
    assert hardware.closed and len(hardware.calls) == 1
    assert seen[0][1]["model_name"] == "test-model"
    assert seen[0][1]["api_base"] == "http://127.0.0.1:1/v1"
    assert seen[0][2] == str(tmp_path / "configured-workspace")
    assert "synthetic-test-credential" not in json.dumps(output)


def test_mcp_cli_migration_keeps_original_default_workspace(tmp_path, monkeypatch, capsys):
    from examples import run_task
    from jiuwensymbiosis.agent import run
    from tests.unit_tests.adapters.instrument.mcp.test_client import remote_config

    selected = []

    def workspace(session, explicit):
        selected.append((session.name, explicit))
        return str(tmp_path / session.name)

    def read_without_model(session, query, cfg, *, conversation_id, cancel_token, principal):
        assert cfg.workspace == str(tmp_path / "instrument_mcp")
        return run_bound_sequence(
            session,
            [{"op": session.action_bindings()[0].contract.name, "params": {}}],
            request_key_prefix=conversation_id,
            principal=principal,
        )

    monkeypatch.setattr("jiuwensymbiosis.agent.builder._resolve_workspace", workspace)
    monkeypatch.setattr(run, "run_session_task", read_without_model)
    with remote_config(tmp_path, "stdio") as (cfg, _):
        remote = asdict(cfg)
        remote["headers_env"] = dict(remote["headers_env"])
        source = tmp_path / "remote.json"
        source.write_text(json.dumps({"adapter": "instrument", "instrument": {"access": "mcp", **remote}}))
        status = run_task.main(
            ["--config", str(source), "--query", "read measurement"],
            resource_manager=ResourceManager(tmp_path / "locks"),
        )
    assert status == 0
    assert json.loads(capsys.readouterr().out)["phase"] == "succeeded"
    assert selected == [("instrument_mcp", None)]


@pytest.mark.parametrize(
    "fail,principal,followup",
    [
        (False, "reader", "none"),
        (True, "reader", "none"),
        (True, "local", "next_turn"),
        (True, "local", "same_batch"),
    ],
)
def test_actual_stepagent_stops_on_unknown_without_another_device_call(
    tmp_path, monkeypatch, fail, principal, followup
):
    from openjiuwen.core.foundation.llm.schema.message import AssistantMessage
    from openjiuwen.core.foundation.llm.schema.tool_call import ToolCall

    from jiuwensymbiosis.agent.lifecycle import CleanupReport
    from jiuwensymbiosis.agent.mock_model import MockModelClient, build_mock_model

    data, names = deployment()
    data["agent"] = {
        "execution": {"mode": "stepagent"},
        "strict_capabilities": True,
        "logging": {"motion_dir": str(tmp_path / "motion")},
    }
    data["instrument"]["permissions"]["reader"] = [names[1]]
    source = tmp_path / "stepagent.json"
    source.write_text(json.dumps(data))

    class Device(Hardware):
        def start(self, action, parameters):
            if fail:
                self.calls.append((action, parameters))
                raise TimeoutError("synthetic instrument lost response")
            return super().start(action, parameters)

    hardware = Device()
    monkeypatch.setattr(MappedAdapterProvider, "create_adapter", lambda *args: hardware)
    model_calls = []

    async def response(self, messages, *, tools=None, **kwargs):
        # The SDK may probe image support in the background before planning.
        # That request must not consume the scripted instrument tool call.
        if not tools:
            return AssistantMessage(content="Synthetic model capability probe.")
        model_calls.append((messages, tools))
        if len(model_calls) == 1:
            calls = [ToolCall(id="model-call-1", type="function", name=names[1], arguments="{}")]
            if followup == "same_batch":
                calls.append(ToolCall(id="model-call-2", type="function", name=names[1], arguments="{}"))
            return AssistantMessage(tool_calls=calls)
        if followup == "next_turn":
            return AssistantMessage(
                tool_calls=[ToolCall(id="model-call-2", type="function", name=names[1], arguments="{}")]
            )
        return AssistantMessage(content="The instrument call finished; report recorded facts.")

    monkeypatch.setattr(MockModelClient, "invoke", response)
    monkeypatch.setattr("jiuwensymbiosis.agent.builder.build_model", lambda *args: build_mock_model())
    runtime = Runtime(tmp_path / "workspace", resource_directory=tmp_path / "locks")
    try:
        binding = runtime.prepare_binding(source)
        job = runtime.submit_task(binding.binding_id, "model-request", "read measurement", principal=principal)
        final = runtime.wait_for_job(job["job_id"], 15)
        assert final["phase"] == ("blocked" if fail else "succeeded"), final
        assert len(hardware.calls) == 1
        assert len(model_calls) == 1 if fail else len(model_calls) >= 2
        records = runtime.list_invocations(job["job_id"])
        assert len(records) == 1 and records[0]["principal"] == principal
        assert records[0]["request_key"].startswith(job["job_id"] + ":tool:")
        assert records[0]["status"] == ("unknown" if fail else "succeeded")
        kinds = [event["kind"] for event in runtime.read_events(job["job_id"], limit=1000)["events"]]
        assert "invocation_intent" in kinds and "step_started" in kinds and "step_finished" in kinds
        if fail:
            runtime.reconcile_invocation(
                records[0]["invocation_id"],
                status="failed",
                completion_evidence={"software_fixture_idle": True},
                cleanup_report=CleanupReport(),
            )
    finally:
        runtime.close()
        for _, _, lease in tuple(runtime._operations.values()):
            runtime.resources.finish(lease, CleanupReport())
        runtime.store.close()
