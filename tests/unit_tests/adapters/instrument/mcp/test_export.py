"""An external SDK client uses the same Runtime as direct instrument calls."""

import json
import os
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager, contextmanager

import httpx
import pytest
import uvicorn
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from jiuwensymbiosis.adapters.instrument.common.naming import tool_name
from jiuwensymbiosis.adapters.instrument.mcp.config import input_schema_digest
from jiuwensymbiosis.runtime import ResourceBusyError
from jiuwensymbiosis.serving.instrument_mcp import RuntimeMCPServer
from tests.unit_tests.adapters.instrument.direct.test_contracts_mapping import config as installation_config
from tests.unit_tests.adapters.instrument.direct.test_session import host as host  # fixture shared at the hardware seam


@contextmanager
def export_http(runtime, binding, names):
    export = RuntimeMCPServer(
        runtime, binding.binding_id, allowed_tools=frozenset(names), authority="fixture-control-host"
    )
    app = export.streamable_http_app(
        lambda request: {"Bearer fixture-local": "local", "Bearer fixture-reader": "reader"}.get(
            request.headers.get("authorization")
        )
    )
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("test MCP export did not start")
        time.sleep(0.01)
    try:
        yield export, f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        thread.join(5)
        sock.close()


@asynccontextmanager
async def client(url, principal="local"):
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer fixture-{principal}"}) as http:
        async with streamable_http_client(url, http_client=http) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


async def test_export_same_schema_authorization_and_runtime_deduplication(host):
    runtime, binding, hardware, names, _ = host
    with export_http(runtime, binding, names) as (export, url):
        async with client(url) as sdk:
            tools = {tool.name: tool for tool in (await sdk.list_tools()).tools}
            original = {item.contract.name: item for item in binding.build_session().action_bindings()}
            assert tools[names[0]].inputSchema == original[names[0]].contract.input_schema
            assert tools[names[0]].meta["symbiosis"]["resultSchema"] == original[names[0]].contract.output_schema
            assert "confirm_task" not in tools and "reconcile" not in tools
            missing_key = await sdk.call_tool(names[0], {"level": 2})
            assert missing_key.isError and not hardware.calls
            submitted = await sdk.call_tool(names[0], {"level": 2}, meta={"symbiosis/request_key": "command-1"})
            assert not submitted.isError
            handle = submitted.structuredContent
            result = await sdk.call_tool("symbiosis_wait_job", {"job_id": handle["job_id"], "timeout_s": 5})
            assert result.structuredContent["phase"] == "succeeded", result
            assert len(result.structuredContent["invocations"]) == 1
            assert "workspace" not in json.dumps(result.structuredContent)
            assert "log_tail" not in json.dumps(result.structuredContent)
            assert hardware.calls == [("controller_set", {"level": 2})]
        # A new transport/RPC/session id retains the same application request key.
        async with client(url) as sdk:
            repeated = await sdk.call_tool(names[0], {"level": 2}, meta={"symbiosis/request_key": "command-1"})
            assert repeated.structuredContent["job_id"] == handle["job_id"]
            assert len(hardware.calls) == 1
            conflict = await sdk.call_tool(names[0], {"level": 3}, meta={"symbiosis/request_key": "command-1"})
            assert conflict.isError and len(hardware.calls) == 1
        async with client(url, "reader") as sdk:
            available = {tool.name for tool in (await sdk.list_tools()).tools}
            assert names[0] not in available and names[1] in available
            # Cached/disclosed names do not bypass call-time authorization.
            denied = await sdk.call_tool(names[0], {"level": 2}, meta={"symbiosis/request_key": "forged"})
            assert denied.isError and len(hardware.calls) == 1
            denied_job = await sdk.call_tool("symbiosis_get_job", {"job_id": handle["job_id"]})
            assert denied_job.isError
        direct = runtime.submit_actions(
            binding.binding_id, "after-mcp", [{"op": names[0], "params": {"level": 2}, "request_key": "command-1"}]
        )
        assert runtime.wait_for_job(direct["job_id"], 5)["phase"] == "succeeded"
        assert len(hardware.calls) == 1
        assert export.runtime is runtime


async def test_direct_then_mcp_same_action_key_dispatches_once(host):
    runtime, binding, hardware, names, _ = host
    direct = runtime.submit_actions(
        binding.binding_id, "direct-job", [{"op": names[0], "params": {"level": 2}, "request_key": "shared-action-key"}]
    )
    assert runtime.wait_for_job(direct["job_id"], 5)["phase"] == "succeeded"
    with export_http(runtime, binding, names) as (_, url):
        async with client(url) as sdk:
            submitted = await sdk.call_tool(names[0], {"level": 2}, meta={"symbiosis/request_key": "shared-action-key"})
            result = await sdk.call_tool(
                "symbiosis_wait_job", {"job_id": submitted.structuredContent["job_id"], "timeout_s": 5}
            )
            assert result.structuredContent["phase"] == "succeeded"
            assert hardware.calls == [("controller_set", {"level": 2})]
            assert (
                result.structuredContent["invocations"][0]["invocation_id"]
                == runtime.list_invocations(direct["job_id"])[0]["invocation_id"]
            )


async def test_long_action_handle_survives_disconnect_and_direct_path_competes(host):
    runtime, binding, hardware, names, _ = host
    entered, finish = threading.Event(), threading.Event()
    actual_start = hardware.start

    def long_start(action, parameters):
        entered.set()
        if not finish.wait(10):
            raise TimeoutError("software fixture was not released")
        return actual_start(action, parameters)

    hardware.start = long_start
    try:
        with export_http(runtime, binding, names) as (_, url):
            async with client(url) as sdk:
                submitted = await sdk.call_tool(names[0], {"level": 2}, meta={"symbiosis/request_key": "long-command"})
                handle = submitted.structuredContent
                assert handle["job_id"] and entered.wait(2)
                assert not finish.is_set() and not hardware.calls
            # Transport is closed but device-owning Runtime is still executing.
            assert runtime.get_job(handle["job_id"])["phase"] == "running"
            with pytest.raises(ResourceBusyError):
                runtime.submit_actions(binding.binding_id, "direct-overlap", [{"op": names[0], "params": {"level": 3}}])
            async with client(url) as sdk:
                stop = await sdk.call_tool("symbiosis_request_stop", {"job_id": handle["job_id"]})
                assert stop.structuredContent["cancel_requested"] is True
                assert stop.structuredContent["cleanup"]["released"] is False
                finish.set()
                final = await sdk.call_tool("symbiosis_wait_job", {"job_id": handle["job_id"], "timeout_s": 5})
                assert final.structuredContent["cleanup"]["released"] is True
                assert len(hardware.calls) == 1
    finally:
        finish.set()


def test_exported_instrument_can_be_imported_through_mcp_adapter(host, tmp_path):
    runtime, binding, hardware, names, _ = host
    with export_http(runtime, binding, names) as (_, url):
        # Separate process models a separate calling host. The device-control
        # Runtime alone owns physical:<id>; the client holds a delegated key.
        data = {
            "adapter": "instrument",
            "instrument": {
                "access": "mcp",
                "physical_device_id": "test-controller-serial-001",
                "authority": "fixture-control-host",
                "transport": "streamable_http",
                "url": url,
                "headers_env": {"Authorization": "MCP_FIXTURE_AUTH"},
                "tools": [
                    {
                        "name": names[0],
                        "input_schema_sha256": input_schema_digest(
                            next(
                                item.contract.input_schema
                                for item in binding.build_session().action_bindings()
                                if item.contract.name == names[0]
                            )
                        ),
                        "semantics": {
                            "request_key_meta": "symbiosis/request_key",
                            "handle_field": "job_id",
                            "status_field": "phase",
                            "query_tool": "symbiosis_get_job",
                            "stop_tool": "symbiosis_request_stop",
                            "completion_scope": "response",
                        },
                    }
                ],
            },
        }
        config = tmp_path / "import-export.json"
        config.write_text(json.dumps(data))
        output = tmp_path / "remote-result.json"
        code = """
import json,sys
from pathlib import Path
from jiuwensymbiosis.runtime import Runtime
from jiuwensymbiosis.adapters.instrument.common.naming import tool_name
runtime = Runtime(Path(sys.argv[1]).parent / 'remote-work', resource_directory=Path(sys.argv[1]).parent / 'locks')
try:
    binding = runtime.prepare_binding(sys.argv[1])
    op = tool_name('test-controller-serial-001', sys.argv[3])
    submitted = runtime.submit_actions(binding.binding_id, 'remote-job', [{'op':op, 'params':{'level':2},'request_key':'cross-host-key'}])
    job = runtime.wait_for_job(submitted['job_id'], 15)
    Path(sys.argv[2]).write_text(json.dumps({'job':job,'invocations':runtime.list_invocations(job['job_id'])}))
finally:
    runtime.close()
"""
        completed = subprocess.run(
            [sys.executable, "-c", code, str(config), str(output), names[0]],
            env={**os.environ, "MCP_FIXTURE_AUTH": "Bearer fixture-local"},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stderr
        evidence = json.loads(output.read_text())
        assert evidence["job"]["phase"] == "succeeded", evidence
        assert hardware.calls == [("controller_set", {"level": 2})]
        (invocation,) = evidence["invocations"]
        assert invocation["remote_handle"] == invocation["result"]["structuredContent"]["job_id"]
        remote_job = runtime.get_job(invocation["remote_handle"])
        (remote_invocation,) = runtime.list_invocations(remote_job["job_id"])
        assert remote_invocation["request_key"] == "cross-host-key"
        assert invocation["scope"] == "response"
        assert (
            invocation["device_binding"]["physical_device_id"]
            == remote_invocation["device_binding"]["physical_device_id"]
        )


async def test_stdio_export_cli_uses_runtime_and_owned_http_adapter(tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    calls = []

    class Device(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true,"level":3}')

        def log_message(self, *args):
            pass

    device = ThreadingHTTPServer(("127.0.0.1", 0), Device)
    thread = threading.Thread(target=device.serve_forever, daemon=True)
    thread.start()
    try:
        installation = installation_config(f"http://127.0.0.1:{device.server_port}")
        installation["devices"][0]["physical_device_id"] = "stdio-fixture-001"
        name = tool_name("controller", "controller_read")
        config = tmp_path / "stdio-config.json"
        config.write_text(
            json.dumps(
                {
                    "adapter": "instrument",
                    "instrument": {"access": "direct", "installation": installation, "permissions": {"local": [name]}},
                }
            )
        )
        params = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "jiuwensymbiosis.serving.instrument_mcp",
                "--config",
                str(config),
                "--workspace",
                str(tmp_path / "stdio-work"),
                "--resource-directory",
                str(tmp_path / "stdio-locks"),
                "--authority",
                "stdio-host",
                "--principal",
                "local",
                "--allow",
                name,
            ],
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as sdk:
                initialized = await sdk.initialize()
                assert initialized.protocolVersion == "2025-11-25"
                available = {item.name for item in (await sdk.list_tools()).tools}
                assert name in available
                submitted = await sdk.call_tool(name, {}, meta={"symbiosis/request_key": "stdio-stable"})
                assert not submitted.isError, submitted
                final = await sdk.call_tool(
                    "symbiosis_wait_job", {"job_id": submitted.structuredContent["job_id"], "timeout_s": 5}
                )
                assert final.structuredContent["phase"] == "succeeded", final
                assert final.structuredContent["invocations"][0]["result"]["level"] == 3
        assert calls == ["/read"]
    finally:
        device.shutdown()
        device.server_close()
        thread.join(2)
