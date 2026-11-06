"""Exercise real stdio and loopback HTTP SDK transports, without hardware."""

import json
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from mcp.shared.exceptions import McpError

from jiuwensymbiosis.adapters.instrument.mcp.client import MCPConnection
from jiuwensymbiosis.adapters.instrument.mcp.config import (
    InstrumentMCPConfig,
    RemoteToolPolicy,
    input_schema_digest,
    resource_keys,
)

SERVER = Path(__file__).with_name("fixture_server.py")


@contextmanager
def remote_config(tmp_path, transport, *, tool="read_temperature", timeout_s=5):
    counter = tmp_path / "commands.jsonl"
    base = {
        "physical_device_id": "thermometer-serial-42",
        "authority": "lab-host-1",
        "transport": transport,
        "tools": (
            RemoteToolPolicy(
                tool,
                read_only=tool == "read_temperature",
                input_schema_sha256=None if tool == "read_temperature" else "0" * 64,
            ),
        ),
        "timeout_s": timeout_s,
    }
    process = None
    if transport == "stdio":
        config = InstrumentMCPConfig(**base, command=sys.executable, args=(str(SERVER), transport, str(counter)))
    else:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        process = subprocess.Popen(
            [sys.executable, str(SERVER), transport, str(counter), "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        deadline = time.monotonic() + 10
        while True:
            if process.poll() is not None:
                raise RuntimeError(process.stderr.read().decode())
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)
        config = InstrumentMCPConfig(**base, url=f"http://127.0.0.1:{port}/mcp")
    try:
        if tool != "read_temperature":
            # The test author owns this fixture and reviews its schema before
            # pinning. Production sessions never auto-approve a live schema.
            reviewer = MCPConnection(config)
            reviewer.connect()
            try:
                declaration = next(item for item in reviewer.list_tools() if item["name"] == tool)
            finally:
                reviewer.disconnect()
            config = replace(
                config,
                tools=(replace(config.tools[0], input_schema_sha256=input_schema_digest(declaration["inputSchema"])),),
            )
        yield config, counter
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            process.stderr.close()


@pytest.mark.parametrize("transport", ["stdio", "streamable_http"])
def test_raw_schema_and_full_result_are_preserved(tmp_path, transport):
    with remote_config(tmp_path, transport) as (cfg, _):
        client = MCPConnection(cfg)
        client.connect()
        try:
            tools = {tool["name"]: tool for tool in client.list_tools()}
            assert tools["read_temperature"]["outputSchema"]["properties"]["temperature_c"]["type"] == "number"
            result = client.call_tool("read_temperature", {})
            assert result["structuredContent"] == {"temperature_c": 20}
            assert [part["text"] for part in result["content"]] == ["first", "second"]
            assert result["isError"] is False
            assert result["_meta"] == {"remote_fact": "kept"}
            error = client.call_tool("error_result", {})
            assert error["isError"] is True
            assert error["structuredContent"] == {"code": "REFUSED"}
            # A malformed success payload still carries reconciliation facts.
            # The transport must return it intact for the domain's validation.
            malformed = client.call_tool("malformed_result", {})
            assert tools["malformed_result"]["outputSchema"]["properties"]["phase"]["type"] == "string"
            assert malformed["structuredContent"]["phase"] == 123
            assert malformed["structuredContent"]["job_id"] == "remote-job-7"
            assert [part["text"] for part in malformed["content"]] == ["receipt", "details"]
            assert malformed["_meta"] == {"remote_fact": "kept"}
            assert client.protocol_version == "2025-11-25"
        finally:
            client.disconnect()
        assert not client.connected


@pytest.mark.parametrize("transport", ["stdio", "streamable_http"])
def test_timeout_does_not_transparently_resend_write(tmp_path, transport):
    with remote_config(tmp_path, transport, tool="write_timeout", timeout_s=2.0) as (cfg, counter):
        client = MCPConnection(cfg)
        client.connect()
        try:
            with pytest.raises(McpError):
                client.call_tool("write_timeout", {})
            lines = [json.loads(line) for line in counter.read_text().splitlines()]
            assert [line["name"] for line in lines] == ["write_timeout"]
        finally:
            try:
                client.disconnect()
            except Exception:
                # SDK transport failure can also make cleanup uncertain; retain
                # that fact rather than treating a second empty close as proof.
                assert client.connected
                with pytest.raises(RuntimeError, match="unconfirmed"):
                    client.disconnect()


def test_delegated_identity_is_physical_and_authority_not_url():
    first = InstrumentMCPConfig(
        "serial42",
        "hostA",
        "streamable_http",
        (RemoteToolPolicy("read", read_only=True),),
        url="http://localhost:8000/mcp",
    )
    second = InstrumentMCPConfig(
        "serial42",
        "hostA",
        "streamable_http",
        (RemoteToolPolicy("read", read_only=True),),
        url="http://127.0.0.1:8001/mcp",
    )
    assert resource_keys(first) == resource_keys(second) == ("remote-owner:hostA:physical:serial42",)
    with pytest.raises(ValueError, match="explicit stable identity"):
        InstrumentMCPConfig(
            "http://localhost/device",
            "hostA",
            "streamable_http",
            (RemoteToolPolicy("read", read_only=True),),
            url=first.url,
        )
