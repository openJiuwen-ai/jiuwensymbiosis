# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under Apache-2.0.

"""Pure contract checks and isolated localhost/PTY transport verification."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import select
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from jiuwensymbiosis.adapters.instrument.direct.action_registry import ActionRegistry, InstrumentOperation
from jiuwensymbiosis.adapters.instrument.direct.config import action_ids, action_ready, load_config, validate_config
from jiuwensymbiosis.adapters.instrument.direct.contracts import ensure_json_value
from jiuwensymbiosis.adapters.instrument.direct.errors import DriverError
from jiuwensymbiosis.adapters.instrument.direct.mapped import MappedProtocolAdapter
from jiuwensymbiosis.adapters.instrument.direct.providers import MappedAdapterProvider

SOURCE = "# Generic controller\nREAD returns a JSON receipt. SET accepts an integer level.\n"


def definition(*, name="controller_set", read_only=False, transport="http_json"):
    schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    mapping = {
        "transport": "http_json",
        "method": "GET" if read_only else "POST",
        "path": "/read" if read_only else "/set",
        "response": {"field": "ok", "equals": True},
    }
    if not read_only:
        schema["properties"] = {"level": {"type": "integer", "minimum": -10, "maximum": 20}}
        schema["required"] = ["level"]
        mapping["body"] = {"level": {"$parameter": "level"}}
    if transport == "serial_line":
        mapping = {
            "transport": "serial_line",
            "request": ["READ"] if read_only else ["SET ", {"parameter": "level", "format": "d"}],
            "line_ending": "CRLF",
            "response": {"number_prefix": "VALUE ", "field": "level"} if read_only else {"contains": "ACK"},
        }
    return {
        "name": name,
        "description": "Read or configure a generic controller; a receipt does not prove a physical outcome.",
        "parameters_schema": schema,
        "read_only": read_only,
        "completion_rule": "protocol_response_verified",
        "source": {
            "document": "generic.md",
            "section": "Generic controller",
            "sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
            "excerpt": "READ returns a JSON receipt.",
        },
        "mapping": mapping,
    }


def extension(*, transport="http_json"):
    return {
        "extension_id": "generic_controller",
        "version": "1",
        "actions": [
            definition(transport=transport),
            definition(name="controller_read", read_only=True, transport=transport),
        ],
        "source_text": SOURCE,
    }


def config(origin="http://127.0.0.1:1", *, transport="http_json"):
    return {
        "config_version": "1",
        "devices": [
            {
                "device_id": "controller",
                "adapter": "mapped_protocol",
                "operation_id": "controller_set",
                "operation_ids": ["controller_set", "controller_read"],
                "model": "GenericController",
                "connection": {"base_url": origin, "timeout_s": 1.0}
                if transport == "http_json"
                else {"port": origin, "baudrate": 9600, "timeout_s": 1.0},
                "configured": True,
                "configuration_evidence": "Local test fixture only; no field hardware validation.",
            }
        ],
        "extensions": [extension(transport=transport)],
    }


def test_empty_default_catalog_and_independent_installations():
    assert not ActionRegistry().operations
    first = ActionRegistry([extension()])
    assert set(first.operations) == {"controller_set", "controller_read"}
    assert first.read_only == {"controller_read"}
    assert not ActionRegistry().operations
    assert first.action_extensions["controller_read"] == "generic_controller"
    assert first.operations["controller_set"].completion_rule == "protocol_response_verified"


def test_instrument_declaration_view_is_isolated_and_has_no_workflow():
    original = definition()
    op = InstrumentOperation.from_definition(original, "1")
    original["parameters_schema"]["properties"]["level"]["maximum"] = 999
    exported = op.request_schema
    exported["properties"]["level"]["maximum"] = 888
    assert op.request_schema["properties"]["level"]["maximum"] == 20
    assert op.required_conditions == ("idle",)
    assert not hasattr(op, "steps") and not hasattr(op, "workflow")


def test_parameters_are_validated_without_coercion_or_implicit_defaults():
    registry = ActionRegistry([extension()])
    for invalid in (True, 3.0, "3", 99):
        with pytest.raises(ValueError):
            registry.validate_parameters("controller_set", {"level": invalid})
    with pytest.raises(ValueError):
        registry.validate_parameters("controller_set", {})
    registry.validate_parameters("controller_set", {"level": 3})


def test_mcp_envelope_mapping_must_use_sdk_adapter():
    value = extension()
    value["actions"][0]["mapping"]["unwrap"] = "jsonrpc_mcp"
    with pytest.raises(ValueError, match="SDK adapter"):
        ActionRegistry([value])


@pytest.mark.parametrize("keyword", ["$ref", "$dynamicRef", "$id"])
def test_source_schema_cannot_fetch_external_content(keyword):
    value = extension()
    value["actions"][0]["output_schema"] = {keyword: "https://invalid.example/schema"}
    with pytest.raises(ValueError, match="external"):
        ActionRegistry([value])


@pytest.mark.parametrize("change", ["native", "path", "unbound", "source", "default", "coercing_boolean"])
def test_unsafe_or_ambiguous_mappings_rejected_before_io(change):
    value = extension()
    action = value["actions"][0]
    if change == "native":
        action["mapping"]["transport"] = "unsupported_device_protocol"
    elif change == "path":
        action["mapping"]["path"] = "//other-host/execute"
    elif change == "unbound":
        action["mapping"]["body"] = {}
    elif change == "source":
        action["source"]["sha256"] = "0" * 64
    elif change == "default":
        action["parameters_schema"]["properties"]["level"]["default"] = 100
    else:
        action["read_only"] = 1
    with pytest.raises(ValueError):
        ActionRegistry([value])


def test_generic_limits_readiness_and_no_object_model():
    value = config()
    value["devices"][0]["limits"] = {
        "level": {"min": -5.0, "max": 15.0, "step": 1.0, "default": 3, "unit": "controller units"}
    }
    # A documented HTTP-body minimum need not fabricate a maximum.
    del value["extensions"][0]["actions"][0]["parameters_schema"]["properties"]["level"]["maximum"]
    result = validate_config(value)
    profile = result["devices"][0]
    assert action_ids(profile) == ("controller_set", "controller_read")
    assert action_ready(profile, "controller_set")
    profile["action_readiness"]["controller_set"] = {"configured": False, "evidence": "pending review"}
    assert not action_ready(profile, "controller_set")
    assert result["object_types"] == [] and profile["slots"] == []
    for field, extra in (("object_types", [{}]),):
        invalid = copy.deepcopy(value)
        invalid[field] = extra
        with pytest.raises(ValueError):
            validate_config(invalid)
    invalid = copy.deepcopy(value)
    invalid["devices"][0]["slots"] = ["slot"]
    with pytest.raises(ValueError):
        validate_config(invalid)


def test_providers_are_explicit_and_validation_is_pure():
    calls = []

    class LocalProvider:
        def validate_profile(self, profile, registry):
            calls.append((profile["device_id"], tuple(registry.operations)))

        def binding(self, profile, registry):
            return {"resource": profile["device_id"]}

        def create_adapter(self, profile, registry):
            pytest.fail("configuration validation must not create adapters")

    value = config()
    value["devices"][0]["adapter"] = "host_supplied"
    validate_config(value, providers={"host_supplied": LocalProvider()})
    assert len(calls) == 1
    with pytest.raises(ValueError, match="explicitly installed"):
        validate_config(value)
    with pytest.raises(ValueError, match="explicitly installed"):
        validate_config(config(), providers={})


def test_duplicate_config_keys_and_cyclic_json_rejected(tmp_path):
    path = tmp_path / "ambiguous.json"
    path.write_text('{"config_version":"1","config_version":"2"}')
    with pytest.raises(ValueError, match="duplicate"):
        load_config(path)
    value = []
    value.append(value)
    with pytest.raises(ValueError, match="cyclic"):
        ensure_json_value(value)


@contextmanager
def http_controller():
    calls = []
    state = {"response": {"ok": True}, "redirect": False}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.answer()

        def do_POST(self):
            self.answer()

        def answer(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            calls.append((self.command, self.path, json.loads(body) if body else None))
            if state["redirect"]:
                self.send_response(307)
                self.send_header("Location", "/unexpected")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(state.get("raw_response", json.dumps(state["response"]).encode()))

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_loopback_http_receipt_is_not_physical_completion_and_failures_never_retry():
    with http_controller() as (origin, calls, state):
        value = validate_config(config(origin))
        registry = ActionRegistry(value["extensions"])
        provider = MappedAdapterProvider()
        profile = value["devices"][0]
        assert provider.binding(profile, registry) == provider.binding(copy.deepcopy(profile), registry)
        adapter = provider.create_adapter(profile, registry)
        try:
            result = adapter.start("controller_set", {"level": -2})
            assert result["status"] == "succeeded"
            assert result["data"]["physical_completion_verified"] is False
            assert result["data"]["completion_scope"] == "protocol_response"
            assert calls == [("POST", "/set", {"level": -2})]
            adapter.query_action("controller_read", {})
            assert calls[-1] == ("GET", "/read", None)
            state["response"] = {"ok": False}
            with pytest.raises(DriverError) as caught:
                adapter.start("controller_set", {"level": 4})
            assert caught.value.uncertain is True and len(calls) == 3
            state["redirect"] = True
            with pytest.raises(DriverError, match="never retried"):
                adapter.start("controller_set", {"level": 5})
            assert len(calls) == 4 and all(path != "/unexpected" for _, path, _ in calls)
            with pytest.raises(ValueError):
                adapter.start("controller_set", {"level": True})
            assert len(calls) == 4
        finally:
            adapter.close()


@pytest.mark.skipif(os.name != "posix", reason="PTY loopback requires POSIX")
def test_local_pty_serial_mapping_and_wrong_acknowledgement():
    import pty

    master, slave = pty.openpty()
    captured = []
    stopped = threading.Event()

    def controller():
        pending = b""
        while not stopped.is_set():
            if not select.select([master], [], [], 0.05)[0]:
                continue
            try:
                pending += os.read(master, 4096)
            except OSError:
                return
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                captured.append(line + b"\n")
                os.write(master, b"VALUE 7\n" if line == b"READ\r" else b"REJECTED\n")

    thread = threading.Thread(target=controller, daemon=True)
    value = validate_config(config(os.ttyname(slave), transport="serial_line"))
    adapter = MappedProtocolAdapter(value["devices"][0], ActionRegistry(value["extensions"]))
    thread.start()
    try:
        result = adapter.query_action("controller_read", {})
        assert result["data"]["level"] == 7.0
        with pytest.raises(DriverError, match="acknowledgement") as caught:
            adapter.start("controller_set", {"level": 4})
        assert caught.value.uncertain is True
        assert captured == [b"READ\r\n", b"SET 4\r\n"]
        assert adapter.mutation_attempts == 1
    finally:
        stopped.set()
        thread.join(2)
        adapter.close()
        os.close(master)
        os.close(slave)


@pytest.mark.parametrize(
    "origin",
    ["http://127.0.0.1:bad", "http://127.0.0.1:0", "http://user:password@localhost:80", "http://localhost\n:80"],
)
def test_http_connection_rejected_during_pure_validation(origin):
    with pytest.raises(ValueError):
        validate_config(config(origin))


def test_endpoint_aliases_share_admission_identity_but_preserve_full_binding(tmp_path):
    provider = MappedAdapterProvider()
    bindings = []
    for origin in ("http://LOCALHOST.:1234", "http://127.0.0.1:1234", "https://[::1]:1234"):
        value = validate_config(config(origin))
        bindings.append(provider.binding(value["devices"][0], ActionRegistry(value["extensions"])))
    assert all(item["endpoint"] == {"transport": "tcp", "host": "loopback", "port": 1234} for item in bindings)
    assert len({json.dumps(item["connection"], sort_keys=True) for item in bindings}) == 3
    target = tmp_path / "serial-target"
    target.touch()
    alias = tmp_path / "serial-alias"
    alias.symlink_to(target)
    endpoints = []
    for port in (str(target), str(alias)):
        value = validate_config(config(port, transport="serial_line"))
        endpoints.append(provider.binding(value["devices"][0], ActionRegistry(value["extensions"]))["endpoint"])
    assert endpoints[0] == endpoints[1]


def test_completion_requires_changed_evidence_after_baseline_before_advancing():
    with http_controller() as (origin, calls, state):
        value = config(origin)
        mapping = value["extensions"][0]["actions"][0]["mapping"]
        mapping["completion"] = {
            "observe": "controller_read",
            "changed": ["receipt_sequence"],
            "checks": [{"path": ["ok"], "equals": True}],
        }
        value = validate_config(value)
        registry = ActionRegistry(value["extensions"])
        state["response"] = {"ok": True, "receipt_sequence": 5}
        adapter = MappedProtocolAdapter(value["devices"][0], registry)
        try:
            assert adapter.start("controller_set", {"level": 2})["status"] == "running"
            assert adapter.poll()["status"] == "running"
            assert calls[:2] == [("GET", "/read", None), ("POST", "/set", {"level": 2})]
            with pytest.raises(DriverError, match="unresolved"):
                adapter.start("controller_set", {"level": 3})
            assert len(calls) == 3
            state["response"]["receipt_sequence"] = 6
            result = adapter.poll()
            assert result["status"] == "succeeded"
            assert result["data"]["physical_completion_verified"] is False
            assert sum(method == "POST" for method, _, _ in calls) == 1
        finally:
            adapter.close()


def test_ambiguous_http_response_rejected_without_retry():
    with http_controller() as (origin, calls, state):
        value = validate_config(config(origin))
        adapter = MappedProtocolAdapter(value["devices"][0], ActionRegistry(value["extensions"]))
        state["raw_response"] = b'{"ok":false,"ok":true}'
        try:
            with pytest.raises(DriverError, match="not JSON") as caught:
                adapter.start("controller_set", {"level": 1})
            assert caught.value.uncertain is True and len(calls) == 1
        finally:
            adapter.close()


@pytest.mark.skipif(os.name != "posix", reason="PTY loopback requires POSIX")
def test_ascii_serial_write_uses_runtime_state_admission_and_post_write_idle(tmp_path):
    import pty

    from jiuwensymbiosis.adapters.instrument.common.naming import tool_name
    from jiuwensymbiosis.runtime import Runtime

    master, slave = pty.openpty()
    captured = []
    stopped = threading.Event()

    def controller():
        pending = b""
        while not stopped.is_set():
            if not select.select([master], [], [], 0.05)[0]:
                continue
            try:
                pending += os.read(master, 4096)
            except OSError:
                return
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                captured.append(line)
                os.write(master, b"VALUE 7\n" if line == b"READ\r" else b"ACK\n")

    installation = config(os.ttyname(slave), transport="serial_line")
    installation["devices"][0].update(
        physical_device_id="synthetic-serial-controller",
        protocol={
            "status_action": "controller_read",
            "state_path": ["level"],
            "state_values": {"7.0": "idle"},
            "idle_checks": [{"path": ["level"], "equals": 7.0}],
        },
    )
    name = tool_name("controller", "controller_set")
    path = tmp_path / "serial.json"
    path.write_text(
        json.dumps(
            {
                "adapter": "instrument",
                "instrument": {"access": "direct", "installation": installation, "permissions": {"local": [name]}},
            }
        )
    )
    runtime = Runtime(tmp_path / "workspace", resource_directory=tmp_path / "locks")
    worker = threading.Thread(target=controller, daemon=True)
    worker.start()
    try:
        binding = runtime.prepare_binding(path)
        job = runtime.submit_actions(binding.binding_id, "serial-write", [{"op": name, "params": {"level": 4}}])
        result = runtime.wait_for_job(job["job_id"], 5)
        assert result["phase"] == "succeeded", result
        assert result["cleanup"]["released"]
        assert captured == [b"READ\r", b"SET 4\r", b"READ\r"]
        record = runtime.list_invocations(job["job_id"])[0]
        assert record["scope"] == "response"
        assert record["completion_evidence"]["post_write_observation"]["device_state"] == "idle"
    finally:
        runtime.close()
        stopped.set()
        worker.join(2)
        os.close(master)
        os.close(slave)
