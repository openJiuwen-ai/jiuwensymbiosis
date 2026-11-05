"""Instrument-domain MCP calls; the remote server remains the device owner."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator

from jiuwensymbiosis.adapters.instrument.common.naming import tool_name
from jiuwensymbiosis.adapters.instrument.common.vocabulary import CAPABILITIES
from jiuwensymbiosis.adapters.instrument.mcp.client import MCPConnection
from jiuwensymbiosis.adapters.instrument.mcp.config import (
    InstrumentMCPConfig,
    RemoteToolPolicy,
    input_schema_digest,
    resource_keys,
)
from jiuwensymbiosis.agent.lifecycle import CleanupReport
from jiuwensymbiosis.runtime.invocations import InvocationOutcome
from jiuwensymbiosis.tools.bindings import ActionBinding

if TYPE_CHECKING:
    from jiuwensymbiosis.runtime.invocations import InvocationContext


def _field(value: dict[str, Any], path: str | None) -> Any:
    if path is None:
        return None
    current: Any = value
    for segment in path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]
    return current


def _validate_remote_schema(schema: dict[str, Any]) -> None:
    """Remote schemas are data, never authority to fetch another URL/file."""

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"$ref", "$dynamicRef", "$id"} and (not isinstance(item, str) or not item.startswith("#")):
                    raise ValueError("remote schemas may only reference local definitions")
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    if len(json.dumps(schema, allow_nan=False)) > 200_000:
        raise ValueError("remote schema exceeds the supported size")
    visit(schema)
    Draft202012Validator.check_schema(schema)


@dataclass(frozen=True)
class RemoteContractView:
    """Immutable projection from the full SDK Tool, plus reviewed host policy."""

    name: str
    _tool_json: str
    policy: RemoteToolPolicy
    _query_tool_json: str | None = None

    @property
    def remote_tool(self) -> dict[str, Any]:
        return json.loads(self._tool_json)

    @property
    def query_tool(self) -> dict[str, Any] | None:
        return json.loads(self._query_tool_json) if self._query_tool_json is not None else None

    @property
    def description(self) -> str:
        return self.remote_tool.get("description", self.policy.name)

    @property
    def input_schema(self) -> dict[str, Any]:
        schema = self.remote_tool["inputSchema"]
        key = self.policy.semantics.request_key_parameter if self.policy.semantics else None
        if key:
            schema.get("properties", {}).pop(key, None)
            if key in schema.get("required", []):
                schema["required"].remove(key)
        return schema

    @property
    def output_schema(self) -> dict[str, Any]:
        # A generic reference can address structuredContent.temperature_c while
        # the original content blocks and outer error flag remain available.
        structured = self.remote_tool.get("outputSchema", {"type": "object"})
        if self._query_tool_json is not None:
            query_schema = json.loads(self._query_tool_json).get("outputSchema", {"type": "object"})
            structured = {"anyOf": [structured, query_schema]}
        return {
            "type": "object",
            "properties": {
                "content": {"type": "array"},
                "structuredContent": structured,
                "isError": {"type": "boolean"},
            },
            "required": ["content", "isError"],
        }

    @property
    def required_capabilities(self) -> frozenset[str]:
        return frozenset({"instrument.observe" if self.policy.read_only else "instrument.control"})

    @property
    def planner_visible(self) -> bool:
        return True

    @property
    def version(self) -> str:
        return hashlib.sha256(
            json.dumps([self.remote_tool, self._query_tool_json, asdict(self.policy)], sort_keys=True).encode()
        ).hexdigest()


class InstrumentMCPSession:
    """Connection lifecycle and policy, with no local physical task coordinator."""

    capabilities = CAPABILITIES

    def __init__(self, config: InstrumentMCPConfig, *, connection: MCPConnection | None = None) -> None:
        self.name = config.physical_device_id
        self.config = config
        self.cancel_token = None
        self._connection = connection or MCPConnection(config)
        self._context: InvocationContext | None = None
        self._views: tuple[RemoteContractView, ...] = ()
        self._cleanup_errors: tuple[str, ...] = ()

    def bind_execution(self, context: InvocationContext) -> None:
        if self._context is not None or self._connection.connected:
            raise RuntimeError("remote execution binding is frozen before connect")
        self._context = context

    @property
    def execution_request_prefix(self) -> str:
        return self._context.job_id if self._context is not None else ""

    def connect(self) -> None:
        if self._context is None:
            raise RuntimeError("MCP instrument session requires Runtime admission")
        if self._connection.connected:
            return
        try:
            self._connection.connect()
            tools = {item["name"]: item for item in self._connection.list_tools()}
            for tool in tools.values():
                _validate_remote_schema(tool["inputSchema"])
                if tool.get("outputSchema") is not None:
                    _validate_remote_schema(tool["outputSchema"])
            views = []
            for policy in self.config.tools:
                tool = tools[policy.name]
                if (
                    policy.input_schema_sha256 is not None
                    and input_schema_digest(tool["inputSchema"]) != policy.input_schema_sha256
                ):
                    raise ValueError("remote input schema differs from the deployer-reviewed pin")
                advertised = tool.get("_meta", {}).get("symbiosis")
                if advertised is not None:
                    if not isinstance(advertised, dict):
                        raise ValueError("remote device/authority differs from the frozen deployment binding")
                    if (
                        advertised.get("authority") != self.config.authority
                        or advertised.get("physical_device_id") != self.config.physical_device_id
                    ):
                        raise ValueError("remote device/authority differs from the frozen deployment binding")
                semantics = policy.semantics
                if semantics:
                    for name in (semantics.query_tool, semantics.stop_tool):
                        if name is not None and name not in tools:
                            raise ValueError("reviewed remote management tool is unavailable")
                    if semantics.request_key_parameter and semantics.request_key_parameter not in tool[
                        "inputSchema"
                    ].get("properties", {}):
                        raise ValueError("reviewed remote request key is absent from input schema")
                views.append(
                    RemoteContractView(
                        tool_name(self.config.physical_device_id, policy.name),
                        json.dumps(tool, sort_keys=True, allow_nan=False),
                        policy,
                        json.dumps(tools[semantics.query_tool], sort_keys=True, allow_nan=False)
                        if semantics and semantics.query_tool
                        else None,
                    )
                )
            self._views = tuple(views)
        except BaseException:
            # This boundary acquired only a protocol connection, never a device
            # command. Preserve cleanup errors for Runtime's release decision.
            self.disconnect()
            raise

    def action_bindings(self) -> tuple[ActionBinding, ...]:
        return tuple(
            ActionBinding(
                contract=view,
                device_id=self.config.physical_device_id,
                resource_keys=resource_keys(self.config),
                domain="instrument",
                call=lambda params, key, principal, v=view: self._invoke(v, params, key, principal),
                authorize=lambda principal, p=view.policy: principal in p.principals,
            )
            for view in self._views
        )

    def _outcome(
        self,
        raw: dict[str, Any],
        policy: RemoteToolPolicy,
        prior_handle: Any = None,
        *,
        output_schema: dict[str, Any] | None = None,
    ) -> InvocationOutcome:
        semantics = policy.semantics
        structured = raw.get("structuredContent") or {}
        handle = _field(structured, semantics.handle_field) if semantics else None
        if handle is None:
            handle = prior_handle
        evidence = {
            "authority": self.config.authority,
            "physical_device_id": self.config.physical_device_id,
            "protocol_version": self._connection.protocol_version,
        }
        is_error = raw.get("isError", False)
        # MCP error results need not match a success outputSchema. For other
        # responses, invalid content is retained as evidence, never discarded
        # by transport validation or promoted to an authoritative result.
        if output_schema is not None and not is_error:
            try:
                Draft202012Validator(output_schema).validate(raw.get("structuredContent"))
            except Exception as exc:
                return InvocationOutcome(
                    "unknown",
                    raw,
                    "response",
                    {**evidence, "output_schema_valid": False, "validation_error": type(exc).__name__},
                    handle,
                )
        if semantics is None:
            return InvocationOutcome(
                "failed" if is_error else "succeeded",
                raw,
                "response",
                {**evidence, "protocol_response": True, "isError": is_error},
                handle,
            )
        status = _field(structured, semantics.status_field)
        failed = is_error
        if semantics.status_field:
            failed = status in semantics.failure_values
            if not failed and (status not in semantics.success_values or is_error):
                # An errored query/response is not evidence that the remote
                # task ended, even when it also carries a valid task handle.
                return InvocationOutcome(
                    "pending" if handle is not None and not is_error else "unknown", raw, "response", evidence, handle
                )
        completion = _field(structured, semantics.evidence_field)
        if semantics.completion_scope == "physical":
            verified = isinstance(completion, dict) and all(
                key in completion and type(completion[key]) is type(expected) and completion[key] == expected
                for key, expected in dict(semantics.evidence_equals).items()
            )
            if not verified:
                return InvocationOutcome("unknown", raw, "response", evidence, handle)
        return InvocationOutcome(
            "failed" if failed else "succeeded",
            raw,
            semantics.completion_scope,
            {**evidence, "remote_status": status, "remote_evidence": completion, "isError": is_error},
            handle,
        )

    def _invoke(self, view: RemoteContractView, parameters: dict[str, Any], key: str | None, principal: str) -> Any:
        if self._context is None or not self._connection.connected:
            raise RuntimeError("remote action requires a connected Runtime session")
        context = self._context
        if principal != context.principal or principal not in view.policy.principals:
            raise PermissionError("remote instrument caller is not authorized")
        semantics = view.policy.semantics

        def validate() -> None:
            json.dumps(parameters, allow_nan=False)
            Draft202012Validator(view.input_schema).validate(parameters)
            if semantics and semantics.request_key_parameter in parameters:
                raise ValueError("remote deduplication key is supplied by Runtime, not model parameters")

        def dispatch(_invocation_id: str, request_key: str) -> InvocationOutcome:
            arguments = deepcopy(parameters)
            if semantics and semantics.request_key_parameter:
                arguments[semantics.request_key_parameter] = request_key
            meta = {semantics.request_key_meta: request_key} if semantics and semantics.request_key_meta else None
            raw = self._connection.call_tool(view.policy.name, arguments, meta=meta)
            return self._outcome(raw, view.policy, output_schema=view.remote_tool.get("outputSchema"))

        def poll(record: dict[str, Any]) -> InvocationOutcome:
            raw = self._connection.call_tool(
                semantics.query_tool, {semantics.handle_parameter: record["remote_handle"]}
            )
            return self._outcome(
                raw,
                view.policy,
                record["remote_handle"],
                output_schema=(view.query_tool or {}).get("outputSchema"),
            )

        def stop(record: dict[str, Any]) -> None:
            # Acknowledging this call is only a stop request. The next owner
            # query, not protocol cancellation, must provide terminal evidence.
            raw = self._connection.call_tool(semantics.stop_tool, {semantics.handle_parameter: record["remote_handle"]})
            if raw.get("isError"):
                raise RuntimeError("remote owner rejected the stop request")

        if not key:
            raise ValueError("remote instrument action requires a stable Runtime request key")
        return context.invoke(
            resource_key=resource_keys(self.config)[0],
            action_name=view.name,
            contract_version=view.version,
            parameters=parameters,
            request_key=key,
            validate=validate,
            dispatch=dispatch,
            poll=poll if semantics and semantics.query_tool else None,
            stop=stop if semantics and semantics.stop_tool else None,
            device_binding={"physical_device_id": self.config.physical_device_id, "authority": self.config.authority},
        )

    def disconnect(self) -> None:
        try:
            self._connection.disconnect()
        except Exception as exc:
            self._cleanup_errors = (f"MCP connection cleanup failed: {type(exc).__name__}",)
            raise
        self._cleanup_errors = ()

    def cleanup_report(self) -> CleanupReport:
        return CleanupReport(
            connected=self._connection.connected,
            errors=self._cleanup_errors,
            pending_work=self._context.pending_work if self._context else (),
        )
