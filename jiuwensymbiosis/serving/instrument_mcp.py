"""MCP projection of Runtime-managed instruments, using the installed MCP SDK.

The server has no hardware connection, action scheduler, or device state store.
An action returns a durable Runtime job handle; closing its MCP connection does
not cancel that job. Physical stop is a separate, evidence-bearing operation.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator

from jiuwensymbiosis.tools.bindings import select_bindings

if TYPE_CHECKING:
    from jiuwensymbiosis.runtime.jobs import Runtime

_MANAGEMENT = {"symbiosis_get_job", "symbiosis_wait_job", "symbiosis_request_stop"}
_REQUEST_KEY = "symbiosis/request_key"


class RuntimeMCPServer:
    """Export an explicit allowlist from the same frozen Runtime binding.

    A stdio identity is supplied by its trusted host. An HTTP deployment must
    supply an authenticator for every request; tool metadata never sets identity.
    """

    def __init__(
        self,
        runtime: Runtime,
        binding_id: str,
        *,
        allowed_tools: frozenset[str],
        authority: str,
        stdio_principal: str | None = None,
    ) -> None:
        from mcp.server.lowlevel import Server

        if not authority or not isinstance(allowed_tools, frozenset):
            raise ValueError("authority and frozen export allowlist are required")
        self.runtime, self.binding_id, self.authority = runtime, binding_id, authority
        self.allowed_tools = allowed_tools
        self.stdio_principal = stdio_principal
        self._declarations = runtime.get_binding(binding_id).build_session()
        if not hasattr(self._declarations, "action_bindings"):
            raise ValueError("MCP export requires managed action bindings")
        bindings = self._declarations.action_bindings()
        if not bindings:
            raise ValueError("MCP export requires an offline local action declaration")
        names = {binding.contract.name for binding in bindings}
        if not allowed_tools <= names or names & _MANAGEMENT:
            raise ValueError("export allowlist must name declared, nonreserved actions")
        self.server = Server("JiuwenSymbiosis instruments")
        self.server.list_tools()(self._list_tools)
        self.server.call_tool()(self._call_tool)

    def _principal(self) -> str:
        request = self.server.request_context.request
        principal = request.scope.get("symbiosis.principal") if request is not None else self.stdio_principal
        if not isinstance(principal, str) or not principal:
            raise PermissionError("MCP caller is not authenticated")
        return principal

    def _catalog(self, principal: str):
        return select_bindings(
            self._declarations.action_bindings(),
            self._declarations.capabilities,
            principal=principal,
            allow=self.allowed_tools,
        )

    async def _list_tools(self):
        from mcp import types

        principal = self._principal()
        result = []
        for name, binding in self._catalog(principal).items():
            result.append(
                types.Tool(
                    name=name,
                    description=binding.contract.description,
                    inputSchema=binding.contract.input_schema,
                    outputSchema={
                        "type": "object",
                        "properties": {
                            "job_id": {"type": "string"},
                            "phase": {"type": "string"},
                            "authority": {"type": "string"},
                            "physical_device_id": {"type": "string"},
                        },
                        "required": ["job_id", "phase", "authority", "physical_device_id"],
                    },
                    _meta={
                        "symbiosis": {
                            "authority": self.authority,
                            "physical_device_id": binding.device_id,
                            "resultSchema": binding.contract.output_schema,
                            "requestKeyMeta": _REQUEST_KEY,
                            "queryTool": "symbiosis_get_job",
                            "waitTool": "symbiosis_wait_job",
                            "stopTool": "symbiosis_request_stop",
                            "stopMeans": "request_only",
                        }
                    },
                )
            )
        if result:
            for name in sorted(_MANAGEMENT):
                properties = {"job_id": {"type": "string", "minLength": 1}}
                if name == "symbiosis_wait_job":
                    properties["timeout_s"] = {"type": "number", "minimum": 0, "maximum": 30}
                result.append(
                    types.Tool(
                        name=name,
                        description=(
                            "Read or request stop for this caller's Runtime job; "
                            "a stop request is not physical stop evidence."
                        ),
                        inputSchema={
                            "type": "object",
                            "properties": properties,
                            "required": ["job_id"],
                            "additionalProperties": False,
                        },
                    )
                )
        return result

    def _owned_job(self, job_id: str, principal: str) -> dict[str, Any]:
        job = self.runtime.get_job(job_id)
        if job.get("principal") != principal or job.get("binding_id") != self.binding_id:
            raise PermissionError("job belongs to a different caller or binding")
        return job

    def _job_view(self, job: dict[str, Any]) -> dict[str, Any]:
        # Do not expose workspace, deployment config, raw log tail or credentials.
        value = {key: job[key] for key in ("job_id", "phase", "cancel_requested", "cleanup") if key in job}
        outcome = job.get("result")
        if isinstance(outcome, dict):
            value["result"] = {
                key: outcome[key] for key in ("ok", "result", "error", "error_type", "error_code") if key in outcome
            }
        public_fields = {
            "invocation_id",
            "action_name",
            "device_binding",
            "status",
            "scope",
            "result",
            "remote_handle",
            "completion_evidence",
            "stop_requested",
            "uncertainty",
        }
        value["authority"] = self.authority
        value["invocations"] = [
            {key: item for key, item in record.items() if key in public_fields}
            for record in self.runtime.list_invocations(job["job_id"])
        ]
        return value

    async def _call_tool(self, name: str, arguments: dict[str, Any]):
        import anyio
        from mcp import types

        principal = self._principal()
        catalog = self._catalog(principal)  # permissions checked again for cached tools
        if name in _MANAGEMENT:
            if not catalog:
                raise PermissionError("caller has no exported instrument permission")
            job = self._owned_job(arguments["job_id"], principal)
            if name == "symbiosis_wait_job":
                timeout = float(arguments.get("timeout_s", 0))
                if not 0 <= timeout <= 30:
                    raise ValueError("timeout_s must be between 0 and 30")
                job = await anyio.to_thread.run_sync(self.runtime.wait_for_job, job["job_id"], timeout)
            elif name == "symbiosis_request_stop":
                self.runtime.cancel(job["job_id"])
                job = self.runtime.get_job(job["job_id"])
            value = self._job_view(job)
        else:
            if name not in catalog:
                raise PermissionError("action is not exported for this caller")
            binding = catalog[name]
            Draft202012Validator(binding.contract.input_schema).validate(arguments)
            meta = self.server.request_context.meta
            request_key = meta.model_dump().get(_REQUEST_KEY) if meta is not None else None
            if not isinstance(request_key, str) or not request_key.strip():
                raise ValueError(f"a stable {_REQUEST_KEY} is required in MCP request _meta")
            # RPC ids are transport correlation only. This application key is
            # persistent across reconnects, independently of MCP sessions.
            key = json.dumps([principal, request_key], separators=(",", ":"))
            job = self.runtime.submit_actions(
                self.binding_id,
                key,
                [{"op": name, "params": arguments, "request_key": request_key}],
                principal=principal,
            )
            value = {
                "job_id": job["job_id"],
                "phase": job["phase"],
                "authority": self.authority,
                "physical_device_id": binding.device_id,
            }
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(value, ensure_ascii=False))],
            structuredContent=value,
            isError=False,
        )

    async def run_stdio(self) -> None:
        from mcp.server.stdio import stdio_server

        if not self.stdio_principal:
            raise ValueError("stdio requires a trusted host identity")
        async with stdio_server() as (read, write):
            await self.server.run(read, write, self.server.create_initialization_options())

    def streamable_http_app(self, authenticate: Callable[[Any], str | None]):
        """Return an ASGI app. The deployment supplies authentication/TLS policy.

        Sessions are bound to their authenticated principal even if a different
        caller learns the session id. No plaintext credentials are stored here.
        """
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        manager = StreamableHTTPSessionManager(self.server, json_response=True)
        owners: dict[str, str] = {}

        @asynccontextmanager
        async def lifespan(_app):
            async with manager.run():
                yield

        async def endpoint(scope, receive, send):
            request = Request(scope, receive)
            principal = authenticate(request)
            if not isinstance(principal, str) or not principal:
                await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
                return
            session_id = request.headers.get("mcp-session-id")
            if session_id and owners.get(session_id) != principal:
                await JSONResponse({"error": "session identity mismatch"}, status_code=403)(scope, receive, send)
                return
            scope["symbiosis.principal"] = principal

            async def identified_send(message):
                if message["type"] == "http.response.start":
                    headers = dict(message.get("headers", []))
                    new_session = headers.get(b"mcp-session-id")
                    if new_session:
                        owners[new_session.decode()] = principal
                    if request.method == "DELETE" and session_id and message["status"] < 400:
                        owners.pop(session_id, None)
                await send(message)

            await manager.handle_request(scope, receive, identified_send)

        # An ASGI endpoint class avoids Starlette adapting it as Request->Response.
        class Endpoint:
            async def __call__(self, scope, receive, send):
                await endpoint(scope, receive, send)

        return Starlette(routes=[Route("/mcp", Endpoint(), methods=["GET", "POST", "DELETE"])], lifespan=lifespan)


__all__ = ["RuntimeMCPServer"]


def main() -> None:
    """Run a stdio MCP host with an explicit identity and export allowlist."""
    import argparse

    import anyio

    from jiuwensymbiosis.runtime import Runtime

    parser = argparse.ArgumentParser(description="Export Runtime-managed instrument tools over MCP stdio")
    parser.add_argument("--config", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--resource-directory")
    parser.add_argument("--authority", required=True)
    parser.add_argument("--principal", required=True)
    parser.add_argument("--allow", nargs="+", required=True)
    args = parser.parse_args()
    runtime = Runtime(args.workspace, resource_directory=args.resource_directory)
    try:
        binding = runtime.prepare_binding(args.config)
        server = RuntimeMCPServer(
            runtime,
            binding.binding_id,
            authority=args.authority,
            allowed_tools=frozenset(args.allow),
            stdio_principal=args.principal,
        )
        anyio.run(server.run_stdio)
    finally:
        # Runtime owns pending work through host shutdown. A disconnected MCP
        # client is not a physical stop acknowledgement; uncertain work blocks.
        runtime.close()


if __name__ == "__main__":
    main()
