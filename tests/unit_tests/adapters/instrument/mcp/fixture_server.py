"""A real MCP SDK server; all 'device' effects are append-only test-file writes."""

import argparse
import json
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager


def build_server(counter: Path) -> Server:
    server = Server("synthetic-third-party-instrument")
    polls = 0

    @server.list_tools()
    async def tools():
        empty = {"type": "object", "properties": {}, "additionalProperties": False}
        task_result = {
            "type": "object",
            "properties": {"job_id": {"type": "string"}, "phase": {"type": "string"}, "evidence": {"type": "object"}},
            "required": ["job_id", "phase", "evidence"],
        }
        return [
            types.Tool(
                name="read_temperature",
                inputSchema=empty,
                outputSchema={
                    "type": "object",
                    "properties": {"temperature_c": {"type": "number"}},
                    "required": ["temperature_c"],
                },
            ),
            types.Tool(name="error_result", inputSchema=empty),
            types.Tool(
                name="echo_temperature",
                inputSchema={
                    "type": "object",
                    "properties": {"temperature_c": {"type": "number"}},
                    "required": ["temperature_c"],
                    "additionalProperties": False,
                },
                outputSchema={
                    "type": "object",
                    "properties": {"temperature_c": {"type": "number"}},
                    "required": ["temperature_c"],
                },
            ),
            types.Tool(name="write_timeout", inputSchema=empty),
            types.Tool(
                name="start",
                inputSchema={
                    "type": "object",
                    "properties": {"request_key": {"type": "string"}},
                    "required": ["request_key"],
                    "additionalProperties": False,
                },
                outputSchema={
                    "type": "object",
                    "properties": {"job_id": {"type": "string"}, "phase": {"type": "string"}},
                    "required": ["job_id", "phase"],
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="query",
                inputSchema={"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]},
                outputSchema={
                    "type": "object",
                    "properties": {"phase": {"type": "string"}, "evidence": {"type": "object"}},
                    "required": ["phase", "evidence"],
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="stop",
                inputSchema={"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]},
            ),
            types.Tool(
                name="terminal_result",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "phase": {"type": "string", "enum": ["failed", "cancelled", "running", "succeeded"]},
                        "stopped": {"type": "boolean"},
                        "is_error": {"type": "boolean"},
                    },
                    "required": ["phase", "stopped", "is_error"],
                    "additionalProperties": False,
                },
                outputSchema=task_result,
            ),
            types.Tool(name="malformed_result", inputSchema=empty, outputSchema=task_result),
            types.Tool(
                name="query_bad",
                inputSchema={"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]},
                outputSchema=task_result,
            ),
        ]

    @server.call_tool()
    async def call(name, arguments):
        nonlocal polls
        if name == "read_temperature":
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="first"), types.TextContent(type="text", text="second")],
                structuredContent={"temperature_c": 20},
                isError=False,
                _meta={"remote_fact": "kept"},
            )
        if name == "error_result":
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="not successful")],
                structuredContent={"code": "REFUSED"},
                isError=True,
            )
        if name == "echo_temperature":
            return dict(arguments)
        if name in {"write_timeout", "start", "stop", "terminal_result", "malformed_result", "query_bad"}:
            with counter.open("a") as output:
                output.write(json.dumps({"name": name, "arguments": arguments}) + "\n")
        if name in {"terminal_result", "malformed_result", "query_bad"}:
            return types.CallToolResult(
                content=[
                    types.TextContent(type="text", text="receipt"),
                    types.TextContent(type="text", text="details"),
                ],
                structuredContent={
                    "job_id": arguments.get("job_id", "remote-job-7"),
                    "phase": arguments["phase"] if name == "terminal_result" else 123,
                    "evidence": {"sensor_stopped": arguments.get("stopped", False)},
                },
                isError=arguments.get("is_error", False),
                _meta={"remote_fact": "kept"},
            )
        if name == "write_timeout":
            await anyio.sleep(10)
        if name == "start":
            return {"job_id": "remote-job-7", "phase": "running"}
        if name == "query":
            polls += 1
            return {
                "phase": "succeeded" if polls >= 2 else "running",
                "evidence": {"sensor_stopped": True},
            }
        if name == "stop":
            return {"accepted": True}
        return {}

    return server


async def main(args):
    server = build_server(Path(args.counter))
    if args.transport == "stdio":
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
    else:
        import uvicorn
        from starlette.applications import Starlette
        from starlette.routing import Route

        manager = StreamableHTTPSessionManager(server, json_response=True)

        @asynccontextmanager
        async def lifespan(_app):
            async with manager.run():
                yield

        class Endpoint:
            async def __call__(self, scope, receive, send):
                await manager.handle_request(scope, receive, send)

        app = Starlette(routes=[Route("/mcp", Endpoint(), methods=["GET", "POST", "DELETE"])], lifespan=lifespan)
        await uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="error")).serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("transport", choices=["stdio", "streamable_http"])
    parser.add_argument("counter")
    parser.add_argument("--port", type=int, default=0)
    anyio.run(main, parser.parse_args())
