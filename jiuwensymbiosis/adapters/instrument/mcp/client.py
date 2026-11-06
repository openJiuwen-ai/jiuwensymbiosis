"""Raw MCP SDK transport, without transparent tools/call retries."""

from __future__ import annotations

import os
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager
from datetime import timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jiuwensymbiosis.adapters.instrument.mcp.config import InstrumentMCPConfig


class MCPConnection:
    """Synchronous session-facing bridge to one SDK connection.

    The portal owns protocol I/O only. Runtime owns work, cancellation, records
    and device admission. Closing the transport is not proof of physical stop.
    """

    def __init__(self, config: InstrumentMCPConfig) -> None:
        self.config = config
        self.protocol_version: str | None = None
        self._stack: ExitStack | None = None
        self._session: Any = None
        self._portal: Any = None
        self._cleanup_error: Exception | None = None

    @asynccontextmanager
    async def _connection(self):
        import httpx
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamable_http_client

        async with AsyncExitStack() as stack:
            if self.config.transport == "stdio":
                streams = await stack.enter_async_context(
                    stdio_client(StdioServerParameters(command=self.config.command, args=list(self.config.args)))
                )
            else:
                headers = {header: os.environ[variable] for header, variable in self.config.headers_env}
                # No HTTP transport retries; SDK GET stream resumption does not
                # replay a command POST. Credentials are resolved only at connect.
                http = await stack.enter_async_context(
                    httpx.AsyncClient(
                        headers=headers,
                        timeout=self.config.timeout_s,
                        transport=httpx.AsyncHTTPTransport(retries=0),
                        follow_redirects=False,
                    )
                )
                streams = await stack.enter_async_context(streamable_http_client(self.config.url, http_client=http))
            session = await stack.enter_async_context(
                ClientSession(streams[0], streams[1], read_timeout_seconds=timedelta(seconds=self.config.timeout_s))
            )
            initialized = await session.initialize()
            # Tested SDK generation. A future protocol needs its own lifecycle
            # and retry audit rather than silently inheriting these guarantees.
            if initialized.protocolVersion not in {"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"}:
                raise ValueError("unvalidated MCP protocol version")
            self.protocol_version = initialized.protocolVersion
            yield session

    def connect(self) -> None:
        from anyio.from_thread import start_blocking_portal

        if self._stack is not None:
            return
        stack = ExitStack()
        try:
            portal = stack.enter_context(start_blocking_portal(name="jiuwen-mcp-transport"))
            session = stack.enter_context(portal.wrap_async_context_manager(self._connection()))
        except BaseException:
            # Connection acquisition rolls back even on host interruption.
            stack.close()
            raise
        self._stack, self._portal, self._session = stack, portal, session

    def list_tools(self) -> list[dict[str, Any]]:
        if self._session is None:
            raise RuntimeError("MCP connection is not open")
        result = []
        cursor = None
        seen = set()
        while True:
            page = self._portal.call(self._session.list_tools, cursor)
            result.extend(tool.model_dump(mode="json", by_alias=True, exclude_none=True) for tool in page.tools)
            cursor = page.nextCursor
            if cursor is None:
                return result
            if cursor in seen:
                raise ValueError("remote tools pagination repeated a cursor")
            seen.add(cursor)

    def call_tool(self, name: str, arguments: dict[str, Any], *, meta: dict[str, Any] | None = None) -> dict[str, Any]:
        from mcp import types

        if self._session is None:
            raise RuntimeError("MCP connection is not open")

        # Deliberately one call. A failed/uncertain write goes to Runtime's
        # unknown record; neither reconnect nor caller cancellation resends it.
        async def call_once():
            # ClientSession.call_tool validates outputSchema before returning,
            # losing the raw response/handle when validation fails. The public
            # SDK request API preserves the typed result for domain validation.
            return await self._session.send_request(
                types.ClientRequest(
                    types.CallToolRequest(
                        params=types.CallToolRequestParams(name=name, arguments=arguments, _meta=meta)
                    )
                ),
                types.CallToolResult,
            )

        result = self._portal.call(call_once)
        return result.model_dump(mode="json", by_alias=True, exclude_none=True)

    def disconnect(self) -> None:
        if self._cleanup_error is not None:
            raise RuntimeError("prior MCP cleanup remains unconfirmed") from self._cleanup_error
        if self._stack is not None:
            try:
                self._stack.close()
            except Exception as exc:
                # ExitStack removes callbacks even if one failed. Repeating an
                # empty close must never turn that failure into clean evidence.
                self._cleanup_error = exc
                raise
            self._stack = None
            self._session = None
            self._portal = None

    @property
    def connected(self) -> bool:
        return self._stack is not None
