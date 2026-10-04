# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Bounded HTTP ingress and inference admission for the optional model servers."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from jiuwensymbiosis.utils.validation import require_positive_number


def error_response(status: int, request_id: str | None, code: str, message: str) -> JSONResponse:
    """Return the shared versioned failure envelope without exception internals."""
    return JSONResponse(
        status_code=status,
        content={
            "schema_version": 1,
            "request_id": request_id,
            "error": {"code": code, "message": message, "retryable": status in {429, 503}},
        },
    )


def configure_service(app: FastAPI, *, max_request_bytes: int = 16_777_216) -> None:
    """Bound raw request bytes before JSON parsing."""
    if type(max_request_bytes) is not int or max_request_bytes <= 0:
        raise ValueError("max_request_bytes must be a positive integer")

    class BoundedIngress:
        def __init__(self, app: Any) -> None:
            self.app = app

        async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
            if scope["type"] != "http":
                await self.app(scope, receive, send)
                return
            request = Request(scope)
            request_id = request.headers.get("x-request-id")
            size = 0
            chunks = []
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                size += len(chunk)
                if size > max_request_bytes:
                    await error_response(413, request_id, "inference_protocol_error", "Request exceeds byte limit")(
                        scope, receive, send
                    )
                    return
                chunks.append(chunk)
                if not message.get("more_body", False):
                    break
            replayed = False

            async def replay() -> dict:
                nonlocal replayed
                if replayed:
                    return cast(dict[str, Any], await receive())
                replayed = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}

            await self.app(scope, replay, send)

    app.add_middleware(BoundedIngress)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        detail: dict[str, Any] = exc.detail if isinstance(exc.detail, dict) else {}
        default_code = {
            429: "inference_busy",
            503: "inference_unavailable",
            504: "inference_timeout",
        }.get(exc.status_code, "inference_protocol_error")
        return error_response(
            exc.status_code,
            getattr(request.state, "request_id", request.headers.get("x-request-id")),
            detail.get("code", default_code),
            detail.get("message", "Inference request failed"),
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(
            422, request.headers.get("x-request-id"), "inference_protocol_error", "Invalid request schema"
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        return error_response(
            500,
            getattr(request.state, "request_id", request.headers.get("x-request-id")),
            "inference_unavailable",
            "Inference service failed",
        )


class InferenceAdmission:
    """One running model call with bounded queued requests and queue wait time.

    Cancelling a waiter cannot release a running model's slot: CUDA work is not
    preemptible, so its completion callback owns that release.
    """

    def __init__(self, *, max_queue: int = 8, queue_timeout_s: float = 30.0) -> None:
        if type(max_queue) is not int or max_queue < 0:
            raise ValueError("max_queue must be a nonnegative integer")
        require_positive_number(queue_timeout_s, message="queue_timeout_s must be finite and positive")
        self._slot = asyncio.Semaphore(1)
        self._pending = 0
        self.max_queue = max_queue
        self.queue_timeout_s = queue_timeout_s
        self._idle = asyncio.Event()
        self._idle.set()

    async def wait_for_idle(self, timeout_s: float = 5.0) -> bool:
        """Wait without cancelling model workers; False means ownership must be retained."""
        try:
            await asyncio.wait_for(self._idle.wait(), timeout_s)
        except TimeoutError:
            return False
        return True

    async def run(self, fn: Callable, *args: Any, request: Request | None = None, **kwargs: Any) -> Any:
        if self._pending >= self.max_queue + 1:
            raise HTTPException(429, detail={"code": "inference_busy", "message": "Inference queue is full"})
        self._pending += 1
        self._idle.clear()
        acquired = False
        transferred = False
        try:
            try:
                await asyncio.wait_for(self._slot.acquire(), self.queue_timeout_s)
                acquired = True
            except TimeoutError as exc:
                raise HTTPException(
                    429, detail={"code": "inference_busy", "message": "Inference queue wait expired"}
                ) from exc
            if request is not None and await request.is_disconnected():
                raise HTTPException(499, detail={"code": "inference_unavailable", "message": "Caller disconnected"})
            future = asyncio.get_running_loop().run_in_executor(None, functools.partial(fn, *args, **kwargs))

            def finished(done: asyncio.Future) -> None:
                self._slot.release()
                self._pending -= 1
                if self._pending == 0:
                    self._idle.set()
                if not done.cancelled():
                    done.exception()  # retrieve failures after a disconnected caller

            future.add_done_callback(finished)
            transferred = True
            return await asyncio.shield(future)
        finally:
            if not transferred:
                if acquired:
                    self._slot.release()
                self._pending -= 1
                if self._pending == 0:
                    self._idle.set()
