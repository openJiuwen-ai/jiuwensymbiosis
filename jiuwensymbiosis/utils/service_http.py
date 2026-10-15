# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Bounded, cancellable HTTP for model services, with no model dependencies.

The synchronous interface owns one lazy I/O loop. Cancellation is delivered to
the actual async request; work stays registered until its coroutine has exited.
The module does not know perception, audio, robot actions, or their schemas.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Protocol
from urllib.parse import urlsplit

from jiuwensymbiosis.errors import INFERENCE_ERROR_CODES, InferenceServiceError
from jiuwensymbiosis.utils.validation import require_positive_number

__all__ = ["HttpEndpointConfig", "HttpServiceClient"]


def _positive(value: Any, name: str) -> None:
    require_positive_number(value, message=f"{name} must be a finite positive number")


@dataclass(frozen=True)
class HttpEndpointConfig:
    """Serializable HTTP endpoint settings."""

    url: str
    connect_timeout_s: float = 3.0
    request_timeout_s: float = 30.0
    max_request_bytes: int = 16 * 1024 * 1024
    max_response_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        if not isinstance(self.url, str):
            raise ValueError("endpoint.url must be an HTTP(S) URL")
        try:
            parsed = urlsplit(self.url)
            port = parsed.port
            valid = (
                parsed.scheme in {"http", "https"}
                and bool(parsed.hostname)
                and parsed.username is None
                and parsed.password is None
                and not parsed.query
                and not parsed.fragment
                and not any(c.isspace() or ord(c) < 32 for c in self.url)
                and (port is None or port > 0)
            )
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("endpoint.url must be HTTP(S), without credentials, query parameters or fragments")
        for name in ("connect_timeout_s", "request_timeout_s", "max_request_bytes", "max_response_bytes"):
            _positive(getattr(self, name), f"endpoint.{name}")
        for name in ("max_request_bytes", "max_response_bytes"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"endpoint.{name} must be an integer")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HttpEndpointConfig:
        if not isinstance(data, Mapping):
            raise ValueError("endpoint must be a mapping")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown endpoint fields: {sorted(unknown)}")
        if "url" not in data:
            raise ValueError("endpoint.url is required")
        return cls(**dict(data))


class _CancellationContext(Protocol):
    """Local task cancellation and work tracking, unrelated to HTTP authentication.

    This protocol keeps the transport independent of the owner's implementation.
    """

    # fmt: off
    def raise_if_set(self) -> None:
        ...

    def register_work(self, label: str) -> Callable[[], None]:
        ...

    def on_cancel(self, closer: Callable[[], None]) -> Callable[[], None]:
        ...
    # fmt: on


def _no_work() -> None:
    return None


@dataclass(eq=False)
class _Request:
    method: str
    path: str
    body: bytes | None
    request_id: str | None
    deadline: float
    done: threading.Event = field(default_factory=threading.Event)
    task: asyncio.Task | None = None
    result: dict[str, Any] | None = None
    error: BaseException | None = None
    finish_work: Callable[[], None] = _no_work


class HttpServiceClient:
    """One managed connection pool, with one in-flight request and no backlog.

    ``open`` prepares request headers (or defers them with ``lazy=True``) without I/O.
    Concurrent requests fail busy rather than build an unbounded queue. ``close`` cancels and joins the
    I/O thread; a timed-out close retains its handles so the owner can retry.
    """

    def __init__(
        self,
        endpoint: HttpEndpointConfig,
        *,
        service: str,
        cancel_token: _CancellationContext | None = None,
        transport: Any = None,
    ) -> None:
        self.endpoint = endpoint
        self.service = service
        self._cancel_token = cancel_token
        self._transport = transport
        self._lock = threading.RLock()
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client: Any = None
        self._job: _Request | None = None
        self._opened = False
        self._closed = False
        self._closing = False
        self._headers: dict[str, str] = {}

    @property
    def closed(self) -> bool:
        return self._closed

    def bind_cancel_token(self, token: _CancellationContext | None) -> None:
        with self._lock:
            if self._job is not None:
                raise RuntimeError("cannot change cancellation owner during an HTTP request")
            self._cancel_token = token

    def open(self, *, lazy: bool = False) -> HttpServiceClient:
        """Open a session without I/O; ``lazy`` defers setup to its first request."""
        with self._lock:
            if self._closing:
                raise RuntimeError("HTTP client cleanup is incomplete")
            if lazy:
                self._closed = False
                return self
            if self._opened:
                return self
            headers = {"Accept": "application/json", "Accept-Encoding": "identity", "Content-Type": "application/json"}
            self._headers = headers
            self._opened = True
            self._closed = False
        return self

    def __enter__(self) -> HttpServiceClient:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _error(
        self, message: str, code: str = "inference_unavailable", request_id: str | None = None
    ) -> InferenceServiceError:
        return InferenceServiceError(message, code=code, service=self.service, request_id=request_id)

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    def _cancel(self, job: _Request) -> None:
        with self._lock:
            if job.done.is_set():
                return
            # A request abandoned before task creation has no coroutine to
            # finish its work registration. Serialize with _start so a queued
            # callback cannot start I/O after we declare that work complete.
            if job.task is None:
                job.error = self._error("HTTP inference request cancelled")
                self._complete(job)
                return
            loop = self._loop
        if loop is not None and not loop.is_closed():

            def cancel_task() -> None:
                if job.task is not None and not job.task.done() and not job.task.cancelling():
                    job.task.cancel()

            loop.call_soon_threadsafe(cancel_task)

    def _complete(self, job: _Request, task: asyncio.Task | None = None) -> None:
        if task is not None:
            try:
                job.result = task.result()
            except asyncio.CancelledError:
                job.error = self._error("HTTP inference request cancelled")
            except Exception as exc:
                job.error = exc
        job.finish_work()
        with self._lock:
            if self._job is job:
                self._job = None
        job.done.set()

    def _start(self, job: _Request) -> None:
        with self._lock:
            if job.done.is_set():
                return
            job.task = asyncio.create_task(self._request(job))
            job.task.add_done_callback(lambda task: self._complete(job, task))

    def request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        request_id: str | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Request bounded JSON, preserving cancellation and an absolute deadline."""
        budget = (
            self.endpoint.request_timeout_s if timeout_s is None else min(timeout_s, self.endpoint.request_timeout_s)
        )
        _positive(budget, "request timeout")
        deadline = time.monotonic() + budget
        if not isinstance(path, str) or not path.startswith("/") or path.startswith("//"):
            raise ValueError("service path must be an absolute relative route")
        if "?" in path or "#" in path:
            raise ValueError("service path must be an absolute relative route")
        token = self._cancel_token
        if token is not None:
            token.raise_if_set()
        try:
            body = (
                None if payload is None else json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
            )
        except (ValueError, TypeError):
            raise self._error("invalid JSON request", "inference_protocol_error", request_id) from None
        if body is not None and len(body) > self.endpoint.max_request_bytes:
            raise self._error("HTTP request exceeds byte limit", "inference_protocol_error", request_id)
        with self._lock:
            if self._closed or self._closing:
                raise self._error("HTTP client is closed", request_id=request_id)
            self.open()
            if self._job is not None:
                raise self._error("HTTP client already has an in-flight request", "inference_busy", request_id)
            job = _Request(method=method, path=path, body=body, request_id=request_id, deadline=deadline)
            if token is not None:
                job.finish_work = token.register_work(f"http:{self.service}")
            self._job = job
            if self._thread is None:
                self._ready.clear()
                self._thread = threading.Thread(target=self._run_loop, name=f"jws-http-{self.service}", daemon=True)
                try:
                    self._thread.start()
                except Exception:
                    self._thread = None
                    self._complete(job)
                    raise
        unregister = token.on_cancel(lambda: self._cancel(job)) if token is not None else _no_work
        try:
            if not self._ready.wait(max(0, deadline - time.monotonic())):
                raise self._error("HTTP executor startup timed out", "inference_timeout", request_id)
            with self._lock:
                loop = self._loop
                if self._closing or self._closed or loop is None:
                    self._complete(job)
                    raise self._error("HTTP executor unavailable", request_id=request_id)
                if loop.is_closed():
                    self._complete(job)
                    raise self._error("HTTP executor unavailable", request_id=request_id)
                loop.call_soon_threadsafe(self._start, job)
            while not job.done.wait(min(0.05, max(0, deadline - time.monotonic()))):
                if token is not None:
                    token.raise_if_set()
                if time.monotonic() >= deadline:
                    self._cancel(job)
                    raise self._error("HTTP inference deadline exceeded", "inference_timeout", request_id)
            if token is not None:
                token.raise_if_set()
            if job.error is not None:
                raise job.error
            if time.monotonic() >= deadline:
                raise self._error("HTTP inference deadline exceeded", "inference_timeout", request_id)
            if job.result is None:
                raise self._error("missing HTTP result", "inference_protocol_error", request_id)
            return job.result
        finally:
            try:
                # Ctrl-C can unwind the caller before its owner sets the token.
                # Cancel abandoned I/O before removing that cancellation route.
                if not job.done.is_set():
                    self._cancel(job)
            finally:
                unregister()

    async def _request(self, job: _Request) -> dict[str, Any]:
        try:
            import httpx
        except ImportError:
            raise ImportError('HTTP clients require pip install -e ".[remote]"') from None

        if self._client is None:
            self._client = httpx.AsyncClient(
                trust_env=False,
                follow_redirects=False,
                transport=self._transport,
                headers=self._headers,
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
            )
        try:
            async with asyncio.timeout(max(0, job.deadline - time.monotonic())):
                for attempt in range(2):
                    try:
                        timeout = httpx.Timeout(
                            max(0.001, job.deadline - time.monotonic()),
                            connect=min(self.endpoint.connect_timeout_s, max(0.001, job.deadline - time.monotonic())),
                        )
                        async with self._client.stream(
                            job.method,
                            self.endpoint.url.rstrip("/") + job.path,
                            content=job.body,
                            timeout=timeout,
                            headers={"X-Request-ID": job.request_id} if job.request_id is not None else None,
                        ) as response:
                            if response.headers.get("content-encoding", "identity").lower() != "identity":
                                raise self._error(
                                    "compressed HTTP responses are not supported",
                                    "inference_protocol_error",
                                    job.request_id,
                                )
                            raw = bytearray()
                            async for chunk in response.aiter_bytes():
                                if len(raw) + len(chunk) > self.endpoint.max_response_bytes:
                                    raise self._error(
                                        "HTTP response exceeds byte limit", "inference_protocol_error", job.request_id
                                    )
                                raw.extend(chunk)
                            if response.status_code != 200:
                                self._raise_status(response.status_code, raw, job.request_id)
                            try:
                                data = json.loads(raw, parse_constant=_reject_constant)
                            except (ValueError, RecursionError):
                                raise self._error(
                                    "invalid service JSON", "inference_protocol_error", job.request_id
                                ) from None
                            if not isinstance(data, dict):
                                raise self._error(
                                    "service JSON must be an object", "inference_protocol_error", job.request_id
                                )
                            if job.request_id is not None and data.get("request_id") != job.request_id:
                                raise self._error("mismatched request_id", "inference_protocol_error", job.request_id)
                            return data
                    except (httpx.ConnectError, httpx.ConnectTimeout):
                        if attempt:
                            raise
                raise self._error("HTTP connection failed", request_id=job.request_id)
        except (TimeoutError, httpx.TimeoutException):
            raise self._error("HTTP inference deadline exceeded", "inference_timeout", job.request_id) from None
        except httpx.HTTPError:
            raise self._error("HTTP inference connection failed", request_id=job.request_id) from None

    def _raise_status(self, status: int, raw: bytes | bytearray, request_id: str | None) -> None:
        code = {
            408: "inference_timeout",
            429: "inference_busy",
            502: "inference_unavailable",
            503: "inference_unavailable",
            504: "inference_timeout",
        }.get(status, "inference_protocol_error" if status < 500 else "inference_unavailable")
        retryable = False
        try:
            data = json.loads(raw)
            error = data.get("error") if isinstance(data, dict) else None
            if isinstance(error, dict):
                error_code = error.get("code")
                if isinstance(error_code, str) and error_code in INFERENCE_ERROR_CODES:
                    code = error_code
                retryable = error.get("retryable") is True and status in {429, 503}
        except (ValueError, RecursionError):
            pass
        raise InferenceServiceError(
            f"{self.service} service returned HTTP {status}",
            code=code,
            service=self.service,
            request_id=request_id,
            retryable=retryable,
        )

    async def _shutdown(self) -> None:
        job = self._job
        if job is not None and job.task is not None:
            if not job.task.cancelling():
                job.task.cancel()
            await asyncio.gather(job.task, return_exceptions=True)
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def close(self, timeout_s: float = 5.0) -> None:
        """Cancel in-flight I/O and wait boundedly; retain handles on failure."""
        _positive(timeout_s, "close timeout")
        deadline = time.monotonic() + timeout_s
        with self._lock:
            if self._closed:
                return
            self._closing = True
            job, thread = self._job, self._thread
        if job is not None:
            self._cancel(job)
        if thread is not None:
            if not self._ready.wait(max(0, deadline - time.monotonic())):
                raise RuntimeError("HTTP executor startup has not finished during cleanup")
            loop = self._loop
            if loop is not None and not loop.is_closed() and thread.is_alive():
                future = asyncio.run_coroutine_threadsafe(self._shutdown(), loop)
                try:
                    future.result(timeout=max(0, deadline - time.monotonic()))
                except TimeoutError:
                    raise RuntimeError("HTTP request cleanup is incomplete") from None
                loop.call_soon_threadsafe(loop.stop)
            thread.join(max(0, deadline - time.monotonic()))
            if thread.is_alive():
                raise RuntimeError("HTTP executor cleanup is incomplete")
        with self._lock:
            self._thread = None
            self._loop = None
            self._headers = {}
            self._opened = False
            self._closing = False
            self._closed = True


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON number")
