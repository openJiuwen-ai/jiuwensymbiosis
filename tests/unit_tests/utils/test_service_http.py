# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Public HTTP client contracts: deadlines, cleanup and bounds."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from jiuwensymbiosis.agent.cancel import CancelToken, RunCancelled
from jiuwensymbiosis.agent.fast.runner import _detect_once
from jiuwensymbiosis.errors import InferenceServiceError
from jiuwensymbiosis.utils.service_http import HttpEndpointConfig, HttpServiceClient


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/model",
        "http://user:secret@host",
        "https://host/?token=x",
        "http://host/#x",
        "http://host:99999",
        " http://host",
        "http://host:0",
    ],
)
def test_rejects_unsafe_or_invalid_urls(url):
    with pytest.raises(ValueError):
        HttpEndpointConfig(url)


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_timeout_s", float("inf")),
        ("connect_timeout_s", -1),
        ("max_response_bytes", True),
        ("max_request_bytes", 1.5),
    ],
)
def test_bounds_must_be_finite_positive(field, value):
    with pytest.raises(ValueError):
        HttpEndpointConfig.from_dict({"url": "http://service", field: value})


def test_strict_config():
    with pytest.raises(ValueError, match="unknown"):
        HttpEndpointConfig.from_dict({"url": "http://service", "requset_timeout": 5})


def test_open_is_network_free_and_requests_need_no_credentials():
    calls = []

    def handle(request):
        assert request.headers["x-request-id"] == "r"
        calls.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"request_id": "r", "result": []})

    cfg = HttpEndpointConfig("http://service/prefix")
    with HttpServiceClient(cfg, service="vision", transport=httpx.MockTransport(handle)) as client:
        assert not calls
        assert client.request_json("POST", "/v1/segment", {"request_id": "r"}, request_id="r")["result"] == []
        assert calls == [None]
    client.close()


def test_http_failure_never_exposes_response_body():
    with HttpServiceClient(
        HttpEndpointConfig("http://service"),
        service="speech",
        transport=httpx.MockTransport(lambda _: httpx.Response(401, text="private-secret")),
    ) as client:
        with pytest.raises(InferenceServiceError) as caught:
            client.request_json("GET", "/v1/health")
    assert caught.value.code == "inference_protocol_error"
    assert "private-secret" not in str(caught.value)


@pytest.mark.parametrize("malformed_code", [[], {}])
@pytest.mark.parametrize("status,code", [(503, "inference_unavailable"), (401, "inference_protocol_error")])
def test_malformed_error_code_is_service_failure_not_tracking_miss(malformed_code, status, code):
    with HttpServiceClient(
        HttpEndpointConfig("http://service"),
        service="vision",
        transport=httpx.MockTransport(lambda _: httpx.Response(status, json={"error": {"code": malformed_code}})),
    ) as client:
        api = SimpleNamespace(get_grasp_info_simple=lambda _: client.request_json("POST", "/v1/segment"))
        with pytest.raises(InferenceServiceError) as caught:
            _detect_once(api, "box")
        assert caught.value.code == code


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json=[]),
        httpx.Response(200, content=b'{"x":NaN}'),
        httpx.Response(200, json={"request_id": "wrong"}),
    ],
)
def test_invalid_success_is_protocol_failure(response):
    with HttpServiceClient(
        HttpEndpointConfig("http://service"), service="vision", transport=httpx.MockTransport(lambda _: response)
    ) as client:
        with pytest.raises(InferenceServiceError) as caught:
            client.request_json("POST", "/v1/segment", request_id="expected")
        assert caught.value.code == "inference_protocol_error"


def test_enforces_request_and_response_byte_limits():
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, content=b"x" * 200)

    with HttpServiceClient(
        HttpEndpointConfig("http://service", max_request_bytes=32, max_response_bytes=100),
        service="vision",
        transport=httpx.MockTransport(handle),
    ) as client:
        with pytest.raises(InferenceServiceError, match="request exceeds"):
            client.request_json("POST", "/test", {"data": "x" * 100})
        assert not calls
        with pytest.raises(InferenceServiceError, match="response exceeds"):
            client.request_json("GET", "/test")


def test_only_connection_failure_is_retried():
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            raise httpx.ConnectError("secret internal host", request=request)
        return httpx.Response(503, json={"error": {"code": "inference_busy", "retryable": True}})

    with HttpServiceClient(
        HttpEndpointConfig("http://service"), service="vision", transport=httpx.MockTransport(handle)
    ) as client:
        with pytest.raises(InferenceServiceError) as caught:
            client.request_json("POST", "/test", {"request_id": "same"})
        assert calls == [{"request_id": "same"}, {"request_id": "same"}]
        assert caught.value.code == "inference_busy"
        assert caught.value.retryable


def test_absolute_deadline_stops_a_slow_response():
    finished = threading.Event()

    async def handle(request):
        try:
            await asyncio.sleep(30)
        finally:
            finished.set()
        return httpx.Response(200, json={})

    start = time.monotonic()
    with HttpServiceClient(
        HttpEndpointConfig("http://service", request_timeout_s=0.1),
        service="vision",
        transport=httpx.MockTransport(handle),
    ) as client:
        with pytest.raises(InferenceServiceError) as caught:
            client.request_json("GET", "/test")
        assert caught.value.code == "inference_timeout"
    assert finished.wait(1)
    assert time.monotonic() - start < 2


def test_cancel_keeps_work_registered_until_request_cleanup_finishes():
    entered = threading.Event()
    token = CancelToken()
    errors = []

    async def handle(request):
        try:
            entered.set()
            await asyncio.sleep(30)
        finally:
            await asyncio.sleep(0.15)
        return httpx.Response(200, json={})

    client = HttpServiceClient(
        HttpEndpointConfig("http://service"),
        service="vision",
        cancel_token=token,
        transport=httpx.MockTransport(handle),
    )

    def request():
        try:
            client.request_json("GET", "/test")
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=request)
    try:
        thread.start()
        assert entered.wait(2)
        assert token.pending_work
        token.set()
        thread.join(1)
        assert isinstance(errors[0], RunCancelled)
        assert token.pending_work
        assert token.wait_for_idle(2)
    finally:
        client.close()
        thread.join(2)


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), SystemExit("exit"), RuntimeError("caller failed")])
def test_interrupted_caller_cancels_request_before_owner_cleanup(interruption):
    entered, cancelled = threading.Event(), threading.Event()
    release = asyncio.Event()

    class InterruptedToken(CancelToken):
        def raise_if_set(self):
            super().raise_if_set()
            if entered.is_set():
                raise interruption

    token = InterruptedToken()

    async def handle(request):
        entered.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
            raise

    client = HttpServiceClient(
        HttpEndpointConfig("http://service"),
        service="asr",
        cancel_token=token,
        transport=httpx.MockTransport(handle),
    )
    try:
        with pytest.raises(type(interruption)) as caught:
            client.request_json("POST", "/v1/transcribe")
        assert caught.value is interruption
        assert not token.is_set()
        # No stop()/close() yet: abandoning the caller must cancel the HTTP task.
        assert cancelled.wait(1)
        assert token.pending_work == ("http:asr",)
        client._loop.call_soon_threadsafe(release.set)
        assert token.wait_for_idle(1)
    finally:
        if client._loop is not None and not client._loop.is_closed():
            client._loop.call_soon_threadsafe(release.set)
        client.close()


def test_concurrent_calls_are_bounded_and_close_interrupts_request():
    entered = threading.Event()
    errors = []

    async def handle(request):
        entered.set()
        await asyncio.sleep(30)
        return httpx.Response(200, json={})

    client = HttpServiceClient(
        HttpEndpointConfig("http://service"), service="vision", transport=httpx.MockTransport(handle)
    )

    def request():
        try:
            client.request_json("GET", "/test")
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=request)
    thread.start()
    try:
        assert entered.wait(2)
        with pytest.raises(InferenceServiceError) as caught:
            client.request_json("GET", "/test")
        assert caught.value.code == "inference_busy"
        client.close()
        thread.join(2)
        assert not thread.is_alive()
        assert isinstance(errors[0], InferenceServiceError)
        assert client.closed
    finally:
        client.close()


@pytest.mark.asyncio
async def test_sync_client_can_be_called_from_an_existing_event_loop():
    with HttpServiceClient(
        HttpEndpointConfig("http://service"),
        service="vision",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"ok": True})),
    ) as client:
        assert client.request_json("GET", "/test") == {"ok": True}


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), SystemExit("exit"), RuntimeError("caller failed")])
def test_interrupted_executor_startup_releases_work_and_allows_reopen(monkeypatch, interruption):
    starting, release = threading.Event(), threading.Event()
    token = CancelToken()
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={"ok": True})

    client = HttpServiceClient(
        HttpEndpointConfig("http://service"),
        service="asr",
        cancel_token=token,
        transport=httpx.MockTransport(handle),
    )
    run_loop = client._run_loop

    def delayed_start():
        starting.set()
        assert release.wait(2)
        run_loop()

    def interrupted_wait(timeout):
        assert starting.wait(1)
        raise interruption

    try:
        with monkeypatch.context() as patch:
            patch.setattr(client, "_run_loop", delayed_start)
            patch.setattr(client._ready, "wait", interrupted_wait)
            with pytest.raises(type(interruption)) as caught:
                client.request_json("GET", "/v1/health")
        assert caught.value is interruption
        assert not token.is_set()
        assert token.wait_for_idle(0.1)
        assert calls == []
        http_thread = client._thread
        release.set()
        client.close()
        assert client.closed and not http_thread.is_alive()
        client.open()
        assert client.request_json("GET", "/v1/health") == {"ok": True}
        assert len(calls) == 1
    finally:
        release.set()
        client.close()


def test_cancel_before_queued_request_starts_never_sends_abandoned_request(monkeypatch):
    queued, release = threading.Event(), threading.Event()
    token = CancelToken()
    calls, errors = [], []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={"ok": True})

    client = HttpServiceClient(
        HttpEndpointConfig("http://service"),
        service="asr",
        cancel_token=token,
        transport=httpx.MockTransport(handle),
    )
    start = client._start

    def delayed_start(*args):
        queued.set()
        assert release.wait(2)
        start(*args)

    def request():
        try:
            client.request_json("GET", "/v1/health")
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(client, "_start", delayed_start)
    caller = threading.Thread(target=request)
    caller.start()
    try:
        assert queued.wait(1)
        token.set()
        caller.join(1)
        assert not caller.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], RunCancelled)
        assert token.wait_for_idle(0.1)
        release.set()
        client.close()
        assert calls == []
        client.bind_cancel_token(None)
        client.open()
        assert client.request_json("GET", "/v1/health") == {"ok": True}
        assert len(calls) == 1
    finally:
        release.set()
        client.close()
        caller.join(1)
