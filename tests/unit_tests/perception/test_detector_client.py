# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Real HTTP codec boundaries without any model, GPU or network dependency."""

import base64
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from jiuwensymbiosis.agent.cancel import CancelToken, RunCancelled
from jiuwensymbiosis.errors import InferenceServiceError
from jiuwensymbiosis.perception import detector_client
from jiuwensymbiosis.perception.config import DetectorConfig, parse_detector_config
from jiuwensymbiosis.perception.detector_client import (
    _encode_image,
    create_detector_client,
    decode_mask_png,
    encode_mask_png,
    init_detector,
)


def test_png_preserves_binary_mask():
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:7, 4:10] = True
    assert np.array_equal(decode_mask_png(encode_mask_png(mask), width=12, height=10), mask)


def test_empty_detection_and_service_failure_are_distinct():
    def response(method, path, payload, **kwargs):
        assert path == "/v1/segment"
        return {
            "schema_version": 1,
            "request_id": payload["request_id"],
            "result": {"frame_id": payload["frame_id"], "width": 10, "height": 10, "detections": []},
        }

    with init_detector() as client:
        client._http.request_json = Mock(side_effect=response)
        assert client(np.zeros((10, 10, 3), dtype=np.uint8), "box") == []
        client._http.request_json.side_effect = InferenceServiceError("unreachable")
        with pytest.raises(InferenceServiceError):
            client(np.zeros((10, 10, 3), dtype=np.uint8), "box")


@pytest.mark.parametrize("mode", ["init_detector", "local", "remote"])
def test_default_frame_age_does_not_shorten_request_timeout(monkeypatch, mode):
    if mode == "init_detector":
        client = init_detector(timeout_s=30)
    else:
        raw = {"mode": mode}
        if mode == "remote":
            raw["endpoint"] = {"url": "http://vision", "request_timeout_s": 30}
        client = create_detector_client(parse_detector_config({"detector": raw}))
    now = [100.0]
    monkeypatch.setattr(detector_client, "time", SimpleNamespace(monotonic=lambda: now[0]))

    def response(method, path, payload, **kwargs):
        assert kwargs["timeout_s"] == 30
        now[0] += 12  # Healthy slow inference, within the original request timeout.
        assert path == "/v1/segment"
        return {
            "schema_version": 1,
            "request_id": payload["request_id"],
            "result": {"frame_id": payload["frame_id"], "width": 10, "height": 10, "detections": []},
        }

    client._http.request_json = response
    try:
        for prompt in ("table", "box"):
            assert client(np.zeros((10, 10, 3), dtype=np.uint8), prompt, captured_monotonic_s=100) == []
    finally:
        client.close()


def test_explicit_frame_age_caps_request_and_rejects_late_result(monkeypatch):
    client = init_detector(timeout_s=30, max_frame_age_s=10)
    now = [100.0]
    monkeypatch.setattr(detector_client, "time", SimpleNamespace(monotonic=lambda: now[0]))

    def response(method, path, payload, **kwargs):
        assert kwargs["timeout_s"] == 10
        now[0] += 12
        return {
            "schema_version": 1,
            "request_id": payload["request_id"],
            "result": {"frame_id": payload["frame_id"], "width": 10, "height": 10, "detections": []},
        }

    client._http.request_json = response
    try:
        with pytest.raises(InferenceServiceError) as exc:
            client(np.zeros((10, 10, 3), dtype=np.uint8), "box")
        assert exc.value.code == "inference_result_stale"
    finally:
        client.close()


def test_disabled_frame_age_still_checks_cancellation_after_response():
    client = init_detector()
    token = CancelToken()
    client.bind_cancel_token(token)

    def response(method, path, payload, **kwargs):
        token.set()
        return {
            "schema_version": 1,
            "request_id": payload["request_id"],
            "result": {"frame_id": payload["frame_id"], "width": 10, "height": 10, "detections": []},
        }

    client._http.request_json = response
    try:
        with pytest.raises(RunCancelled):
            client(np.zeros((10, 10, 3), dtype=np.uint8), "box")
    finally:
        client.close()


def test_v1_matches_frame_and_returns_mask():
    client = create_detector_client(
        parse_detector_config({"detector": {"mode": "remote", "endpoint": {"url": "http://vision"}}})
    )
    mask = np.ones((10, 10), dtype=bool)

    def response(method, path, payload, **kwargs):
        assert path == "/v1/segment"
        return {
            "schema_version": 1,
            "request_id": payload["request_id"],
            "result": {
                "frame_id": payload["frame_id"],
                "width": 10,
                "height": 10,
                "detections": [{"mask": encode_mask_png(mask), "box": [0, 0, 10, 10], "score": 0.8, "label": "box"}],
            },
        }

    client._http.request_json = response
    assert np.array_equal(client(np.zeros((10, 10, 3), dtype=np.uint8), "box")[0]["mask"], mask)
    client.close()


@pytest.mark.parametrize("bad", ["frame", "mask", "box", "score", "box_overflow", "score_overflow", "request", "count"])
def test_v1_rejects_malformed_results(bad):
    client = create_detector_client(
        parse_detector_config({"detector": {"mode": "remote", "endpoint": {"url": "http://vision"}}})
    )

    def response(method, path, payload, **kwargs):
        item = {
            "mask": encode_mask_png(np.ones((10, 10), dtype=bool)),
            "box": [0, 0, 10, 10],
            "score": 0.9,
            "label": "box",
        }
        result = {"frame_id": payload["frame_id"], "width": 10, "height": 10, "detections": [item]}
        if bad == "frame":
            result["frame_id"] = "other"
        if bad == "mask":
            item["mask"] = encode_mask_png(np.ones((1, 1), dtype=bool))
        if bad == "box":
            item["box"][2] = 11
        if bad == "score":
            item["score"] = float("nan")
        if bad == "score_overflow":
            item["score"] = 10**400
        if bad == "box_overflow":
            item["box"][2] = 10**400
        if bad == "count":
            result["detections"] = [item] * 33
        return {
            "schema_version": 1,
            "request_id": "other" if bad == "request" else payload["request_id"],
            "result": result,
        }

    client._http.request_json = response
    with pytest.raises(InferenceServiceError) as exc:
        client(np.zeros((10, 10, 3), dtype=np.uint8), "box")
    assert exc.value.code == "inference_protocol_error"
    client.close()


def test_disabled_and_stale_do_not_issue_requests():
    with pytest.raises(InferenceServiceError):
        create_detector_client(DetectorConfig())(np.zeros((1, 1, 3), dtype=np.uint8), "box")
    client = init_detector(max_frame_age_s=10)
    client._http.request_json = Mock()
    with pytest.raises(InferenceServiceError) as exc:
        client(np.zeros((1, 1, 3), dtype=np.uint8), "box", captured_monotonic_s=time.monotonic() - 20)
    assert exc.value.code == "inference_result_stale"
    client._http.request_json.assert_not_called()
    client.close()


def test_encode_float_image():
    assert base64.b64decode(_encode_image(np.full((10, 10, 3), 128.0)))
