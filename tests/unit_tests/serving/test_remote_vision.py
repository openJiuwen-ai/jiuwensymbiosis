# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Client -> HTTP application contract using a deterministic model stub."""

import asyncio
import threading

import httpx
import numpy as np
import pytest

from jiuwensymbiosis.errors import InferenceServiceError
from jiuwensymbiosis.perception.config import DetectorServerConfig, parse_detector_config
from jiuwensymbiosis.perception.detector_client import create_detector_client, encode_mask_png, init_detector
from jiuwensymbiosis.serving import grounding_dino_sam2_server as server
from jiuwensymbiosis.serving.http_support import InferenceAdmission
from jiuwensymbiosis.utils.service_http import HttpServiceClient


@pytest.mark.parametrize(
    "entry", ["local", "remote", "old_local_yaml", "old_remote_yaml", "old_python", "init_detector"]
)
def test_all_client_entries_use_v1_without_protocol_configuration(monkeypatch, entry):
    mask = np.zeros((12, 16), dtype=bool)
    mask[2:8, 3:11] = True
    monkeypatch.setattr(server, "_GDINO_MODEL", object())
    monkeypatch.setattr(server, "_USE_SAM2", False)
    monkeypatch.setattr(server, "_ADMISSION", InferenceAdmission())
    monkeypatch.setattr(
        server,
        "_do_segment",
        lambda image, prompt: [server.MaskData(mask=encode_mask_png(mask), box=[3, 2, 11, 8], score=0.9, label=prompt)],
    )
    if entry == "init_detector":
        client = init_detector("http://test")
    elif entry == "old_python":
        client = create_detector_client(DetectorServerConfig(url="http://test", spawn=False))
    else:
        if entry.startswith("old_"):
            raw = {"api_servers": [{"_target_": "gdino", "spawn": entry == "old_local_yaml"}]}
        else:
            detector = {"mode": entry}
            if entry == "remote":
                detector["endpoint"] = {"url": "http://test"}
            raw = {"detector": detector}
        client = create_detector_client(parse_detector_config(raw))
    paths = []
    transport = httpx.ASGITransport(app=server.app)

    async def handle(request):
        paths.append(request.url.path)
        return await transport.handle_async_request(request)

    client._http = HttpServiceClient(client.endpoint, service="vision", transport=httpx.MockTransport(handle))
    with client:
        assert client.readiness()["status"] == "ready"
        results = client(np.zeros((12, 16, 3), dtype=np.uint8), "box")
        assert np.array_equal(results[0]["mask"], mask)
    assert paths == ["/v1/health", "/v1/segment"]


def test_model_loading_is_not_empty_detections(monkeypatch):
    monkeypatch.setattr(server, "_GDINO_MODEL", None)
    client = create_detector_client(
        parse_detector_config({"detector": {"mode": "remote", "endpoint": {"url": "http://test"}}})
    )
    client._http = HttpServiceClient(client.endpoint, service="vision", transport=httpx.ASGITransport(app=server.app))
    with client, pytest.raises(InferenceServiceError) as exc:
        client(np.zeros((12, 16, 3), dtype=np.uint8), "box")
    assert exc.value.code == "inference_unavailable"


async def test_cancelled_request_keeps_model_slot_until_worker_finishes():
    entered, release = threading.Event(), threading.Event()
    admission = InferenceAdmission(max_queue=0)

    def model():
        entered.set()
        release.wait(5)

    task = asyncio.create_task(admission.run(model))
    await asyncio.to_thread(entered.wait, 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not await admission.wait_for_idle(0.02)
    from fastapi import HTTPException

    try:
        with pytest.raises(HTTPException) as exc:
            await admission.run(lambda: None)
        assert exc.value.status_code == 429
    finally:
        release.set()
    assert await admission.wait_for_idle(2)


@pytest.mark.parametrize(
    "config",
    [
        {"max_queue": -1},
        {"max_queue": True},
        {"queue_timeout_s": float("inf")},
        {"queue_timeout_s": 0},
        {"queue_timeout_s": True},
    ],
)
def test_admission_rejects_invalid_limits(config):
    with pytest.raises(ValueError):
        InferenceAdmission(**config)


async def test_disconnected_queued_request_does_not_start_a_model():
    from types import SimpleNamespace

    from fastapi import HTTPException

    async def disconnected():
        return True

    calls = []
    admission = InferenceAdmission()
    with pytest.raises(HTTPException) as exc:
        await admission.run(lambda: calls.append("model"), request=SimpleNamespace(is_disconnected=disconnected))
    assert exc.value.status_code == 499
    assert not calls
    assert await admission.wait_for_idle(0.1)


@pytest.mark.parametrize("use_sam2", [False, True])
def test_edge_boxes_are_clipped_before_segmentation_and_client_validation(monkeypatch, use_sam2):
    monkeypatch.setattr(server, "_GDINO_MODEL", object())
    monkeypatch.setattr(server, "_SAM2_MODEL", object())
    monkeypatch.setattr(server, "_USE_SAM2", use_sam2)
    monkeypatch.setattr(server, "_ADMISSION", InferenceAdmission())
    # cxcywh postprocessing can put corners outside the image; wholly exterior or
    # collapsed boxes must be discarded before they reach SAM2 or the HTTP response.
    monkeypatch.setattr(
        server,
        "_gdino_detect",
        lambda image, prompt: (
            np.array(
                [
                    [-1, 2, 2, 8],
                    [8, -2, 11, 12],
                    [-3, 2, -1, 8],
                    [5, 2, 5, 8],
                    [-np.inf, 2, 5, 8],
                    [2, 2, np.nan, 8],
                ],
                dtype=float,
            ),
            np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
        ),
    )
    sam_boxes = []

    def segment_masks(image, boxes):
        sam_boxes.extend(boxes.tolist())
        return [server._box_to_mask(box, image.height, image.width) for box in boxes]

    monkeypatch.setattr(server, "_sam2_masks", segment_masks)
    client = create_detector_client(
        parse_detector_config({"detector": {"mode": "remote", "endpoint": {"url": "http://test"}}})
    )
    client._http = HttpServiceClient(client.endpoint, service="vision", transport=httpx.ASGITransport(app=server.app))
    with client:
        results = client(np.zeros((10, 10, 3), dtype=np.uint8), "box")
    expected = [[0, 2, 2, 8], [8, 0, 10, 10]]
    assert [result["box"] for result in results] == expected
    assert all(result["mask"].any() and result["mask"].shape == (10, 10) for result in results)
    assert sam_boxes == (expected if use_sam2 else [])
