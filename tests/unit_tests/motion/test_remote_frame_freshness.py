# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Remote coarse search consumes a single capture budget through final publication."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from jiuwensymbiosis.agent.cancel import CancelToken, RunCancelled
from jiuwensymbiosis.errors import InferenceServiceError
from jiuwensymbiosis.motion import approach
from jiuwensymbiosis.perception import detector_client
from jiuwensymbiosis.perception.config import parse_detector_config
from jiuwensymbiosis.perception.frame import CameraFrame


@pytest.fixture
def client():
    detector = detector_client.create_detector_client(
        parse_detector_config(
            {"detector": {"mode": "remote", "endpoint": {"url": "http://vision"}, "max_frame_age_s": 10}}
        )
    )
    try:
        yield detector
    finally:
        detector.close()


def _api(client, *, legacy=False, captured=0.0):
    frame = CameraFrame(np.zeros((10, 10, 3), dtype=np.uint8), captured_monotonic_s=captured, frame_id="camera-frame")
    api = SimpleNamespace(_seg_fn=client, env=SimpleNamespace(grab_calibrated_frame=lambda camera: frame))
    if legacy:
        api.search_frames = lambda camera: (frame.rgb, None)
    return api


def _response(payload):
    return {
        "schema_version": 1,
        "request_id": payload["request_id"],
        "result": {
            "frame_id": payload["frame_id"],
            "width": 10,
            "height": 10,
            "detections": [
                {
                    "mask": detector_client.encode_mask_png(np.ones((10, 10), dtype=bool)),
                    "box": [0, 0, 10, 10],
                    "score": 0.9,
                    "label": "box",
                }
            ],
        },
    }


@pytest.mark.parametrize("legacy", [False, True])
def test_two_prompts_share_capture_id_and_remaining_deadline(monkeypatch, client, legacy):
    now = [0.0]
    monkeypatch.setattr(detector_client.time, "monotonic", lambda: now[0])
    requests = []

    def response(method, path, payload, **kwargs):
        requests.append((payload["frame_id"], kwargs["timeout_s"]))
        now[0] += 9.0
        return _response(payload)

    client._http.request_json = response
    with pytest.raises(InferenceServiceError) as exc:
        approach.look_once(_api(client, legacy=legacy), "box", on="table")
    assert exc.value.code == "inference_result_stale"
    assert len(requests) == 2
    assert requests[0][0] == requests[1][0]
    assert [timeout for _, timeout in requests] == [10.0, 1.0]


def test_cached_frame_age_is_checked_before_any_request(monkeypatch, client):
    monkeypatch.setattr(detector_client.time, "monotonic", lambda: 11.0)
    client._http.request_json = Mock()
    with pytest.raises(InferenceServiceError) as exc:
        approach.look_once(_api(client), "box")
    assert exc.value.code == "inference_result_stale"
    client._http.request_json.assert_not_called()


def test_legacy_capture_duration_consumes_budget(monkeypatch, client):
    now = [0.0]
    monkeypatch.setattr(detector_client.time, "monotonic", lambda: now[0])
    api = _api(client, legacy=True)

    def grab(camera):
        now[0] = 11.0
        return np.zeros((10, 10, 3), dtype=np.uint8), None

    api.search_frames = grab
    client._http.request_json = Mock()
    with pytest.raises(InferenceServiceError) as exc:
        approach.look_once(api, "box")
    assert exc.value.code == "inference_result_stale"
    client._http.request_json.assert_not_called()


@pytest.mark.parametrize("cancel", [False, True])
def test_late_local_bearing_cannot_be_published(monkeypatch, client, cancel):
    now = [0.0]
    monkeypatch.setattr(detector_client.time, "monotonic", lambda: now[0])
    token = CancelToken()
    client.bind_cancel_token(token)
    client._http.request_json = lambda method, path, payload, **kwargs: _response(payload)
    api = _api(client)

    def after_detection(*args):
        if cancel:
            token.set()
        else:
            now[0] = 11.0

    api.viz_update = after_detection
    with pytest.raises(RunCancelled if cancel else InferenceServiceError):
        approach.look_once(api, "box")
