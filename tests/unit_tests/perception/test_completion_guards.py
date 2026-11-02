# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Cancellation and frame budgets also cover local encoding/decoding/geometry."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from jiuwensymbiosis.agent.cancel import CancelToken, RunCancelled
from jiuwensymbiosis.errors import InferenceServiceError
from jiuwensymbiosis.perception import detector_client, scene3d, vision
from jiuwensymbiosis.perception.config import parse_detector_config
from jiuwensymbiosis.perception.frame import CameraFrame


def _client():
    return detector_client.create_detector_client(
        parse_detector_config(
            {"detector": {"mode": "remote", "endpoint": {"url": "http://vision"}, "max_frame_age_s": 10}}
        )
    )


def test_encoding_consumes_frame_budget_before_network_dispatch(monkeypatch):
    client = _client()
    now = [10.0]
    monkeypatch.setattr(detector_client.time, "monotonic", lambda: now[0])

    def encode(image):
        now[0] = 21.0
        return "encoded"

    monkeypatch.setattr(detector_client, "_encode_image", encode)
    client._http.request_json = Mock()
    with pytest.raises(InferenceServiceError) as exc:
        client(np.zeros((10, 10, 3), dtype=np.uint8), "box")
    assert exc.value.code == "inference_result_stale"
    client._http.request_json.assert_not_called()
    client.close()


def test_cancellation_during_mask_decode_rejects_late_result(monkeypatch):
    client = _client()
    token = CancelToken()
    client.bind_cancel_token(token)
    encoded = detector_client.encode_mask_png(np.ones((10, 10), dtype=bool))

    def response(method, path, payload, **kwargs):
        return {
            "schema_version": 1,
            "request_id": payload["request_id"],
            "result": {
                "frame_id": payload["frame_id"],
                "width": 10,
                "height": 10,
                "detections": [{"mask": encoded, "box": [0, 0, 10, 10], "score": 0.9, "label": "box"}],
            },
        }

    client._http.request_json = response
    decode = detector_client.decode_mask_png

    def cancel_after_decode(*args, **kwargs):
        mask = decode(*args, **kwargs)
        token.set()
        return mask

    monkeypatch.setattr(detector_client, "decode_mask_png", cancel_after_decode)
    with pytest.raises(RunCancelled):
        client(np.zeros((10, 10, 3), dtype=np.uint8), "box")
    client.close()


def test_scene_geometry_cancel_clears_staged_caches(monkeypatch):
    client = _client()
    token = CancelToken()
    client.bind_cancel_token(token)
    api = SimpleNamespace(_seg_fn=client, last_detection=None, last_surface=None)
    frame = CameraFrame(np.zeros((10, 10, 3), dtype=np.uint8), np.ones((10, 10)), np.eye(3), np.eye(4))
    monkeypatch.setattr(scene3d, "_calibrated_frame_or_reason", lambda api, name: (frame, None))

    def geometry(*args, **kwargs):
        api.last_surface = {"ok": True, "old": True}
        token.set()
        return {"ok": True, "center_mm": [1, 2, 3]}

    monkeypatch.setattr(scene3d, "detect_object_geometry", geometry)
    with pytest.raises(RunCancelled):
        scene3d.locate_for_grasp(api, "box")
    assert api.last_detection is None
    assert api.last_surface is None
    client.close()


def test_default_grasp_cancel_during_projection_cannot_publish_position(monkeypatch):
    client = _client()
    token = CancelToken()
    client.bind_cancel_token(token)
    ll = SimpleNamespace(
        grab_frames=lambda: (np.zeros((10, 10, 3), dtype=np.uint8), np.ones((10, 10))),
        tf_flange_cam=np.eye(4),
        calibration={"intrinsics": np.eye(3)},
    )
    api = SimpleNamespace(
        env=SimpleNamespace(low_level=ll, z_min_safe=0, get_flange_pose=lambda: None), invalidate_sensing_cache=Mock()
    )
    monkeypatch.setattr(
        vision,
        "detect_and_centroid",
        lambda **kwargs: {"ok": True, "u": 1, "v": 1, "depth_m": 0.5, "best": {"score": 0.9}},
    )

    def correction(xyz, **kwargs):
        token.set()
        return xyz, "none"

    monkeypatch.setattr(vision, "apply_xy_correction", correction)
    with pytest.raises(RunCancelled):
        vision.default_get_grasp_info_simple(api, "box", seg_fn=client, pose_to_tf=lambda pose: np.eye(4))
    api.invalidate_sensing_cache.assert_called_once()
    client.close()


@pytest.mark.parametrize(
    ("action", "helper", "reference"),
    [
        ("locate_for_grasp", "detect_object_geometry", None),
        ("locate_for_grasp", "_detect_related", "table"),
        ("locate_for_place", "_sense_surface_plain", None),
        ("locate_for_place", "_sense_surface_related", "cup"),
        ("analyze_scene", "detect_all_object_geometry", None),
    ],
)
def test_scene_postprocessing_uses_capture_age_before_publishing(monkeypatch, action, helper, reference):
    client = _client()
    now = [9.5]
    monkeypatch.setattr(detector_client.time, "monotonic", lambda: now[0])
    frame = CameraFrame(
        np.zeros((10, 10, 3), dtype=np.uint8), np.ones((10, 10)), np.eye(3), np.eye(4), captured_monotonic_s=0.0
    )
    api = SimpleNamespace(
        _seg_fn=client,
        last_detection={"old": True},
        last_surface={"old": True},
        env=SimpleNamespace(grab_calibrated_frame=lambda camera: frame),
    )

    def geometry(*args, **kwargs):
        now[0] = 11.0
        return [] if action == "analyze_scene" else {"ok": True, "center_mm": [1, 2, 3]}

    monkeypatch.setattr(vision if action == "analyze_scene" else scene3d, helper, geometry)
    kwargs = {"reference": reference} if reference else {}
    try:
        result = getattr(scene3d, action)(api, "box", **kwargs)
        assert result["ok"] is False
        assert result["error_code"] == "inference_result_stale"
        assert api.last_detection is None
        assert api.last_surface is None
    finally:
        client.close()
