# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.perception.detector_client."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import numpy as np

from jiuwensymbiosis.perception.detector_client import _encode_image, init_detector


class TestInitDetector:
    def test_returns_callable(self):
        seg_fn = init_detector("http://127.0.0.1:8114")
        assert callable(seg_fn)

    def test_seg_fn_calls_http(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "mask_base64": base64.b64encode(np.zeros((10, 10), dtype=bool).astype(np.uint8).tobytes()).decode(),
                    "shape": [10, 10],
                    "box": [1.0, 2.0, 3.0, 4.0],
                    "score": 0.9,
                    "label": "box",
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("jiuwensymbiosis.perception.detector_client.requests.post", return_value=mock_response):
            seg_fn = init_detector("http://127.0.0.1:8114")
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            results = seg_fn(img, text_prompt="box")
            assert len(results) == 1
            assert results[0]["score"] == 0.9

    def test_seg_fn_empty_results(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        with patch("jiuwensymbiosis.perception.detector_client.requests.post", return_value=mock_response):
            seg_fn = init_detector("http://127.0.0.1:8114")
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            results = seg_fn(img, text_prompt="nothing")
            assert results == []

    def test_seg_fn_server_unreachable(self):
        with patch(
            "jiuwensymbiosis.perception.detector_client.requests.post",
            side_effect=Exception("connection refused"),
        ):
            seg_fn = init_detector("http://127.0.0.1:8114")
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            results = seg_fn(img, text_prompt="box")
            assert results == []


class TestEncodeImage:
    def test_ndarray_to_b64(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = _encode_image(img)
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_float_to_uint8(self):
        img = np.full((10, 10, 3), 128.0, dtype=np.float64)
        result = _encode_image(img)
        assert isinstance(result, str)
