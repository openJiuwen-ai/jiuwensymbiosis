# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""imaging:ndarray 相机帧 → JPEG 字节 / data URI(纯逻辑,Pillow 已在 [dev])。"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from jiuwensymbiosis.gui import imaging


def test_jpeg_bytes_have_jpeg_magic():
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    data = imaging.ndarray_to_jpeg_bytes(frame)
    assert data[:2] == b"\xff\xd8"  # JPEG SOI marker


def test_data_uri_prefix_and_roundtrip_size():
    frame = np.zeros((8, 5, 3), dtype=np.uint8)
    uri = imaging.to_data_uri(frame)
    assert uri.startswith("data:image/jpeg;base64,")
    decoded = base64.b64decode(uri.split(",", 1)[1])
    img = Image.open(io.BytesIO(decoded))
    assert img.size == (5, 8)  # PIL size is (width, height)


def test_grayscale_frame_supported():
    frame = np.zeros((4, 4), dtype=np.uint8)
    uri = imaging.to_data_uri(frame)
    assert uri.startswith("data:image/jpeg;base64,")


def test_rgba_frame_flattened_to_rgb():
    frame = np.zeros((4, 4, 4), dtype=np.uint8)
    assert imaging.ndarray_to_jpeg_bytes(frame)[:2] == b"\xff\xd8"


def test_non_uint8_frame_is_clipped():
    frame = np.full((3, 3, 3), 300, dtype=np.int32)
    assert imaging.ndarray_to_jpeg_bytes(frame)[:2] == b"\xff\xd8"


def test_unsupported_shape_raises():
    with pytest.raises(ValueError, match="unsupported frame shape"):
        imaging.ndarray_to_jpeg_bytes(np.zeros((2, 2, 5), dtype=np.uint8))
