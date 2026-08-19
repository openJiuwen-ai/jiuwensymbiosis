# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for cruzr camera image decoding (pure)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.ros2.image_decode import decode_image_msg


def _msg(height, width, encoding, step, data):
    return SimpleNamespace(height=height, width=width, encoding=encoding, step=step, data=data)


def test_rgb8_roundtrip():
    h, w = 2, 3
    rgb = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
    msg = _msg(h, w, "rgb8", w * 3, rgb.tobytes())
    out = decode_image_msg(msg)
    assert out.shape == (h, w, 3)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, rgb)


def test_bgr8_is_flipped_to_rgb():
    h, w = 1, 2
    bgr = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)  # B,G,R
    msg = _msg(h, w, "bgr8", w * 3, bgr.tobytes())
    out = decode_image_msg(msg)
    expected = bgr[:, :, ::-1]
    np.testing.assert_array_equal(out, expected)


def test_depth_16uc1_scaled_to_meters():
    h, w = 1, 3
    raw = np.array([[1000, 2000, 0]], dtype=np.uint16)  # millimetres
    msg = _msg(h, w, "16UC1", w * 2, raw.tobytes())
    out = decode_image_msg(msg, depth_scale=0.001)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, np.array([[1.0, 2.0, 0.0]], dtype=np.float32))


def test_fixed_buffer_is_truncated_to_h_times_step():
    h, w = 1, 2
    rgb = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    padded = rgb.tobytes() + b"\xff" * 100  # shm fixed-size buffer is longer than valid bytes
    msg = _msg(h, w, "rgb8", w * 3, padded)
    out = decode_image_msg(msg)
    np.testing.assert_array_equal(out, rgb)


def test_step_padding_per_row_is_dropped():
    # step bigger than width*channels (row padding) — only first width*3 bytes per row are pixels.
    h, w = 2, 2
    row0 = bytes([1, 2, 3, 4, 5, 6, 0, 0])  # 2 px rgb + 2 pad bytes
    row1 = bytes([7, 8, 9, 10, 11, 12, 0, 0])
    msg = _msg(h, w, "rgb8", 8, row0 + row1)
    out = decode_image_msg(msg)
    expected = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)
    np.testing.assert_array_equal(out, expected)


def test_unsupported_encoding_raises():
    msg = _msg(1, 1, "rgba8", 4, b"\x00\x00\x00\x00")
    with pytest.raises(ValueError):
        decode_image_msg(msg)


def test_yuv422_uyvy_gray_to_rgb():
    # UYVY = [U, Y0, V, Y1] per 2 px. U=V=128 (no chroma), Y=100 -> RGB ~ (100,100,100).
    h, w = 2, 4
    row = bytes([128, 100, 128, 100, 128, 100, 128, 100])  # 2 groups -> 4 px
    msg = _msg(h, w, "yuv422", w * 2, row * h)
    out = decode_image_msg(msg)
    assert out.shape == (h, w, 3)
    assert out.dtype == np.uint8
    np.testing.assert_allclose(out, np.full((h, w, 3), 100, dtype=np.uint8), atol=1)


def test_yuv422_uyvy_color_direction():
    # UYVY [U=90, Y=200, V=240]: V>128 pushes red up (saturates), U<128 pulls blue down.
    h, w = 1, 2
    msg = _msg(h, w, "yuv422", w * 2, bytes([90, 200, 240, 200]))
    out = decode_image_msg(msg)
    r, g, b = (int(c) for c in out[0, 0])
    assert r == 255           # 200 + 1.402*112 = 357 -> clipped to 255
    assert 128 <= g <= 138    # ~133
    assert 128 <= b <= 138    # ~133


def test_yuv422_odd_width_raises():
    msg = _msg(1, 3, "yuv422", 6, bytes([128, 100, 128, 100, 128, 100]))
    with pytest.raises(ValueError):
        decode_image_msg(msg)


def test_yuv422_shm_nested_string_encoding():
    # Head stereo publishes shm_msgs/Image2m with encoding nested as a String.
    h, w = 1, 2
    data = np.zeros(2_097_152, dtype=np.uint8)
    data[:4] = np.array([128, 100, 128, 100], dtype=np.uint8)  # 1 group -> 2 gray px
    msg = _msg(h, w, _shm_string("yuv422"), w * 2, data)
    out = decode_image_msg(msg)
    assert out.shape == (h, w, 3)
    np.testing.assert_allclose(out, np.full((h, w, 3), 100, dtype=np.uint8), atol=1)


def _shm_string(text):
    """Mimic shm_msgs/String: char[256] data as a uint8 ndarray + size."""
    buf = np.zeros(256, dtype=np.uint8)
    encoded = text.encode("ascii")
    buf[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return SimpleNamespace(data=buf, size=len(encoded))


def test_shm_msgs_nested_string_encoding_rgb8():
    # shm_msgs/Image1m nests `encoding` as a String (char[256] data + size),
    # and `data` arrives as a fixed-size uint8 ndarray. Verified against a live
    # waist_front_rgbd frame: 640x360 rgb8, step=1920, data buffer 1048576.
    h, w = 1, 2
    rgb = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    data = np.zeros(1_048_576, dtype=np.uint8)
    data[: rgb.size] = rgb.reshape(-1)
    msg = _msg(h, w, _shm_string("rgb8"), w * 3, data)
    out = decode_image_msg(msg)
    np.testing.assert_array_equal(out, rgb)


def test_shm_msgs_nested_string_encoding_depth16():
    h, w = 1, 3
    raw = np.array([[1000, 2000, 0]], dtype=np.uint16)
    data = np.zeros(1_048_576, dtype=np.uint8)
    data[: raw.nbytes] = np.frombuffer(raw.tobytes(), dtype=np.uint8)
    msg = _msg(h, w, _shm_string("16UC1"), w * 2, data)
    out = decode_image_msg(msg, depth_scale=0.001)
    np.testing.assert_allclose(out, np.array([[1.0, 2.0, 0.0]], dtype=np.float32))
