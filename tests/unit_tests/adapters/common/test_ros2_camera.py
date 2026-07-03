# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters._common.ros2_camera.

These tests run without rclpy/sensor_msgs installed: the encoding converters
are pure-numpy and accept any object with the right shape attributes, and the
``start()`` degradation path is exercised by monkeypatching ``import rclpy`` to
raise ``ImportError``.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from jiuwensymbiosis.adapters._common.ros2_camera import (
    Ros2Camera,
    _ros_to_depth_m,
    _ros_to_rgb,
)


def _img_msg(encoding: str, height: int, width: int, *, dtype: str, fill=0):
    """Build a sensor_msgs/Image-shaped object without sensor_msgs."""
    channels = 4 if encoding in ("rgba8", "bgra8") else 3 if encoding in ("rgb8", "bgr8") else 1
    shape = (height, width, channels) if channels > 1 else (height, width)
    arr = np.full(shape, fill, dtype=dtype)
    if encoding in ("rgba8", "bgra8"):
        arr[..., 3] = 255  # opaque alpha
    return types.SimpleNamespace(
        encoding=encoding,
        height=height,
        width=width,
        data=arr.tobytes(),
    )


class TestRosToRgb:
    def test_bgr8_flips_to_rgb(self):
        # Distinct per-channel values so the flip is observable.
        bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        bgr[..., 0] = 10  # B
        bgr[..., 1] = 20  # G
        bgr[..., 2] = 30  # R
        msg = types.SimpleNamespace(encoding="bgr8", height=2, width=2, data=bgr.tobytes())
        rgb = _ros_to_rgb(msg)
        assert rgb is not None
        assert rgb.dtype == np.uint8
        assert rgb[0, 0].tolist() == [30, 20, 10]  # R, G, B

    def test_rgb8_passthrough(self):
        rgb_arr = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb_arr[..., 0] = 100  # R
        rgb_arr[..., 1] = 110  # G
        rgb_arr[..., 2] = 120  # B
        msg = types.SimpleNamespace(encoding="rgb8", height=2, width=2, data=rgb_arr.tobytes())
        rgb = _ros_to_rgb(msg)
        assert rgb is not None
        assert rgb[0, 0].tolist() == [100, 110, 120]

    def test_rgba8_drops_alpha(self):
        rgba = np.zeros((2, 2, 4), dtype=np.uint8)
        rgba[..., 0] = 5  # R
        rgba[..., 1] = 6  # G
        rgba[..., 2] = 7  # B
        rgba[..., 3] = 255  # alpha (dropped)
        msg = types.SimpleNamespace(encoding="rgba8", height=2, width=2, data=rgba.tobytes())
        rgb = _ros_to_rgb(msg)
        assert rgb is not None
        assert rgb.shape == (2, 2, 3)
        assert rgb[0, 0].tolist() == [5, 6, 7]

    def test_bgra8_drops_alpha_and_flips(self):
        bgra = np.zeros((2, 2, 4), dtype=np.uint8)
        bgra[..., 0] = 40  # B
        bgra[..., 1] = 41  # G
        bgra[..., 2] = 42  # R
        bgra[..., 3] = 255  # alpha
        msg = types.SimpleNamespace(encoding="bgra8", height=2, width=2, data=bgra.tobytes())
        rgb = _ros_to_rgb(msg)
        assert rgb is not None
        assert rgb[0, 0].tolist() == [42, 41, 40]

    def test_unknown_encoding_returns_none(self):
        msg = types.SimpleNamespace(encoding="yuv422", height=2, width=2, data=b"\x00" * 8)
        assert _ros_to_rgb(msg) is None

    def test_missing_attributes_returns_none(self):
        # No .height/.width/.data — must not raise, just return None.
        assert _ros_to_rgb(types.SimpleNamespace(encoding="rgb8")) is None

    def test_buffer_shape_mismatch_returns_none(self):
        # 2x2 rgb8 needs 12 bytes; give 5 → reshape fails → None, no raise.
        msg = types.SimpleNamespace(encoding="rgb8", height=2, width=2, data=b"\x00" * 5)
        assert _ros_to_rgb(msg) is None


class TestRosToDepthM:
    def test_16uc1_applies_scale(self):
        raw = np.array([[0, 1000], [2000, 3000]], dtype=np.uint16)  # mm
        msg = types.SimpleNamespace(encoding="16UC1", height=2, width=2, data=raw.tobytes())
        depth = _ros_to_depth_m(msg, depth_scale_m=0.001)
        assert depth is not None
        assert depth.dtype == np.float32
        # float32: 3000 * 0.001 is not exactly 3.0, compare with tolerance.
        assert np.allclose(depth, [[0.0, 1.0], [2.0, 3.0]])

    def test_16uc1_custom_scale(self):
        raw = np.array([[10]], dtype=np.uint16)
        msg = types.SimpleNamespace(encoding="16UC1", height=1, width=1, data=raw.tobytes())
        depth = _ros_to_depth_m(msg, depth_scale_m=0.0001)  # 0.1 mm/unit
        assert depth is not None
        assert depth[0, 0] == pytest.approx(0.001)

    def test_32fc1_passthrough(self):
        raw = np.array([[0.5, 1.25]], dtype=np.float32)  # already meters
        msg = types.SimpleNamespace(encoding="32FC1", height=1, width=2, data=raw.tobytes())
        depth = _ros_to_depth_m(msg, depth_scale_m=0.001)
        assert depth is not None
        assert depth.dtype == np.float32
        assert depth[0].tolist() == [0.5, 1.25]

    def test_unknown_encoding_returns_none(self):
        msg = types.SimpleNamespace(encoding="mono8", height=1, width=1, data=b"\x00")
        assert _ros_to_depth_m(msg, depth_scale_m=0.001) is None

    def test_missing_attributes_returns_none(self):
        assert _ros_to_depth_m(types.SimpleNamespace(encoding="16UC1"), depth_scale_m=0.001) is None

    def test_invalid_scale_returns_none(self):
        # Non-numeric scale must not raise; degrade to None.
        msg = types.SimpleNamespace(encoding="16UC1", height=1, width=1, data=b"\x00\x00")
        assert _ros_to_depth_m(msg, depth_scale_m="not-a-number") is None  # type: ignore[arg-type]

    def test_buffer_shape_mismatch_returns_none(self):
        # 2x2 16UC1 needs 8 bytes; give 3 → reshape fails → None, no raise.
        msg = types.SimpleNamespace(encoding="16UC1", height=2, width=2, data=b"\x00" * 3)
        assert _ros_to_depth_m(msg, depth_scale_m=0.001) is None


class TestRos2CameraConstructAndDegrade:
    def test_construction_never_raises(self):
        # Even if rclpy is absent, __init__ must succeed (RealSenseCamera parity).
        cam = Ros2Camera(rgb_topic="/x/image_raw", depth_topic="/x/depth")
        assert cam.is_running is False
        assert cam.grab_frames() is None
        assert cam.intrinsics is None

    def test_start_returns_false_when_rclpy_missing(self, monkeypatch):
        # Simulate rclpy not importable: hide the module + the sensor_msgs import
        # path inside Ros2Camera.start().
        for mod in ("rclpy", "rclpy.node", "rclpy.executors", "sensor_msgs", "sensor_msgs.msg"):
            monkeypatch.setitem(sys.modules, mod, None)

        cam = Ros2Camera(rgb_topic="/x/image_raw")
        assert cam.start() is False
        assert cam.is_running is False
        assert cam.grab_frames() is None

    def test_intrinsics_from_constructor(self):
        k = [615.0, 0.0, 320.0, 0.0, 615.0, 240.0, 0.0, 0.0, 1.0]
        cam = Ros2Camera(rgb_topic="/x/image_raw", intrinsics=k)
        intr = cam.intrinsics
        assert intr is not None
        assert intr.shape == (3, 3)
        assert intr[0, 0] == 615.0
        assert intr[0, 2] == 320.0
        assert intr[1, 1] == 615.0
        assert intr[1, 2] == 240.0

    def test_malformed_intrinsics_does_not_raise(self):
        # "Construction never raises" — a list with len != 9 must degrade to
        # None + warning, NOT raise ValueError.
        cam = Ros2Camera(rgb_topic="/x/image_raw", intrinsics=[1.0, 2.0, 3.0])
        assert cam.intrinsics is None
        assert cam.is_running is False
        assert cam.grab_frames() is None

    def test_camera_info_callback_sets_intrinsics(self):
        cam = Ros2Camera(rgb_topic="/x/image_raw")
        # A CameraInfo-shaped message: .k is a 9-element row-major list.
        msg = types.SimpleNamespace(k=[700.0, 0.0, 319.0, 0.0, 700.0, 239.0, 0.0, 0.0, 1.0])
        cam._on_camera_info(msg)
        intr = cam.intrinsics
        assert intr is not None
        assert intr[0, 0] == 700.0
        assert intr[0, 2] == 319.0
        assert intr[1, 1] == 700.0
        assert intr[1, 2] == 239.0

    def test_grab_frames_returns_none_until_both_arrive(self):
        cam = Ros2Camera(rgb_topic="/x/image_raw", depth_topic="/x/depth")
        # Simulate callbacks arriving: RGB first, then depth.
        rgb_msg = _img_msg("rgb8", 2, 2, dtype="uint8", fill=128)
        cam._on_rgb(rgb_msg)
        assert cam.grab_frames() is None  # no depth yet
        depth_msg = _img_msg("16UC1", 2, 2, dtype="uint16", fill=500)
        cam._on_depth(depth_msg)
        frames = cam.grab_frames()
        assert frames is not None
        rgb, depth = frames
        assert rgb.shape == (2, 2, 3)
        assert depth.shape == (2, 2)
        assert depth.dtype == np.float32
        assert depth[0, 0] == pytest.approx(0.5)  # 500 mm * 0.001

    def test_grab_frames_none_when_shapes_mismatch(self):
        cam = Ros2Camera(rgb_topic="/x/image_raw", depth_topic="/x/depth")
        cam._on_rgb(_img_msg("rgb8", 4, 4, dtype="uint8", fill=1))
        cam._on_depth(_img_msg("16UC1", 2, 2, dtype="uint16", fill=1000))
        assert cam.grab_frames() is None

    def test_rgb_callback_ignores_unknown_encoding(self):
        # Provide a depth_topic + a valid depth frame so grab_frames() can only
        # return None if the RGB frame was genuinely dropped for unknown encoding
        # (not because depth is missing). This avoids a vacuous assertion.
        cam = Ros2Camera(rgb_topic="/x/image_raw", depth_topic="/x/depth")
        cam._on_rgb(types.SimpleNamespace(encoding="yuv422", height=2, width=2, data=b"\x00" * 8))
        cam._on_depth(_img_msg("16UC1", 2, 2, dtype="uint16", fill=1000))
        # RGB was ignored → no RGB frame → grab_frames stays None despite depth.
        assert cam.grab_frames() is None
