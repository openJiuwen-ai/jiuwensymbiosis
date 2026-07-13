# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters._common.ros2_scan.

These tests run without rclpy/sensor_msgs installed: the scan extractor is
pure-python and accepts any object with the right attributes, and the
``start()`` degradation path is exercised by hiding ``rclpy`` in ``sys.modules``.
Mirrors ``test_ros2_odom.py``.
"""

from __future__ import annotations

import sys
import types

import pytest

from jiuwensymbiosis.adapters._common.ros2_scan import Ros2Scan, _extract_scan


def _scan_msg(ranges=None, angle_min=-3.14, angle_max=3.14, range_min=0.1, range_max=10.0):
    """A sensor_msgs/LaserScan-shaped object (no rclpy needed)."""
    if ranges is None:
        ranges = [1.0, 2.0, 3.0]
    return types.SimpleNamespace(
        ranges=list(ranges),
        angle_min=angle_min,
        angle_max=angle_max,
        range_min=range_min,
        range_max=range_max,
    )


class TestExtractor:
    def test_extracts_fields(self):
        scan = _extract_scan(_scan_msg([1.0, 2.0, 3.0]))
        assert scan is not None
        assert scan["ranges"] == [1.0, 2.0, 3.0]
        assert scan["angle_min"] == pytest.approx(-3.14)
        assert scan["angle_max"] == pytest.approx(3.14)
        assert scan["range_min"] == pytest.approx(0.1)
        assert scan["range_max"] == pytest.approx(10.0)

    def test_missing_ranges_returns_none(self):
        assert _extract_scan(types.SimpleNamespace()) is None

    def test_missing_angle_min_returns_none(self):
        # ranges present but angle_min missing → None, no raise.
        msg = types.SimpleNamespace(ranges=[1.0], angle_max=1.0, range_min=0.1, range_max=10.0)
        assert _extract_scan(msg) is None

    def test_non_numeric_range_returns_none(self):
        # Non-numeric ranges must not raise; degrade to None.
        msg = types.SimpleNamespace(ranges=["x", 2.0], angle_min=-1.0, angle_max=1.0, range_min=0.1, range_max=10.0)
        assert _extract_scan(msg) is None

    def test_ranges_copied_to_plain_list(self):
        # The returned ranges must be a plain list (not an rclpy-typed array),
        # so callers/tests can index/slice it without surprise.
        scan = _extract_scan(_scan_msg([1.0, 2.0]))
        assert isinstance(scan["ranges"], list)


class TestRos2ScanConstructAndDegrade:
    def test_construction_never_raises(self):
        scan = Ros2Scan(scan_topic="/scan")
        assert scan.is_running is False
        assert scan.grab_scan() is None

    def test_start_returns_false_when_rclpy_missing(self, monkeypatch):
        for mod in (
            "rclpy",
            "rclpy.node",
            "rclpy.executors",
            "sensor_msgs",
            "sensor_msgs.msg",
        ):
            monkeypatch.setitem(sys.modules, mod, None)

        scan = Ros2Scan(scan_topic="/scan")
        assert scan.start() is False
        assert scan.is_running is False
        assert scan.grab_scan() is None

    def test_grab_scan_none_until_first_message(self):
        scan = Ros2Scan(scan_topic="/scan")
        assert scan.grab_scan() is None
        scan._on_scan(_scan_msg([1.0, 2.0]))
        got = scan.grab_scan()
        assert got is not None
        assert got["ranges"] == [1.0, 2.0]

    def test_callback_ignores_malformed_message(self):
        # A malformed message (missing ranges) must be dropped silently — the
        # latest scan stays the prior valid one. Provide a valid first message
        # so the assertion isn't vacuous.
        scan = Ros2Scan(scan_topic="/scan")
        scan._on_scan(_scan_msg([1.0]))  # valid
        assert scan.grab_scan() is not None
        scan._on_scan(types.SimpleNamespace())  # malformed — must not crash / overwrite
        got = scan.grab_scan()
        assert got is not None
        assert got["ranges"] == [1.0]  # valid scan retained
