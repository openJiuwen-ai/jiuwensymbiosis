# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.perception.detector_sidecar."""

from __future__ import annotations

from jiuwensymbiosis.perception.detector_sidecar import _port_open


class TestPortOpen:
    def test_closed_port_returns_false(self):
        result = _port_open("127.0.0.1", 59999, timeout=0.5)
        assert result is False
