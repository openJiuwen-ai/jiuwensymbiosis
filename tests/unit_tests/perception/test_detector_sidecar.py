# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.perception.detector_sidecar."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from jiuwensymbiosis.perception.detector_sidecar import _port_open


class TestPortOpen:
    def test_closed_port_returns_false(self):
        sock = MagicMock()
        sock.connect_ex.return_value = 111
        socket_context = MagicMock()
        socket_context.__enter__.return_value = sock

        with patch("jiuwensymbiosis.perception.detector_sidecar.socket.socket", return_value=socket_context):
            result = _port_open("127.0.0.1", 59999, timeout=0.5)

        assert result is False
        sock.settimeout.assert_called_once_with(0.5)
        sock.connect_ex.assert_called_once_with(("127.0.0.1", 59999))
