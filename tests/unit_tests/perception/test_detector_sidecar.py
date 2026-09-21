# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.perception.detector_sidecar."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from jiuwensymbiosis.perception.detector_sidecar import _port_open, detector_subprocess


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


def test_attached_external_detector_is_never_stopped():
    with (
        patch("jiuwensymbiosis.perception.detector_sidecar._port_open", return_value=True),
        patch("jiuwensymbiosis.perception.detector_sidecar.subprocess.Popen") as spawn,
    ):
        owner = detector_subprocess()
        with owner as child:
            assert child is None
        assert owner.cleanup_report().released
        spawn.assert_not_called()
