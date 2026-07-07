# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters._common.detector_sidecar."""

from __future__ import annotations

import pytest

from jiuwensymbiosis.adapters._common import detector_sidecar as sidecar_mod


class _FakeSocket:
    def __init__(self, connect_result: int) -> None:
        self.connect_result = connect_result
        self.timeout = None
        self.addr = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def settimeout(self, timeout):
        self.timeout = timeout

    def connect_ex(self, addr):
        self.addr = addr
        return self.connect_result


class TestPortOpen:
    @pytest.mark.parametrize(("connect_result", "expected"), [(111, False), (0, True)], ids=["closed", "open"])
    def test_port_open_uses_connect_ex_result(self, monkeypatch, connect_result, expected):
        fake = _FakeSocket(connect_result=connect_result)
        monkeypatch.setattr(sidecar_mod.socket, "socket", lambda *_args: fake)

        assert sidecar_mod._port_open("127.0.0.1", 8114, timeout=0.5) is expected
        assert fake.timeout == 0.5
        assert fake.addr == ("127.0.0.1", 8114)
