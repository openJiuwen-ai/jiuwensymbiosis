# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.tools.slot_pick.strategy."""

from __future__ import annotations

from jiuwensymbiosis.env.mock import MockArmEnv
from jiuwensymbiosis.tools.slot_pick.strategy import (
    GripperStrategy,
    SlotPickStrategy,
    _api_z_min_safe,
    _clamp_radius,
)
from tests.mocks.mock_api import MockApi


class TestClampRadius:
    def test_no_clamp_when_disabled(self):
        x, y, clamped = _clamp_radius(500, 300, 0.0)
        assert (x, y) == (500, 300)
        assert clamped is False

    def test_within_radius(self):
        x, y, clamped = _clamp_radius(10, 10, 100)
        assert (x, y) == (10, 10)
        assert clamped is False

    def test_exceeds_radius(self):
        x, y, clamped = _clamp_radius(300, 400, 100)
        r = (x**2 + y**2) ** 0.5
        assert abs(r - 100) < 1e-6
        assert clamped is True

    def test_zero_point(self):
        x, y, clamped = _clamp_radius(0, 0, 100)
        assert (x, y) == (0, 0)
        assert clamped is False


class TestApiZMinSafe:
    def test_reads_from_env_property(self):
        # Source of truth is the formalized env.z_min_safe contract property.
        api = MockApi(MockArmEnv(z_min_safe=37.5))
        assert _api_z_min_safe(api) == 37.5

    def test_none_when_env_has_no_floor(self):
        class _Api:
            class _Env:
                z_min_safe = None

            env = _Env()

        assert _api_z_min_safe(_Api()) is None


class TestGripperStrategy:
    def test_grasp(self, mock_api):
        s = GripperStrategy(mock_api)
        result = s.grasp()
        assert result.get("ok") is True

    def test_release(self, mock_api):
        s = GripperStrategy(mock_api)
        result = s.release()
        assert result.get("ok") is True

    def test_goto_transit(self, mock_api):
        s = GripperStrategy(mock_api)
        s.goto_transit(100, 50, 300, 0)
        assert any("goto_xyzr" in c or "goto_xyzr_joint" in c for c in mock_api._call_log)

    def test_goto_critical(self, mock_api):
        s = GripperStrategy(mock_api)
        s.goto_critical(100, 50, 300, 0)
        assert any("goto_xyzr" in c for c in mock_api._call_log)

    def test_goto_process(self, mock_api):
        s = GripperStrategy(mock_api)
        s.goto_process(100, 50, 300, 0)
        assert len(mock_api._call_log) > 0

    def test_goto_exact(self, mock_api):
        s = GripperStrategy(mock_api)
        s.goto_exact(100, 50, 300, 0)
        assert "goto_xyzr(100,50,300,0)" in mock_api._call_log

    def test_protocol_satisfied(self, mock_api):
        s = GripperStrategy(mock_api)
        assert isinstance(s, SlotPickStrategy)
