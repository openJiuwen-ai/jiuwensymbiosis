# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters._common.skills."""

from __future__ import annotations

from jiuwensymbiosis.adapters._common.skills import (
    move_tip_xy_then_z,
    pick_object_to_suction,
    place_suction_to_target,
)
from jiuwensymbiosis.env.mock import MockArmEnv
from tests.mocks.mock_api import MockApi


class TestMoveTipXyThenZ:
    def test_xy_then_z(self):
        env = MockArmEnv()
        api = MockApi(env)
        position = [100.0, 50.0, 150.0]
        move_tip_xy_then_z(api, position, stage="test")
        log = api._call_log
        goto_calls = [c for c in log if "goto_xyzr" in c]
        assert len(goto_calls) == 2

    def test_short_position_raises(self):
        env = MockArmEnv()
        api = MockApi(env)
        import pytest

        with pytest.raises(ValueError, match="expected"):
            move_tip_xy_then_z(api, [100.0], stage="test")


class TestPickObjectToSuction:
    def test_success(self):
        env = MockArmEnv()
        api = MockApi(env)
        result = pick_object_to_suction(api, "box")
        assert result.get("ok") is True

    def test_no_detection(self):
        env = MockArmEnv()
        api = MockApi(env, detection_result={"ok": False, "reason": "no_detection"})
        result = pick_object_to_suction(api, "box")
        assert result.get("ok") is False

    def test_missing_position(self):
        env = MockArmEnv()
        api = MockApi(env, detection_result={"ok": True})
        result = pick_object_to_suction(api, "box")
        assert result.get("ok") is False


class TestPlaceSuctionToTarget:
    def test_success(self):
        env = MockArmEnv()
        api = MockApi(env)
        result = place_suction_to_target(api, "target")
        assert result.get("ok") is True
