# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters.piper.slot_pick."""

from __future__ import annotations

from jiuwensymbiosis.env.mock import MockArmEnv
from jiuwensymbiosis.tools.slot_pick.strategy import GripperStrategy
from tests.mocks.mock_api import MockApi


class TestPiperSlotPickFactory:
    def test_build_gripper_strategy(self):
        env = MockArmEnv()
        api = MockApi(env)
        strategy = GripperStrategy(api)
        assert hasattr(strategy, "goto_transit")
        assert hasattr(strategy, "grasp")
        assert hasattr(strategy, "release")
