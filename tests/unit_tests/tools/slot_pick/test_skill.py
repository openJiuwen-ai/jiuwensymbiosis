# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.tools.slot_pick.skill."""

from __future__ import annotations

import pytest

from jiuwensymbiosis.tools.slot_pick.skill import (
    SlotPickConfig,
    geometric_completion_judge,
    run_slot_pick,
)
from jiuwensymbiosis.tools.slot_pick.strategy import GripperStrategy
from tests.mocks.mock_api import MockApi
from jiuwensymbiosis.env.mock import MockArmEnv


class TestSlotPickConfig:
    def test_from_mapping_minimal(self):
        cfg = SlotPickConfig.from_mapping(
            {
                "chip_object_name": "chip",
                "slot_object_name": "slot",
            }
        )
        assert cfg.chip_object_name == "chip"
        assert cfg.slot_object_name == "slot"

    def test_from_mapping_missing_required(self):
        with pytest.raises(ValueError, match="required"):
            SlotPickConfig.from_mapping({})

    def test_from_mapping_full(self):
        cfg = SlotPickConfig.from_mapping(
            {
                "chip_object_name": "chip",
                "slot_object_name": "slot",
                "max_pick_place_cycles": 5,
                "place_done_radius_mm": 80.0,
                "chip_thickness_mm": 10.0,
            }
        )
        assert cfg.max_pick_place_cycles == 5
        assert cfg.place_done_radius_mm == 80.0

    def test_merged(self):
        cfg = SlotPickConfig.from_mapping(
            {
                "chip_object_name": "chip",
                "slot_object_name": "slot",
            }
        )
        merged = cfg.merged({"max_pick_place_cycles": 3})
        assert merged.max_pick_place_cycles == 3
        assert merged.chip_object_name == "chip"

    def test_merged_no_overrides(self):
        cfg = SlotPickConfig.from_mapping(
            {
                "chip_object_name": "chip",
                "slot_object_name": "slot",
            }
        )
        merged = cfg.merged({})
        assert merged is cfg


class TestGeometricCompletionJudge:
    def test_done_when_close(self):
        env = MockArmEnv()
        detection = {
            "ok": True,
            "position": [230.0, 0.0, 50.0],
            "score": 0.9,
        }
        api = MockApi(env, detection_result=detection)
        cfg = SlotPickConfig.from_mapping(
            {
                "chip_object_name": "chip",
                "slot_object_name": "slot",
                "place_done_radius_mm": 100.0,
            }
        )
        result = geometric_completion_judge(api, cfg)
        assert isinstance(result, bool)

    def test_not_done_when_far(self):
        env = MockArmEnv()
        chip_det = {"ok": True, "position": [100.0, 0.0, 50.0], "score": 0.9}
        slot_det = {"ok": True, "position": [500.0, 0.0, 50.0], "score": 0.9}
        call_count = [0]

        class FarApi(MockApi):
            def get_grasp_info_simple(self, object_name):
                call_count[0] += 1
                return chip_det if call_count[0] % 2 == 1 else slot_det

        api = FarApi(env)
        cfg = SlotPickConfig.from_mapping(
            {
                "chip_object_name": "chip",
                "slot_object_name": "slot",
                "place_done_radius_mm": 10.0,
            }
        )
        result = geometric_completion_judge(api, cfg)
        assert result is False


class TestRunSlotPick:
    def test_single_cycle(self):
        env = MockArmEnv()
        api = MockApi(env)
        cfg = SlotPickConfig.from_mapping(
            {
                "chip_object_name": "chip",
                "slot_object_name": "slot",
                "max_pick_place_cycles": 1,
            }
        )
        strategy = GripperStrategy(api)
        result = run_slot_pick(api, cfg, strategy)
        assert isinstance(result, dict)
        assert "ok" in result
