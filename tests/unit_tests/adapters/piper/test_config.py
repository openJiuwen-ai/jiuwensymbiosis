# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters.piper.config."""

from __future__ import annotations

import yaml

from jiuwensymbiosis.adapters.piper.config import (
    PiperConfig,
    _extract_detector_from_api_servers,
)


class TestPiperConfigDefaults:
    def test_from_dict_flat(self):
        cfg = PiperConfig.from_dict({"can_port": "can_right", "move_speed": 80})
        assert cfg.can_port == "can_right"
        assert cfg.move_speed == 80
        assert cfg.tool_offset_mm == 135.8

    def test_from_dict_nested(self):
        data = {
            "env": {
                "cfg": {
                    "low_level": {
                        "can_port": "can_left",
                        "move_speed": 30,
                        "tool_offset_mm": 100.0,
                    },
                    "prompt": "pick the box",
                },
            },
        }
        cfg = PiperConfig.from_dict(data)
        assert cfg.can_port == "can_left"
        assert cfg.move_speed == 30
        assert cfg.tool_offset_mm == 100.0
        assert cfg.task_prompt == "pick the box"

    def test_from_dict_defaults(self):
        cfg = PiperConfig.from_dict({})
        assert cfg.can_port == "can_left"
        assert cfg.z_min_safe_mm == 50.0


class TestExtractDetectorFromApiServers:
    def test_with_grounding_dino(self):
        servers = [
            {
                "_target_": "jiuwensymbiosis.serving.grounding_dino_sam2_server",
                "host": "192.168.1.10",
                "port": 9000,
            }
        ]
        cfg = _extract_detector_from_api_servers(servers)
        assert cfg.url == "http://192.168.1.10:9000"
        assert cfg.spawn is True

    def test_empty_servers(self):
        cfg = _extract_detector_from_api_servers([])
        assert cfg.url == "http://127.0.0.1:8114"

    def test_no_matching_server(self):
        servers = [{"_target_": "other_server", "host": "1.2.3.4", "port": 9999}]
        cfg = _extract_detector_from_api_servers(servers)
        assert cfg.url == "http://127.0.0.1:8114"


class TestPiperConfigFromYaml:
    def test_from_yaml(self, tmp_path):
        data = {
            "can_port": "can_left",
            "move_speed": 40,
            "tool_offset_mm": 135.8,
        }
        p = tmp_path / "test.yaml"
        p.write_text(yaml.dump(data), encoding="utf-8")
        cfg = PiperConfig.from_yaml(p)
        assert cfg.can_port == "can_left"
        assert cfg.move_speed == 40
