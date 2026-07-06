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


class TestEnvVarOverrides:
    """Environment variable overrides for detector model IDs and camera serial."""

    def test_gdino_model_id_env_override(self, monkeypatch):
        monkeypatch.setenv("GDINO_MODEL_ID", "my-org/custom-gdino")
        servers = [
            {
                "_target_": "jiuwensymbiosis.serving.grounding_dino_sam2_server",
                "gdino_model_id": "IDEA-Research/grounding-dino-base",
            }
        ]
        cfg = _extract_detector_from_api_servers(servers)
        assert cfg.gdino_model_id == "my-org/custom-gdino"

    def test_sam2_model_id_env_override(self, monkeypatch):
        monkeypatch.setenv("SAM2_MODEL_ID", "my-org/custom-sam2")
        servers = [
            {
                "_target_": "jiuwensymbiosis.serving.grounding_dino_sam2_server",
                "sam2_model_id": "facebook/sam2.1-hiera-large",
            }
        ]
        cfg = _extract_detector_from_api_servers(servers)
        assert cfg.sam2_model_id == "my-org/custom-sam2"

    def test_env_override_no_yaml_value(self, monkeypatch):
        """Env var should apply even when YAML doesn't set the field (defaults are used)."""
        monkeypatch.setenv("GDINO_MODEL_ID", "env-only-dino")
        monkeypatch.setenv("SAM2_MODEL_ID", "env-only-sam2")
        cfg = _extract_detector_from_api_servers([])
        assert cfg.gdino_model_id == "env-only-dino"
        assert cfg.sam2_model_id == "env-only-sam2"

    def test_camera_serial_env_override(self, monkeypatch):
        monkeypatch.setenv("CAMERA_SERIAL", "999999999999")
        cfg = PiperConfig.from_dict({"camera_serial": "123456789012"})
        assert cfg.camera_serial == "999999999999"

    def test_camera_serial_env_override_no_yaml(self, monkeypatch):
        monkeypatch.setenv("CAMERA_SERIAL", "999999999999")
        cfg = PiperConfig.from_dict({})
        assert cfg.camera_serial == "999999999999"

    def test_ros2_fields_passthrough_from_yaml(self):
        data = {
            "env": {
                "cfg": {
                    "low_level": {
                        "camera_source": "ros2",
                        "ros2_rgb_topic": "/camera/color/image_raw",
                        "ros2_depth_topic": "/camera/aligned_depth_to_color/image_raw",
                        "ros2_depth_scale_m": 0.001,
                        "ros2_camera_info_topic": "/camera/color/camera_info",
                        "ros2_intrinsics": [615.0, 0.0, 320.0, 0.0, 615.0, 240.0, 0.0, 0.0, 1.0],
                    }
                }
            }
        }
        cfg = PiperConfig.from_dict(data)
        assert cfg.camera_source == "ros2"
        assert cfg.ros2_rgb_topic == "/camera/color/image_raw"
        assert cfg.ros2_depth_topic == "/camera/aligned_depth_to_color/image_raw"
        assert cfg.ros2_depth_scale_m == 0.001
        assert cfg.ros2_camera_info_topic == "/camera/color/camera_info"
        # intrinsics kept as a list; Ros2Camera wraps it in a 3x3 ndarray.
        assert cfg.ros2_intrinsics[0] == 615.0
        assert cfg.ros2_intrinsics[4] == 615.0
        assert cfg.ros2_intrinsics[8] == 1.0

    def test_camera_source_defaults_to_realsense(self):
        cfg = PiperConfig.from_dict({})
        assert cfg.camera_source == "realsense"
        assert cfg.ros2_rgb_topic is None

    def test_ros2_odom_fields_passthrough_from_yaml(self):
        data = {
            "env": {
                "cfg": {
                    "low_level": {
                        "ros2_odom_topic": "/odom",
                        "ros2_odom_msg_kind": "pose_stamped",
                    }
                }
            }
        }
        cfg = PiperConfig.from_dict(data)
        assert cfg.ros2_odom_topic == "/odom"
        assert cfg.ros2_odom_msg_kind == "pose_stamped"

    def test_ros2_odom_defaults(self):
        cfg = PiperConfig.from_dict({})
        assert cfg.ros2_odom_topic is None
        assert cfg.ros2_odom_msg_kind == "odometry"
