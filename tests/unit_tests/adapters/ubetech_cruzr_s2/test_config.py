# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters.ubetech_cruzr_s2.config."""

from __future__ import annotations

import yaml

from jiuwensymbiosis.adapters.ubetech_cruzr_s2.config import UbetechCruzrS2Config


class TestUbetechCruzrS2ConfigDefaults:
    def test_from_dict_flat(self):
        cfg = UbetechCruzrS2Config.from_dict({"ros2_cmd_vel_topic": "/cmd_vel", "max_linear_speed_mps": 0.5})
        assert cfg.ros2_cmd_vel_topic == "/cmd_vel"
        assert cfg.max_linear_speed_mps == 0.5
        assert cfg.name == "ubetech_cruzr_s2"
        assert cfg.camera_source == "ros2"

    def test_from_dict_defaults(self):
        cfg = UbetechCruzrS2Config.from_dict({})
        assert cfg.ros2_cmd_vel_topic == "/cmd_vel"
        assert cfg.ros2_cmd_vel_msg_kind == "twist"
        assert cfg.max_linear_speed_mps == 1.0
        assert cfg.max_angular_speed_radps == 1.5
        assert cfg.ros2_odom_topic is None
        assert cfg.ros2_odom_msg_kind == "odometry"
        assert cfg.z_min_safe_mm == 0.0  # planar base — Z not actuated

    def test_home_xy_yaw_defaults(self):
        cfg = UbetechCruzrS2Config.from_dict({})
        assert cfg.home_xy_yaw_m_deg == [0.0, 0.0, 0.0]


class TestUbetechCruzrS2ConfigFromYaml:
    def test_from_yaml_ros2_fields_passthrough(self, tmp_path):
        data = {
            "name": "ubetech_cruzr_s2",
            "ros2_cmd_vel_topic": "/cmd_vel",
            "ros2_cmd_vel_msg_kind": "twist_stamped",
            "camera_source": "ros2",
            "ros2_rgb_topic": "/cruzr/camera/color/image_raw",
            "ros2_depth_topic": "/cruzr/camera/aligned_depth_to_color/image_raw",
            "ros2_depth_scale_m": 0.001,
            "ros2_camera_info_topic": "/cruzr/camera/color/camera_info",
            "ros2_intrinsics": [615.0, 0.0, 320.0, 0.0, 615.0, 240.0, 0.0, 0.0, 1.0],
            "ros2_odom_topic": "/odom",
            "ros2_odom_msg_kind": "pose_stamped",
        }
        p = tmp_path / "cruzr.yaml"
        p.write_text(yaml.dump(data), encoding="utf-8")
        cfg = UbetechCruzrS2Config.from_yaml(p)
        assert cfg.ros2_cmd_vel_topic == "/cmd_vel"
        assert cfg.ros2_cmd_vel_msg_kind == "twist_stamped"
        assert cfg.ros2_rgb_topic == "/cruzr/camera/color/image_raw"
        assert cfg.ros2_depth_topic == "/cruzr/camera/aligned_depth_to_color/image_raw"
        assert cfg.ros2_depth_scale_m == 0.001
        assert cfg.ros2_camera_info_topic == "/cruzr/camera/color/camera_info"
        assert cfg.ros2_intrinsics[0] == 615.0
        assert cfg.ros2_intrinsics[8] == 1.0
        assert cfg.ros2_odom_topic == "/odom"
        assert cfg.ros2_odom_msg_kind == "pose_stamped"

    def test_from_yaml_ignores_unknown_keys(self, tmp_path):
        # Extra YAML keys must not break loading (silently ignored).
        data = {"name": "ubetech_cruzr_s2", "unknown_field": 123, "another": "x"}
        p = tmp_path / "cruzr.yaml"
        p.write_text(yaml.dump(data), encoding="utf-8")
        cfg = UbetechCruzrS2Config.from_yaml(p)
        assert cfg.name == "ubetech_cruzr_s2"

    def test_from_yaml_resolves_relative_calib(self, tmp_path):
        (tmp_path / "calib.json").write_text("{}", encoding="utf-8")
        data = {"calib_path": "calib.json"}
        p = tmp_path / "cruzr.yaml"
        p.write_text(yaml.dump(data), encoding="utf-8")
        cfg = UbetechCruzrS2Config.from_yaml(p)
        assert cfg.calib_path is not None
        assert cfg.calib_path.endswith("calib.json")
        assert cfg.calib_path.startswith(str(tmp_path))
