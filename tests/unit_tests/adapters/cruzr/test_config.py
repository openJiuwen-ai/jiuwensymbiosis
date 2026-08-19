# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters.cruzr.config."""

from __future__ import annotations

import yaml

from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig


class TestCruzrConfig:
    def test_defaults(self):
        cfg = CruzrConfig()
        assert cfg.command_topic == "/mc/sdk/robot_command"
        assert cfg.state_topic == "/mc/sdk/robot_state"
        assert cfg.joint_name_for_arm("left") == "L_shoulder_pitch_joint"
        assert cfg.raise_position_for_arm("left") == 1.0
        assert cfg.raise_position_for_arm("right") == -1.0

    def test_from_dict_flat(self):
        cfg = CruzrConfig.from_dict({"default_arm": "right", "raise_position_rad": 0.8})
        assert cfg.default_arm == "right"
        assert cfg.raise_position_rad == 0.8

    def test_from_dict_nested(self):
        cfg = CruzrConfig.from_dict(
            {
                "env": {
                    "cfg": {
                        "low_level": {
                            "command_topic": "/custom/command",
                            "left_shoulder_pitch_joint": "left_joint",
                        },
                        "prompt": "raise arm",
                    },
                },
            }
        )
        assert cfg.command_topic == "/custom/command"
        assert cfg.left_shoulder_pitch_joint == "left_joint"
        assert cfg.task_prompt == "raise arm"

    def test_from_yaml(self, tmp_path):
        path = tmp_path / "cruzr.yaml"
        path.write_text(yaml.dump({"step_rad": 0.01}), encoding="utf-8")
        cfg = CruzrConfig.from_yaml(path)
        assert cfg.step_rad == 0.01

    def test_invalid_arm(self):
        cfg = CruzrConfig()
        try:
            cfg.joint_name_for_arm("middle")
        except ValueError as exc:
            assert "unknown arm" in str(exc)
        else:
            raise AssertionError("expected ValueError")


def test_camera_defaults():
    from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig

    cfg = CruzrConfig()
    assert cfg.waist_color_topic == "/sensor/camera/waist_front_rgbd/color/raw"
    assert cfg.waist_depth_topic == "/sensor/camera/waist_front_rgbd/depth/raw"
    assert cfg.waist_camera_info_topic == "/sensor/camera/waist_front_rgbd/color/info"
    assert cfg.color_msg_type == "shm_msgs/msg/Image1m"
    assert cfg.depth_scale == 0.001
    assert cfg.detector.url == "http://127.0.0.1:8114"


def test_detector_parsed_from_api_servers():
    from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig

    cfg = CruzrConfig.from_dict({
        "name": "cruzr_vis",
        "api_servers": [
            {"_target_": "jiuwensymbiosis.serving.grounding_dino_sam2_server",
             "host": "10.0.0.5", "port": 9000, "use_sam2": False},
        ],
    })
    assert cfg.detector.host == "10.0.0.5"
    assert cfg.detector.port == 9000
    assert cfg.detector.url == "http://10.0.0.5:9000"
    assert cfg.detector.use_sam2 is False


def test_camera_calib_path_from_low_level():
    from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig

    cfg = CruzrConfig.from_dict({"camera_calib_path": "/tmp/c.json"})
    assert cfg.camera_calib_path == "/tmp/c.json"


def test_cruzr_yaml_loads():
    from pathlib import Path

    from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig

    root = Path(__file__).resolve().parents[4]
    cfg = CruzrConfig.from_yaml(root / "configs" / "cruzr" / "cruzr.yaml")
    assert cfg.urdf_path.endswith("cruzr_s2_v1.urdf")
    assert cfg.left_arm_leaf == "L_sixforce_link"
    assert cfg.right_arm_leaf == "R_sixforce_link"
    assert cfg.contact_force_threshold_n == 2.5
    assert cfg.grasp_inset_mm == 5.0
    assert cfg.detector.url.startswith("http://")


def test_search_nav_defaults():
    from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig

    cfg = CruzrConfig()
    # head stereo (bearing search)
    assert cfg.head_left_topic == "/sensor/camera/stereo_left/image/raw"
    assert cfg.head_color_msg_type == "shm_msgs/msg/Image2m"
    assert cfg.head_yaw_joint == "head_yaw_joint"
    assert cfg.head_pitch_joint == "head_pitch_joint"
    assert cfg.head_hfov_rad > 0
    assert tuple(cfg.head_search_yaw_positions_rad)[0] == 0.0
    # base differential-drive (wheel velocity + odom closed loop; nav2 dropped)
    assert cfg.nav_python == "/usr/bin/python3"
    assert cfg.left_wheel_joint == "driving_wheel_left_joint"
    assert cfg.right_wheel_joint == "driving_wheel_right_joint"
    assert cfg.odom_topic == "/mc/odom"
    assert cfg.base_k_rot > 0 and cfg.base_k_fwd > 0
    # forward-drive deceleration (approach the box without overshooting into it)
    assert 0 < cfg.base_k_fwd_min <= cfg.base_k_fwd
    assert cfg.base_k_fwd_slow_m > 0
    assert cfg.base_safe_dist_m > cfg.base_lidar_self_floor_m > 0
    assert cfg.base_move_timeout_s > 0
    # approach loop
    assert 0.0 < cfg.center_tol_frac < 1.0
    assert cfg.approach_step_m > 0
    assert cfg.approach_max_iterations >= 1
    assert cfg.grasp_forward_min_m < cfg.grasp_forward_max_m
    # 停车距离设定点必须落在双臂可抓取带内
    assert cfg.grasp_forward_min_m <= cfg.grasp_target_forward_m <= cfg.grasp_forward_max_m
    # square-up yaw-consistency gate: a positive tolerance below 90° (a 90° gap is a long/short swap)
    assert 0.0 < cfg.grasp_yaw_consistency_tol_rad < 1.5708


def test_search_nav_fields_from_dict():
    from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig

    cfg = CruzrConfig.from_dict({"approach_step_m": 0.25, "head_hfov_rad": 1.4})
    assert cfg.approach_step_m == 0.25
    assert cfg.head_hfov_rad == 1.4


def test_grasp_target_forward_from_dict():
    from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig

    cfg = CruzrConfig.from_dict({"grasp_target_forward_m": 0.42})
    assert cfg.grasp_target_forward_m == 0.42


def test_waist_fields_defaults():
    """turn_waist config: waist joint name."""
    from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig

    cfg = CruzrConfig()
    assert cfg.waist_yaw_joint == "waist_yaw_joint"
