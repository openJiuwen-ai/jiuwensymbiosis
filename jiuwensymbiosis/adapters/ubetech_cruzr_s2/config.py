# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``UbetechCruzrS2Config`` — UBTECH Cruzr S2 (mobile-base) adapter config.

Form factor: **pure mobile base** (no arm / end-effector). Capabilities:
``motion.cartesian`` (body x/y/yaw via ROS2 velocity commands) +
``vision.camera`` / ``vision.depth`` (ROS2 image topics, via ``Ros2Camera``) +
optional ``vision.detection``. Odometry is read from a ROS2 pose topic via
``Ros2Odom``.

Communication is **pure ROS2**: chassis motion, images, and odometry all flow
over ROS2 topics — no vendor SDK required. The motion topic name and message
type are configurable (``ros2_cmd_vel_topic`` / ``ros2_cmd_vel_msg_kind``) so
the adapter fits whatever the robot's ROS2 driver exposes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class UbetechCruzrS2Config:
    """Hardware configuration for the UBTECH Cruzr S2 (mobile-base form factor).

    Use ``from_yaml(path)`` to load from a YAML file, or construct directly
    with keyword arguments.
    """

    # ==================== 基本信息 [必填] ====================
    name: str = "ubetech_cruzr_s2"

    # ==================== 底盘运动 (ROS2 cmd_vel) [必填-仅 motion.cartesian] ====================
    # ROS2 velocity-command publisher. The driver publishes a velocity command
    # to ``ros2_cmd_vel_topic``; the message type is chosen by
    # ``ros2_cmd_vel_msg_kind``:
    #   * "twist"         → ``geometry_msgs/msg/Twist``       (linear.x/y + angular.z)
    #   * "twist_stamped" → ``geometry_msgs/msg/TwistStamped`` (same, with header)
    # Both carry the standard 2D-planar velocity (vx, vy m/s; wz rad/s).
    # Topic name and type are both configurable to match the robot's driver.
    ros2_cmd_vel_topic: str = "/cmd_vel"
    ros2_cmd_vel_msg_kind: str = "twist"  # or twist_stamped
    # chassis velocity limits (m/s and rad/s). Enforced in the driver at the
    # hardware boundary (clamped before publishing).
    max_linear_speed_mps: float = 1.0  # m/s
    max_angular_speed_radps: float = 1.5  # rad/s
    # [选填] Home / origin pose of the base (x_m, y_m, yaw_deg) — 2D planar.
    home_xy_yaw_m_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # ==================== ROS2 相机 (复用 Ros2Camera) [选填-仅 vision.*] ====================
    camera_source: str = "ros2"  # Cruzr S2 ships images over ROS2
    ros2_rgb_topic: str | None = None
    ros2_depth_topic: str | None = None
    ros2_depth_scale_m: float = 0.001  # 16UC1 raw unit → meters (RealSense default = 1 mm)
    ros2_camera_info_topic: str | None = None
    # Explicit 3x3 intrinsics (row-major 9-list) when no camera_info topic.
    ros2_intrinsics: list[float] | None = None

    # ==================== ROS2 里程计 (复用 Ros2Odom) [选填] ====================
    # The framework is a pure CONSUMER of the odom topic — it does NOT run any
    # SLAM itself. The pose must be produced on the robot side by a SLAM /
    # odometry stack (LiDAR SLAM / VIO / wheel+IMU EKF) you deploy alongside.
    # Surfaced into ``RobotObservation.extra["odom"]``.
    ros2_odom_topic: str | None = None
    ros2_odom_msg_kind: str = "odometry"  # or pose_stamped / pose_with_covariance_stamped

    # ==================== 安全边界 [选填] ====================
    # Base is 2D-planar; z is not actuated. ``z_min_safe`` stays 0.0 to satisfy
    # the SafetyRail contract (it never triggers on a non-z-actuated base);
    # ``x_min/max`` etc. bound the base's XY roaming range in **meters** (base-
    # frame units differ from arm-flange mm; SafetyRail only checks the values,
    # not their unit — so the field is named ``_m`` to match the real unit).
    z_min_safe_mm: float = 0.0
    x_min_m: float | None = -5.0
    x_max_m: float | None = 5.0
    y_min_m: float | None = -5.0
    y_max_m: float | None = 5.0
    z_max_mm: float | None = None  # base doesn't move in Z; no ceiling

    # ==================== 检测校正 [选填-仅 vision.detection] ====================
    z_correction_mm: float = 0.0
    grasp_z_offset_mm: float = -25.0
    chip_thickness_mm: float = 75.0

    # ==================== 检测服务 [选填-仅 vision.detection] ====================
    detector_url: str = "http://127.0.0.1:8114"

    # ==================== 标定 [选填-仅 vision.detection] ====================
    calib_path: str | None = None

    # ==================== 任务 [选填] ====================
    task_prompt: str | None = None

    # ========================================================================
    #  Loaders — framework contract (do NOT modify the shape)
    # ========================================================================

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UbetechCruzrS2Config:
        """Construct config from a flat dict (only matching field names are used)."""
        valid = {f.name for f in dataclasses.fields(cls)}
        clean: dict[str, Any] = {k: v for k, v in data.items() if k in valid}
        return cls(**clean)

    @classmethod
    def from_yaml(cls, path: str | Path) -> UbetechCruzrS2Config:
        """Load config from a YAML file, resolving relative calib_path."""
        path = Path(path).resolve()
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = cls.from_dict(data)
        if cfg.calib_path and not Path(cfg.calib_path).is_absolute():
            candidate = (path.parent / cfg.calib_path).resolve()
            if candidate.exists():
                cfg.calib_path = str(candidate)
        return cfg
