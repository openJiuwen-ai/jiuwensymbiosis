# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``UbetechCruzrS2Env`` — adapter from ``BaseRobotEnv`` to ``UbetechCruzrS2Driver``.

Mobile-base form factor: capabilities = ``motion.cartesian`` (chassis x/y/yaw
via ROS2 cmd_vel) + ``vision.camera`` / ``vision.depth`` (ROS2 images) +
``vision.detection``. No ``motion.joint`` / ``grasp.*`` — a mobile base has no
arm or gripper.

Odometry is surfaced into ``RobotObservation.extra["odom"]`` (raw ROS units:
meters + quaternion + ``yaw_deg``), mirroring the unitree_go2 ROS2-odom pattern.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from jiuwensymbiosis.adapters.ubetech_cruzr_s2.config import UbetechCruzrS2Config
from jiuwensymbiosis.env.base import BaseRobotEnv, RobotObservation
from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger(__name__)


class UbetechCruzrS2Env(BaseRobotEnv):
    """UBTECH Cruzr S2 mobile base + ROS2 camera + odometry."""

    capabilities = frozenset(
        {
            "motion.cartesian",  # chassis x/y/yaw via ROS2 cmd_vel
            "vision.camera",  # ROS2 RGB stream via Ros2Camera
            "vision.depth",  # ROS2 depth stream
            "vision.detection",  # open-vocab detection (detector sidecar)
        }
    )
    name = "ubetech_cruzr_s2"

    def __init__(self, cfg: UbetechCruzrS2Config) -> None:
        """Store config; driver is None until connect().

        ``low_level`` (the driver slot) is inherited as a settable property
        from ``BaseRobotEnv`` — assign via ``self.low_level = ...`` in
        ``connect()`` / tests, read via ``self.low_level`` / ``self._low_level``.
        """
        self.cfg = cfg
        self._connected = False

    @property
    def z_min_safe(self) -> float | None:
        """Z floor (mm): 0.0 for a planar base (never triggers; SafetyRail contract)."""
        if self._low_level is not None:
            return 0.0
        return float(getattr(self.cfg, "z_min_safe_mm", 0.0))

    @z_min_safe.setter
    def z_min_safe(self, _: float | None) -> None:
        raise AttributeError("UbetechCruzrS2Env.z_min_safe is read-only (planar base, fixed at 0.0)")

    @property
    def workspace_bounds(self) -> tuple[float, float, float, float] | None:
        """XY workspace bounds ``(xmin, ymin, xmax, ymax)`` from config (meters), or None."""
        c = self.cfg
        raw = (
            getattr(c, "x_min_m", None),
            getattr(c, "y_min_m", None),
            getattr(c, "x_max_m", None),
            getattr(c, "y_max_m", None),
        )
        if any(v is None for v in raw):
            return None
        return cast("tuple[float, float, float, float]", raw)

    @workspace_bounds.setter
    def workspace_bounds(self, _: tuple[float, float, float, float] | None) -> None:
        raise AttributeError("UbetechCruzrS2Env.workspace_bounds is read-only (computed from config)")

    @property
    def home_pose(self):
        """Home pose (vendor Pose object) from the driver, or None before connect."""
        if self._low_level is not None:
            return self._low_level.home_pose
        return None

    @home_pose.setter
    def home_pose(self, _: Any) -> None:
        raise AttributeError("UbetechCruzrS2Env.home_pose is read-only (read from driver)")

    @property
    def tool_offset_mm(self) -> float:
        """Flange-to-tip offset (mm): always 0.0 for a mobile base."""
        return 0.0

    @tool_offset_mm.setter
    def tool_offset_mm(self, _: float) -> None:
        raise AttributeError("UbetechCruzrS2Env.tool_offset_mm is read-only (planar base, fixed at 0.0)")

    # ----------------------------------------------------------------- connect
    def connect(self) -> None:
        """Instantiate and connect the UbetechCruzrS2Driver from config."""
        if self._connected:
            return
        from jiuwensymbiosis.adapters.ubetech_cruzr_s2.lowlevel import UbetechCruzrS2Driver

        kwargs: dict[str, Any] = {
            "ros2_cmd_vel_topic": self.cfg.ros2_cmd_vel_topic,
            "ros2_cmd_vel_msg_kind": self.cfg.ros2_cmd_vel_msg_kind,
            "max_linear_speed_mps": self.cfg.max_linear_speed_mps,
            "max_angular_speed_radps": self.cfg.max_angular_speed_radps,
            "home_xy_yaw_m_deg": self.cfg.home_xy_yaw_m_deg,
            "camera_source": self.cfg.camera_source,
            "ros2_rgb_topic": self.cfg.ros2_rgb_topic,
            "ros2_depth_topic": self.cfg.ros2_depth_topic,
            "ros2_depth_scale_m": self.cfg.ros2_depth_scale_m,
            "ros2_camera_info_topic": self.cfg.ros2_camera_info_topic,
            "ros2_intrinsics": self.cfg.ros2_intrinsics,
            "ros2_odom_topic": self.cfg.ros2_odom_topic,
            "ros2_odom_msg_kind": self.cfg.ros2_odom_msg_kind,
        }
        self._low_level = UbetechCruzrS2Driver(**kwargs)
        self._low_level.connect()
        self._connected = True
        logger.info(
            "UbetechCruzrS2Env connected (cmd_vel=%s)",
            self.cfg.ros2_cmd_vel_topic or "(none)",
        )

    def disconnect(self) -> None:
        """Close the low-level driver and mark as disconnected."""
        if not self._connected:
            return
        try:
            # _low_level is RobotDriver|None here; close() is on the concrete driver
            self._low_level.close()  # type: ignore[union-attr]
        except Exception as exc:  # disconnect is best-effort
            logger.warning("UbetechCruzrS2Env disconnect failed: %s", exc)
        self._low_level = None
        self._connected = False

    # -------------------------------------------------------------- observation
    def get_observation(self) -> RobotObservation:
        """Collect base pose, RGB, depth, and odometry into a RobotObservation."""
        if self._low_level is None:
            return RobotObservation()
        rgb: np.ndarray | None = None
        depth: np.ndarray | None = None
        try:
            frames = self._low_level.grab_frames()  # type: ignore[attr-defined]  # CameraDriver sibling protocol
            if frames is not None:
                rgb, depth = frames
        except Exception as exc:  # camera read best-effort
            logger.debug("UbetechCruzrS2Env.grab_frames failed: %s", exc)
        pose: dict | None = None
        try:
            p = self._low_level.get_pose()
            pose = {
                "x": p.x,
                "y": p.y,
                "z": p.z,
                "rx": p.rx,
                "ry": p.ry,
                "rz": p.rz,
            }
        except Exception:  # pose read best-effort
            pose = None
        return RobotObservation(
            pose=pose,
            rgb=rgb,
            depth=depth,
            extra={
                "z_min_safe": self.z_min_safe,
                # Optional ROS2 odometry (meters + quaternion + yaw_deg); None when
                # no odom backend configured or no message has arrived yet.
                # ``_low_level`` is typed ``RobotDriver`` (the base Protocol), but
                # ``get_odom_pose`` is a ``UbetechCruzrS2Driver``-specific method
                # with no sibling Protocol — same pattern as piper/unitree_go2's
                # odom call, suppressed for the same reason.
                "odom": (
                    self._low_level.get_odom_pose()  # type: ignore[attr-defined]
                    if getattr(self.cfg, "ros2_odom_topic", None)
                    else None
                ),
            },
        )
