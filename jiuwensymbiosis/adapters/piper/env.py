# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``PiperEnv`` — adapter from ``BaseRobotEnv`` to ``PiperLowLevel``.

Wraps the driver (``low_level``), exposes ``connect``/``disconnect``/
``get_observation`` plus the safety contract (``z_min_safe`` /
``workspace_bounds``). Motion/end-effector use the inherited Env verbs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from jiuwensymbiosis.adapters.piper.config import PiperConfig
from jiuwensymbiosis.env.base import BaseRobotEnv, RobotObservation

if TYPE_CHECKING:
    from jiuwensymbiosis.adapters._common.protocol import RobotDriver

logger = logging.getLogger(__name__)


class PiperEnv(BaseRobotEnv):
    """6-DoF AgileX Piper + parallel gripper + optional wrist RealSense."""

    capabilities = frozenset(
        {
            "motion.cartesian",
            "motion.joint",
            "motion.servo",
            "grasp.parallel",
            "vision.camera",
            "vision.depth",
            "vision.detection",
        }
    )
    name = "piper"

    def __init__(self, cfg: PiperConfig) -> None:
        """Store config; driver is None until connect()."""
        self.cfg = cfg
        self._inner: RobotDriver | None = None  # PiperLowLevel
        self._connected = False

    @property
    def low_level(self) -> RobotDriver | None:
        """The underlying low-level driver (PiperLowLevel), or None before connect()."""
        return self._inner

    @low_level.setter
    def low_level(self, _: RobotDriver | None) -> None:
        raise AttributeError("PiperEnv.low_level is read-only (binds to self._inner via connect/disconnect)")

    @property
    def z_min_safe(self) -> float | None:
        """Tip-frame Z floor (mm): from the live driver if connected, else config."""
        if self._inner is not None:
            return float(self._inner.z_min_safe)
        cfg_val = getattr(self.cfg, "z_min_safe_mm", None)
        return float(cfg_val) if cfg_val is not None else None

    @z_min_safe.setter
    def z_min_safe(self, _: float | None) -> None:
        raise AttributeError("PiperEnv.z_min_safe is read-only (computed from driver/config)")

    @property
    def workspace_bounds(self) -> tuple[float, float, float, float] | None:
        """XY workspace bounds ``(xmin, ymin, xmax, ymax)`` in mm from config, or None."""
        c = self.cfg
        raw = (
            getattr(c, "x_min_mm", None),
            getattr(c, "y_min_mm", None),
            getattr(c, "x_max_mm", None),
            getattr(c, "y_max_mm", None),
        )
        if any(v is None for v in raw):
            return None
        return cast("tuple[float, float, float, float]", raw)

    @workspace_bounds.setter
    def workspace_bounds(self, _: tuple[float, float, float, float] | None) -> None:
        raise AttributeError("PiperEnv.workspace_bounds is read-only (computed from config)")

    @property
    def home_pose(self):
        """Home pose (vendor Pose object) from the driver, or None before connect."""
        if self._inner is not None:
            return self._inner.home_pose
        return None

    @home_pose.setter
    def home_pose(self, _: Any) -> None:
        raise AttributeError("PiperEnv.home_pose is read-only (read from driver)")

    @property
    def tool_offset_mm(self) -> float:
        """Flange-to-tip offset (mm) from the driver, or 0 before connect."""
        if self._inner is not None:
            return float(self._inner.tool_offset_mm)
        return float(getattr(self.cfg, "tool_offset_mm", 0.0))

    @tool_offset_mm.setter
    def tool_offset_mm(self, _: float) -> None:
        raise AttributeError("PiperEnv.tool_offset_mm is read-only (computed from driver/config)")

    # ----------------------------------------------------------------- connect
    def connect(self) -> None:
        """Instantiate and connect the PiperLowLevel driver from config."""
        if self._connected:
            return
        from jiuwensymbiosis.adapters.piper.lowlevel import PiperLowLevel

        kwargs: dict[str, Any] = dict(  # noqa: C408  # mutable builder, conditionally extended below
            can_port=self.cfg.can_port,
            move_speed=self.cfg.move_speed,
            tool_offset_mm=self.cfg.tool_offset_mm,
            home_lift_mm=self.cfg.home_lift_mm,
            z_safe_margin_mm=self.cfg.z_safe_margin_mm,
            home_use_init_pose=self.cfg.home_use_init_pose,
            x_min_mm=self.cfg.x_min_mm,
            x_max_mm=self.cfg.x_max_mm,
            y_min_mm=self.cfg.y_min_mm,
            y_max_mm=self.cfg.y_max_mm,
            z_max_mm=self.cfg.z_max_mm,
            camera_resolution=tuple(self.cfg.camera_resolution),
            camera_fps=self.cfg.camera_fps,
            gripper_open_mm=self.cfg.gripper_open_mm,
            gripper_effort=self.cfg.gripper_effort,
            gripper_settle_s=self.cfg.gripper_settle_s,
        )
        if self.cfg.calib_path:
            kwargs["calib_path"] = self.cfg.calib_path
        else:
            kwargs["home_pose_xyzrxryrz_mm_deg"] = self.cfg.home_pose_xyzrxryrz_mm_deg
            if self.cfg.calib_object_xyzrxryrz_mm_deg:
                kwargs["calib_object_xyzrxryrz_mm_deg"] = self.cfg.calib_object_xyzrxryrz_mm_deg
            kwargs["z_min_safe_mm"] = self.cfg.z_min_safe_mm
        if self.cfg.camera_serial:
            kwargs["camera_serial"] = self.cfg.camera_serial
        if getattr(self.cfg, "camera_source", "realsense") == "ros2":
            kwargs["camera_source"] = "ros2"
            kwargs["ros2_rgb_topic"] = self.cfg.ros2_rgb_topic
            kwargs["ros2_depth_topic"] = self.cfg.ros2_depth_topic
            kwargs["ros2_depth_scale_m"] = self.cfg.ros2_depth_scale_m
            kwargs["ros2_camera_info_topic"] = self.cfg.ros2_camera_info_topic
            kwargs["ros2_intrinsics"] = self.cfg.ros2_intrinsics
        if getattr(self.cfg, "ros2_odom_topic", None):
            kwargs["ros2_odom_topic"] = self.cfg.ros2_odom_topic
            kwargs["ros2_odom_msg_kind"] = self.cfg.ros2_odom_msg_kind

        self._inner = PiperLowLevel(**kwargs)
        self._connected = True
        logger.info("PiperEnv connected (can_port=%s)", self.cfg.can_port)

    def disconnect(self) -> None:
        """Close the low-level driver and mark as disconnected."""
        if not self._connected:
            return
        try:
            # `_inner` is non-None here: `_connected` is set True only after
            # `_inner` is assigned in connect(); mypy can't track the invariant.
            self._inner.close()  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 - disconnect is best-effort
            logger.warning("PiperEnv disconnect failed: %s", exc)
        self._inner = None
        self._connected = False

    # -------------------------------------------------------------- observation
    def get_observation(self) -> RobotObservation:
        """Collect RGB, depth, pose, and joint state into a RobotObservation."""
        if self._inner is None:
            return RobotObservation()
        rgb: np.ndarray | None = None
        depth: np.ndarray | None = None
        try:
            frames = self._inner.grab_frames()  # type: ignore[attr-defined]  # CameraDriver sibling protocol
            if frames is not None:
                rgb, depth = frames
        except Exception as exc:  # noqa: BLE001 - camera read best-effort
            logger.debug("PiperEnv.grab_frames failed: %s", exc)
        pose: dict | None = None
        try:
            p = self._inner.get_pose()
            pose = {"x": p.x, "y": p.y, "z": p.z, "rx": p.rx, "ry": p.ry, "rz": p.rz}
        except Exception:  # noqa: BLE001 - pose read best-effort
            pose = None
        joints: list[float] | None = None
        try:
            a = self._inner.get_angles()  # type: ignore[attr-defined]  # JointDriver sibling protocol
            joints = list(a.as_tuple())
        except Exception:  # noqa: BLE001 - joint read best-effort
            joints = None
        return RobotObservation(
            pose=pose,
            joints=joints,
            rgb=rgb,
            depth=depth,
            extra={
                "z_min_safe": self.z_min_safe,
                # GripperDriver sibling protocol; grasp.parallel-capability-gated
                "gripper_state": (
                    self._inner.gripper_state  # type: ignore[attr-defined]
                    if "grasp.parallel" in self.capabilities
                    else None
                ),
                # Optional ROS2 odometry (meters + quaternion + yaw_deg); None when
                # no odom backend configured or no message has arrived yet.
                "odom": (
                    # ``_inner`` is typed ``RobotDriver`` (the base Protocol), but
                    # ``get_odom_pose`` is a ``PiperLowLevel``-specific method with no
                    # sibling Protocol — same pattern as the grab_frames / get_angles
                    # sibling-protocol calls above, suppressed here for the same reason.
                    self._inner.get_odom_pose()  # type: ignore[attr-defined]
                    if getattr(self.cfg, "ros2_odom_topic", None)
                    else None
                ),
            },
        )

    def get_angles(self) -> Any:
        """Read joint angles from the driver; raise if not connected."""
        if self._inner is None:
            raise RuntimeError("PiperEnv.get_angles: env not connected.")
        return self._inner.get_angles()  # type: ignore[attr-defined]  # JointDriver sibling protocol
