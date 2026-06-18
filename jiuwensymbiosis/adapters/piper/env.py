# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``PiperEnv`` — adapter from ``BaseRobotEnv`` to ``PiperLowLevel``.

Wraps the driver (``low_level``), exposes ``connect``/``disconnect``/
``get_observation`` plus the safety contract (``z_min_safe`` /
``workspace_bounds``). Motion/end-effector use the inherited Env verbs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

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
        self._inner: Optional["RobotDriver"] = None  # PiperLowLevel
        self._connected = False

    @property
    def low_level(self) -> Optional["RobotDriver"]:
        """The underlying low-level driver (PiperLowLevel), or None before connect()."""
        return self._inner

    @property
    def z_min_safe(self) -> Optional[float]:
        """Tip-frame Z floor (mm): from the live driver if connected, else config."""
        if self._inner is not None:
            val = getattr(self._inner, "z_min_safe", None)
            if val is not None:
                return float(val)
        cfg_val = getattr(self.cfg, "z_min_safe_mm", None)
        return float(cfg_val) if cfg_val is not None else None

    @property
    def workspace_bounds(self) -> Optional[tuple[float, float, float, float]]:
        """XY workspace bounds ``(xmin, ymin, xmax, ymax)`` in mm from config, or None."""
        c = self.cfg
        xmin, ymin = getattr(c, "x_min_mm", None), getattr(c, "y_min_mm", None)
        xmax, ymax = getattr(c, "x_max_mm", None), getattr(c, "y_max_mm", None)
        if None in (xmin, ymin, xmax, ymax):
            return None
        return (float(xmin), float(ymin), float(xmax), float(ymax))

    @property
    def home_pose(self):
        """Home pose (vendor Pose object) from the driver, or None before connect."""
        if self._inner is not None:
            return self._inner.home_pose
        return None

    @property
    def tool_offset_mm(self) -> float:
        """Flange-to-tip offset (mm) from the driver, or 0 before connect."""
        if self._inner is not None:
            return float(self._inner.tool_offset_mm)
        return float(getattr(self.cfg, "tool_offset_mm", 0.0))

    # ----------------------------------------------------------------- connect
    def connect(self) -> None:
        """Instantiate and connect the PiperLowLevel driver from config."""
        if self._connected:
            return
        from jiuwensymbiosis.adapters.piper.lowlevel import PiperLowLevel

        kwargs: dict[str, Any] = dict(
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

        self._inner = PiperLowLevel(**kwargs)
        self._connected = True
        logger.info("PiperEnv connected (can_port=%s)", self.cfg.can_port)

    def disconnect(self) -> None:
        """Close the low-level driver and mark as disconnected."""
        if not self._connected:
            return
        try:
            close = getattr(self._inner, "close", None)
            if callable(close):
                close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("PiperEnv disconnect failed: %s", exc)
        self._inner = None
        self._connected = False

    # -------------------------------------------------------------- observation
    def get_observation(self) -> RobotObservation:
        """Collect RGB, depth, pose, and joint state into a RobotObservation."""
        if self._inner is None:
            return RobotObservation()
        rgb: Optional[np.ndarray] = None
        depth: Optional[np.ndarray] = None
        try:
            frames = self._inner.grab_frames()
            if frames is not None:
                rgb, depth = frames
        except Exception as exc:  # noqa: BLE001
            logger.debug("PiperEnv.grab_frames failed: %s", exc)
        pose: Optional[dict] = None
        try:
            p = self._inner.get_pose()
            pose = {"x": p.x, "y": p.y, "z": p.z, "rx": p.rx, "ry": p.ry, "rz": p.rz}
        except Exception:  # noqa: BLE001
            pose = None
        joints: Optional[list[float]] = None
        try:
            a = self._inner.get_angles()
            joints = list(a.as_tuple())
        except Exception:  # noqa: BLE001
            joints = None
        return RobotObservation(
            pose=pose,
            joints=joints,
            rgb=rgb,
            depth=depth,
            extra={
                "z_min_safe": self.z_min_safe,
                "gripper_state": getattr(self._inner, "gripper_state", None),
            },
        )

    def get_angles(self) -> Any:
        """Read joint angles from the driver; raise if not connected."""
        if self._inner is None:
            raise RuntimeError("PiperEnv.get_angles: env not connected.")
        return self._inner.get_angles()
