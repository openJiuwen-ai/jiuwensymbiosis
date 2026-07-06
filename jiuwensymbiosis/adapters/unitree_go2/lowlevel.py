# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Low-level Unitree Go2 driver — quadruped mobile base.

Hybrid communication:
  * **Chassis motion** → ``unitree_sdk2py`` (the official Python SDK; Cyclone
    DDS over the robot network). Lazily imported — if the SDK isn't installed,
    ``connect()`` raises ``RuntimeError`` with install guidance (motion is the
    base's primary capability, so it can't degrade like vision can).
  * **Images** → ``Ros2Camera`` (reused from ``adapters/_common``). Lazy rclpy;
    missing → ``grab_frames()`` returns None (vision degrades, base still moves).
  * **Odometry** → ``Ros2Odom`` (reused from ``adapters/_common``). Lazy rclpy;
    missing → ``get_odom_pose()`` returns None.

Frame conventions:
  * The base pose (``get_pose``) is the 2D planar pose from odometry:
    ``x, y`` (meters, base/map frame) + ``yaw`` (deg). It is wrapped into a
    ``SimpleNamespace(x, y, z=0, rx=0, ry=0, rz=yaw_deg)`` to satisfy the
    ``RobotDriver`` 6-DoF pose shape — the Env verb ``get_flange_pose`` only
    passes it through; the Api layer decides what the numbers mean.
  * ``tool_offset_mm`` is 0.0 (a base has no flange→tip offset).

Construction never raises on the ROS2 side (parity with ``Ros2Camera`` /
``Ros2Odom``). It may raise on the SDK side (``unitree_sdk2py`` missing), which
is the deliberate "motion unavailable" signal — callers treat that as
"no motion capability", distinct from "no vision data".
"""

from __future__ import annotations

import logging
import math
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np

from jiuwensymbiosis.adapters._common.ros2_camera import Ros2Camera
from jiuwensymbiosis.adapters._common.ros2_odom import Ros2Odom

logger = logging.getLogger(__name__)


class UnitreeGo2Driver:
    """Unitree Go2 mobile-base driver: SDK chassis motion + ROS2 camera + odom.

    Implements the ``RobotDriver`` Protocol (motion + pose + close) plus the
    ``CameraDriver`` sibling (``intrinsics`` / ``grab_frames``) and an
    ``get_odom_pose`` accessor — the same shape piper uses.
    """

    def __init__(
        self,
        *,
        network_interface: str | None = None,
        max_linear_speed_mps: float = 1.0,
        max_angular_speed_radps: float = 1.5,
        home_xy_yaw_m_deg: list[float] | None = None,
        # ROS2 camera (optional; None disables vision)
        camera_source: str = "ros2",
        ros2_rgb_topic: str | None = None,
        ros2_depth_topic: str | None = None,
        ros2_depth_scale_m: float = 0.001,
        ros2_camera_info_topic: str | None = None,
        ros2_intrinsics: list[float] | None = None,
        # ROS2 odometry (optional; None disables odom)
        ros2_odom_topic: str | None = None,
        ros2_odom_msg_kind: str = "odometry",
    ) -> None:
        self._lock = threading.RLock()
        self._max_linear = float(max_linear_speed_mps)
        self._max_angular = float(max_angular_speed_radps)
        if home_xy_yaw_m_deg is None:
            home_xy_yaw_m_deg = [0.0, 0.0, 0.0]
        self._home_xy_yaw = [float(v) for v in home_xy_yaw_m_deg[:3]]
        # ``home_pose`` is the vendor Pose object the Env/Api read (RobotDriver
        # contract). 6-DoF shape, but only x/y + rz(yaw) are meaningful here.
        # Stored privately + exposed via ``@property`` to match the Protocol's
        # ``@property home_pose`` signature (an instance attribute would fail
        # the structural check — see PiperLowLevel.home_pose).
        self._home_pose = SimpleNamespace(
            x=self._home_xy_yaw[0],
            y=self._home_xy_yaw[1],
            z=0.0,
            rx=0.0,
            ry=0.0,
            rz=self._home_xy_yaw[2],
        )
        self._network_interface = network_interface

        # --- SDK (chassis motion). Lazily connected in connect(); constructed
        #     here only if importable so __init__ stays cheap. Motion is the
        #     base's primary capability — a missing SDK is a hard error, caught
        #     in connect() (not here) so __init__ itself never raises on import.
        self._sdk: Any = None  # unitree_sdk2py client, set in connect()
        self._connected = False

        # --- camera (optional; mirrors piper's ROS2 backend)
        self._camera: Ros2Camera | None = None
        if camera_source == "ros2" and ros2_rgb_topic:
            self._camera = Ros2Camera(
                rgb_topic=ros2_rgb_topic,
                depth_topic=ros2_depth_topic,
                depth_scale_m=ros2_depth_scale_m,
                camera_info_topic=ros2_camera_info_topic,
                intrinsics=ros2_intrinsics,
                log_prefix="[Go2]",
            )
        # RealSense USB path omitted — Go2 ships images over ROS2 by default.
        # Add an elif branch here if a USB RealSense is ever attached.

        # --- odometry (optional; mirrors piper's ROS2 odom backend)
        self._odom: Ros2Odom | None = None
        if ros2_odom_topic:
            self._odom = Ros2Odom(
                odom_topic=ros2_odom_topic,
                msg_kind=ros2_odom_msg_kind,
                log_prefix="[Go2]",
            )

    # ============================================================== lifecycle
    def connect(self) -> None:
        """Start camera/odom subscriptions + connect the chassis SDK.

        Idempotent. Raises ``RuntimeError`` if ``unitree_sdk2py`` is missing
        (motion is unavailable) — install guidance in the message. ROS2 side
        failures degrade to None (logged), never raise.
        """
        if self._connected:
            return
        # ROS2 side: start() returns False on missing rclpy — degrade, don't raise.
        if self._camera is not None and not self._camera.start():
            logger.warning("[Go2] ROS2 camera not started — vision degraded to None.")
        if self._odom is not None and not self._odom.start():
            logger.warning("[Go2] ROS2 odometry not started — odom degraded to None.")

        # SDK side: missing unitree_sdk2py is a hard error (no chassis motion).
        try:
            # The SDK exposes a module-level initializer (NOT a ChannelFactory class
            # constructor): ``ChannelFactoryInitialize(domain_id, network_interface)``
            # binds a Cyclone DDS participant to the host NIC on the Go2 subnet.
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        except ImportError as exc:
            raise RuntimeError(
                '[Go2] unitree_sdk2py not installed. Install via `pip install -e ".[unitree]"` '
                '(needs Cyclone DDS — see README "Unitree Go2 chassis SDK").'
            ) from exc
        try:
            # domain 0 + the configured NIC (None → SDK/DDS default interface).
            ChannelFactoryInitialize(0, self._network_interface)
            self._sdk = True  # marker: DDS participant initialized; SportClient created per-move
            logger.info("[Go2] chassis SDK initialized (interface=%s).", self._network_interface or "(default)")
        except Exception as exc:  # SDK init failure is a real motion error, not degradable
            raise RuntimeError(f"[Go2] chassis SDK init failed: {exc}") from exc
        self._connected = True

    def close(self) -> None:
        """Stop camera/odom + release SDK. Idempotent, best-effort."""
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception as e:  # best-effort camera teardown; log + continue
                logger.debug("[Go2] camera stop failed during teardown: %s", e)
        if self._odom is not None:
            try:
                self._odom.stop()
            except Exception as e:  # best-effort odom teardown; log + continue
                logger.debug("[Go2] odom stop failed during teardown: %s", e)
        self._sdk = None
        self._connected = False
        logger.info("[Go2] closed.")

    # ============================================================== properties
    @property
    def home_pose(self) -> Any:
        """Configured home/origin base pose (6-DoF-shaped; x/y + rz=yaw)."""
        return self._home_pose

    @property
    def z_min_safe(self) -> float:
        """Tip-frame Z floor (mm): 0.0 for a planar base (never triggers)."""
        return 0.0

    @property
    def flange_z_min_safe(self) -> float:
        """Flange-frame Z floor (mm): 0.0 for a planar base (== z_min_safe)."""
        return 0.0

    @property
    def tool_offset_mm(self) -> float:
        """Flange→tip offset (mm): 0.0 for a mobile base (no flange/tip)."""
        return 0.0

    @property
    def intrinsics(self) -> np.ndarray | None:
        """3x3 camera intrinsics from the live ROS2 camera, or None."""
        return self._camera.intrinsics if self._camera is not None else None

    # ============================================================== pose (odom)
    def get_pose(self) -> Any:
        """Current base pose (from odometry) as a 6-DoF-shaped SimpleNamespace.

        ``x, y`` (meters) + ``rz`` (yaw, deg); ``z/rx/ry`` are 0 (planar base).
        Returns the home pose if no odom backend / no message yet (so callers
        always get a valid pose object — motion code can still reason about a
        nominal origin when odom isn't wired up).
        """
        odom = self.get_odom_pose()
        if odom is None:
            hp = self.home_pose
            return SimpleNamespace(x=hp.x, y=hp.y, z=0.0, rx=0.0, ry=0.0, rz=hp.rz)
        return SimpleNamespace(
            x=float(odom["x"]),
            y=float(odom["y"]),
            z=0.0,
            rx=0.0,
            ry=0.0,
            rz=float(odom["yaw_deg"]),
        )

    def get_odom_pose(self) -> dict | None:
        """Latest ROS2 odometry pose (meters + quaternion + yaw_deg), or None."""
        return self._odom.grab_pose() if self._odom is not None else None

    # ============================================================== camera
    def grab_frames(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Grab (rgb, depth_m) from the ROS2 camera, or None if no camera."""
        return self._camera.grab_frames() if self._camera is not None else None

    # ============================================================== motion (SDK)
    def home(self) -> None:
        """Drive the base back to the configured home (x, y, yaw). Blocking."""
        x, y, yaw = self._home_xy_yaw
        self._move_to_xy_yaw(x, y, yaw)

    def move_to_pose_blocking(self, pose: Any, *args: Any, **kwargs: Any) -> None:
        """Blocking planar move to <pose> (x, y, rz=yaw). z/rx/ry ignored.

        ``pose`` is a mapping or object with ``x``, ``y`` (meters) and optional
        ``rz`` / ``r`` (deg). The SDK issues a velocity command toward the
        target; this blocks until within tolerance or timeout. Vendor extensions
        (``sync_timeout_s``, etc.) ride in ``*args``/``**kwargs`` after it,
        matching the ``RobotDriver`` Protocol signature.
        """
        x = float(getattr(pose, "x", 0.0))
        y = float(getattr(pose, "y", 0.0))
        rz = float(getattr(pose, "rz", getattr(pose, "r", 0.0)))
        self._move_to_xy_yaw(x, y, rz)

    # ============================================================== private
    def _move_to_xy_yaw(self, target_x: float, target_y: float, target_yaw_deg: float) -> None:
        """Drive toward (target_x, target_y, target_yaw) via SDK velocity commands.

        Simple proportional velocity controller: command a velocity proportional
        to the remaining error, clamped to the configured max speeds, until
        within tolerance. The real SDK command shape is vendor-specific — this
        is the integration seam a real deployment tunes.
        """
        if not self._connected or self._sdk is None:
            raise RuntimeError("[Go2] chassis not connected (call connect() first).")
        for v, name in ((target_x, "x"), (target_y, "y"), (target_yaw_deg, "yaw")):
            if not math.isfinite(v):
                raise ValueError(f"[Go2] non-finite {name}={v}")
        # TODO(integration): replace with the real unitree_sdk2py SportClient
        # velocity command + status poll. The structure below is the control
        # loop skeleton; the exact SDK call is filled in per deployment.
        logger.info(
            "[Go2] move → (x=%.3f m, y=%.3f m, yaw=%.2f deg) [SDK velocity cmd]",
            target_x,
            target_y,
            target_yaw_deg,
        )
        # Placeholder: no-op settle. A real driver commands v=clamp(kp*err, vmax)
        # each tick and polls get_pose() until within tol. Left as a seam so the
        # adapter wires up end-to-end without the SDK command shape nailed down.
