# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Low-level UBTECH Cruzr S2 driver — mobile base, pure ROS2.

Pure ROS2 communication (no vendor SDK):
  * **Chassis motion** → a ``geometry_msgs/Twist`` (or ``TwistStamped``)
    publisher on a configurable topic (``ros2_cmd_vel_topic``). The message
    type is chosen by ``ros2_cmd_vel_msg_kind``. Lazy rclpy; missing →
    ``move_to_pose_blocking`` raises ``RuntimeError`` at call time (motion is
    the base's primary capability, but unlike the Go2 SDK path we never raise
    in ``connect()`` — a missing rclpy degrades motion+vision+odom together,
    which is the consistent "ROS2 not available" signal).
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

Construction never raises (parity with ``Ros2Camera`` / ``Ros2Odom``). The
only rclpy import happens in ``connect()``; failure degrades every ROS2 side
to "not started" rather than raising — callers treat that as "ROS2 backend
unavailable", distinct from a per-motion runtime error.
"""

from __future__ import annotations

import importlib
import math
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np

from jiuwensymbiosis.adapters._common.ros2_camera import Ros2Camera
from jiuwensymbiosis.adapters._common.ros2_odom import Ros2Odom
from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger(__name__)


# ``msg_kind`` → (ROS2 module, message type name). Add a row here to support
# another velocity-carrying message type.
_CMD_VEL_MSG_KINDS: dict[str, tuple[str, str]] = {
    "twist": ("geometry_msgs.msg", "Twist"),
    "twist_stamped": ("geometry_msgs.msg", "TwistStamped"),
}


class _Ros2CmdVel:
    """One ROS2 velocity-command publisher, exposed as a synchronous writer.

    Mirrors ``Ros2Odom``'s surface/lifecycle so the two are interchangeable in
    spirit behind the adapter driver.

    Lifecycle:
      * ``__init__`` only stores config.
      * ``start()`` creates the node + publisher + spin thread. Idempotent.
      * ``stop()`` tears them down. Idempotent.
      * ``publish_twist(vx, vy, wz)`` issues one non-blocking velocity command.
        Returns False (no-op) before ``start()`` succeeds.

    The message type is chosen by ``msg_kind`` (see ``_CMD_VEL_MSG_KINDS``).
    ``Twist`` carries ``linear.x/y`` + ``angular.z``; ``TwistStamped`` wraps
    the same ``Twist`` under ``.twist`` with a header (auto-filled by the
    publisher when ``frame_id`` is empty).
    """

    def __init__(
        self,
        cmd_vel_topic: str,
        *,
        msg_kind: str = "twist",
        log_prefix: str = "[ROS2]",
    ) -> None:
        self._cmd_vel_topic = cmd_vel_topic
        self._log_prefix = log_prefix
        # Construction must never raise (parity with Ros2Odom/Ros2Camera): an
        # unknown msg_kind degrades to "twist" + warning rather than ValueError.
        self._msg_kind = msg_kind
        if msg_kind not in _CMD_VEL_MSG_KINDS:
            logger.warning(
                "%s unknown cmd_vel msg_kind=%r (expected one of %s); falling back to 'twist'.",
                self._log_prefix,
                msg_kind,
                sorted(_CMD_VEL_MSG_KINDS),
            )
            self._msg_kind = "twist"

        self._node: Any = None  # rclpy.node.Node once started
        self._publisher: Any = None  # rclpy.publisher once started
        self._executor: Any = None  # SingleThreadedExecutor once started
        self._spin_thread: threading.Thread | None = None
        self._msg_cls: Any = None  # resolved message class
        self._owns_rclpy = False  # whether WE called rclpy.init() (so we shutdown)

    # ----------------------------------------------------------------- state
    @property
    def is_running(self) -> bool:
        """True once ``start()`` has created the node + publisher + spin thread."""
        return self._node is not None

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> bool:
        """Create the node + publisher and start the spin thread.

        Idempotent. Returns True on success, False on any failure (with a
        warning logged). Never raises — missing rclpy / init errors all degrade
        to ``publish_twist()`` being a no-op.
        """
        if self._node is not None:
            return True
        mod_name, type_name = _CMD_VEL_MSG_KINDS[self._msg_kind]
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node

            msg_mod = importlib.import_module(mod_name)
            self._msg_cls = getattr(msg_mod, type_name)
        except ImportError:
            logger.warning(
                "%s rclpy/%s not available — skipping motion publisher. "
                "Source /opt/ros/<distro>/setup.bash (under a ROS2-compatible "
                "interpreter) to enable.",
                self._log_prefix,
                mod_name.split(".")[0],
            )
            return False
        try:
            if not rclpy.ok():
                rclpy.init(args=None)
                self._owns_rclpy = True
            self._node = Node("jiuwensymbiosis_ros2_cmd_vel")
            self._publisher = self._node.create_publisher(self._msg_cls, self._cmd_vel_topic, 10)
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._spin_thread = threading.Thread(
                target=self._executor.spin,
                name="ros2_cmd_vel_spin",
                daemon=True,
            )
            self._spin_thread.start()
            logger.info(
                "%s ROS2 cmd_vel ready (topic=%s, msg_kind=%s).",
                self._log_prefix,
                self._cmd_vel_topic,
                self._msg_kind,
            )
            return True
        except Exception as e:
            logger.warning(
                "%s ROS2 cmd_vel init failed (%s); continuing without motion publisher.",
                self._log_prefix,
                e,
            )
            self._safe_teardown()
            return False

    def stop(self) -> None:
        """Stop the spin thread, destroy the node. Safe to call multiple times."""
        if self._node is None and self._executor is None:
            return
        self._safe_teardown()

    # -------------------------------------------------------------- publish
    def publish_twist(self, vx: float, vy: float, wz: float) -> bool:
        """Publish one velocity command (vx, vy m/s; wz rad/s).

        Non-blocking. Returns True if published, False if the publisher isn't
        running (rclpy missing / start() failed). Never raises.
        """
        if self._publisher is None or self._msg_cls is None:
            return False
        try:
            msg = self._msg_cls()
            twist = msg.twist if self._msg_kind == "twist_stamped" else msg
            twist.linear.x = float(vx)
            twist.linear.y = float(vy)
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = float(wz)
            self._publisher.publish(msg)
            return True
        except Exception as e:  # best-effort publish; log + continue
            logger.debug("%s cmd_vel publish failed: %s", self._log_prefix, e)
            return False

    # ============================================================== private
    def _safe_teardown(self) -> None:
        """Best-effort: stop executor, join thread, destroy node, maybe shutdown."""
        if self._executor is not None:
            try:
                self._executor.shutdown()
            except Exception as e:
                logger.debug("%s executor.shutdown failed during teardown: %s", self._log_prefix, e)
        if self._spin_thread is not None and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception as e:
                logger.debug("%s destroy_node failed during teardown: %s", self._log_prefix, e)
        if self._owns_rclpy:
            try:
                import rclpy

                if rclpy.ok():
                    rclpy.shutdown()
            except Exception as e:
                logger.debug("%s rclpy.shutdown failed during teardown: %s", self._log_prefix, e)
            self._owns_rclpy = False
        self._node = None
        self._publisher = None
        self._executor = None
        self._spin_thread = None
        self._msg_cls = None


class UbetechCruzrS2Driver:
    """UBTECH Cruzr S2 mobile-base driver: ROS2 cmd_vel + camera + odom.

    Implements the ``RobotDriver`` Protocol (motion + pose + close) plus the
    ``CameraDriver`` sibling (``intrinsics`` / ``grab_frames``) and an
    ``get_odom_pose`` accessor — the same shape piper / unitree_go2 use.
    """

    def __init__(
        self,
        *,
        ros2_cmd_vel_topic: str = "/cmd_vel",
        ros2_cmd_vel_msg_kind: str = "twist",
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

        # --- motion (ROS2 cmd_vel publisher). Lazily started in connect().
        self._cmd_vel: _Ros2CmdVel | None = None
        if ros2_cmd_vel_topic:
            self._cmd_vel = _Ros2CmdVel(
                cmd_vel_topic=ros2_cmd_vel_topic,
                msg_kind=ros2_cmd_vel_msg_kind,
                log_prefix="[CruzrS2]",
            )
        self._connected = False

        # --- camera (optional; mirrors unitree_go2's ROS2 backend)
        self._camera: Ros2Camera | None = None
        if camera_source == "ros2" and ros2_rgb_topic:
            self._camera = Ros2Camera(
                rgb_topic=ros2_rgb_topic,
                depth_topic=ros2_depth_topic,
                depth_scale_m=ros2_depth_scale_m,
                camera_info_topic=ros2_camera_info_topic,
                intrinsics=ros2_intrinsics,
                log_prefix="[CruzrS2]",
            )

        # --- odometry (optional; mirrors unitree_go2's ROS2 odom backend)
        self._odom: Ros2Odom | None = None
        if ros2_odom_topic:
            self._odom = Ros2Odom(
                odom_topic=ros2_odom_topic,
                msg_kind=ros2_odom_msg_kind,
                log_prefix="[CruzrS2]",
            )

    # ============================================================== lifecycle
    def connect(self) -> None:
        """Start cmd_vel publisher + camera/odom subscriptions.

        Idempotent. ROS2 side failures (missing rclpy / init error) degrade to
        None / not-running (logged), never raise — consistent with the pure-ROS2
        design: a missing rclpy affects motion+vision+odom together, so raising
        here would be indistinguishable from "no capabilities at all". Per-call
        motion errors surface at ``move_to_pose_blocking`` instead.
        """
        if self._connected:
            return
        # ROS2 side: start() returns False on missing rclpy — degrade, don't raise.
        if self._cmd_vel is not None and not self._cmd_vel.start():
            logger.warning("[CruzrS2] ROS2 cmd_vel not started — motion will raise at call time.")
        if self._camera is not None and not self._camera.start():
            logger.warning("[CruzrS2] ROS2 camera not started — vision degraded to None.")
        if self._odom is not None and not self._odom.start():
            logger.warning("[CruzrS2] ROS2 odometry not started — odom degraded to None.")
        self._connected = True
        logger.info("[CruzrS2] connected (ROS2 backend).")

    def close(self) -> None:
        """Stop cmd_vel publisher + camera/odom. Idempotent, best-effort."""
        if self._cmd_vel is not None:
            try:
                self._cmd_vel.stop()
            except Exception as e:  # best-effort motion teardown; log + continue
                logger.debug("[CruzrS2] cmd_vel stop failed during teardown: %s", e)
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception as e:  # best-effort camera teardown; log + continue
                logger.debug("[CruzrS2] camera stop failed during teardown: %s", e)
        if self._odom is not None:
            try:
                self._odom.stop()
            except Exception as e:  # best-effort odom teardown; log + continue
                logger.debug("[CruzrS2] odom stop failed during teardown: %s", e)
        self._connected = False
        logger.info("[CruzrS2] closed.")

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

    # ============================================================== motion (ROS2 cmd_vel)
    def home(self) -> None:
        """Drive the base back to the configured home (x, y, yaw). Blocking."""
        x, y, yaw = self._home_xy_yaw
        self._move_to_xy_yaw(x, y, yaw)

    def move_to_pose_blocking(self, pose: Any, *args: Any, **kwargs: Any) -> None:
        """Blocking planar move to <pose> (x, y, rz=yaw). z/rx/ry ignored.

        ``pose`` is a mapping or object with ``x``, ``y`` (meters) and optional
        ``rz`` / ``r`` (deg). The driver publishes a velocity command toward
        the target; this blocks until within tolerance or timeout. Vendor
        extensions (``sync_timeout_s``, etc.) ride in ``*args``/``**kwargs``
        after it, matching the ``RobotDriver`` Protocol signature.
        """
        x = float(getattr(pose, "x", 0.0))
        y = float(getattr(pose, "y", 0.0))
        rz = float(getattr(pose, "rz", getattr(pose, "r", 0.0)))
        self._move_to_xy_yaw(x, y, rz)

    # ============================================================== private
    def _move_to_xy_yaw(self, target_x: float, target_y: float, target_yaw_deg: float) -> None:
        """Drive toward (target_x, target_y, target_yaw) via ROS2 velocity cmds.

        Simple proportional velocity controller: command a velocity proportional
        to the remaining error, clamped to the configured max speeds, until
        within tolerance. The exact cmd_vel semantics are vendor-standard but
        the control-loop tuning is the integration seam a real deployment tunes.
        """
        if not self._connected or self._cmd_vel is None or not self._cmd_vel.is_running:
            raise RuntimeError("[CruzrS2] cmd_vel not running (call connect() first; needs rclpy).")
        for v, name in ((target_x, "x"), (target_y, "y"), (target_yaw_deg, "yaw")):
            if not math.isfinite(v):
                raise ValueError(f"[CruzrS2] non-finite {name}={v}")
        # TODO(integration): tune kp / tol / timeout per deployment. The
        # structure below is the control loop skeleton; the exact settling is
        # filled in per robot. Left as a seam so the adapter wires up end-to-end
        # without the control gains nailed down.
        logger.info(
            "[CruzrS2] move → (x=%.3f m, y=%.3f m, yaw=%.2f deg) [ROS2 cmd_vel]",
            target_x,
            target_y,
            target_yaw_deg,
        )
        # Placeholder: publish a zero-velocity settle marker. A real driver
        # commands v=clamp(kp*err, vmax) each tick and polls get_pose() until
        # within tol. Left as a seam so the adapter wires up end-to-end.
        self._cmd_vel.publish_twist(0.0, 0.0, 0.0)
