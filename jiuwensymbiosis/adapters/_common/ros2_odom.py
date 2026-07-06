# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""ROS2 odometry wrapper — robot-agnostic, mirrors ``Ros2Camera``.

Bridges the **async** ROS2 pub/sub model (messages arrive only while an
executor is spinning) to a **synchronous** ``grab_pose()`` contract: one
non-blocking call returning the latest position + orientation (meters +
quaternion xyzw, plus a convenience ``yaw_deg``), or ``None``.

Bridge design (same as ``Ros2Camera``):
  * ``start()`` lazily imports ``rclpy`` + the configured message class,
    creates a node with a subscription to the odom topic, then runs a
    ``SingleThreadedExecutor`` in a daemon thread.
  * The subscription callback converts the incoming message to a canonical
    ``(xyz_m, quat_xyzw)`` numpy pair via the pure-python ``_extract_*``
    helpers (no rclpy needed to test — a ``SimpleNamespace`` works) and
    stores it in a ``threading.Lock``-guarded "latest pose" slot.
  * ``grab_pose()`` reads that slot without blocking; returns ``None``
    until the first message has arrived.

The message type is configurable via ``msg_kind`` (the "Odometry 等" the
config asks for):
  * ``"odometry"``                     → ``nav_msgs/msg/Odometry``
                                         (pose under ``msg.pose.pose``)
  * ``"pose_stamped"``                 → ``geometry_msgs/msg/PoseStamped``
                                         (pose under ``msg.pose``)
  * ``"pose_with_covariance_stamped"`` → ``geometry_msgs/msg/PoseWithCovarianceStamped``
                                         (pose under ``msg.pose.pose``)
Each carries ``position.{x,y,z}`` + ``orientation.{x,y,z,w}`` under one of
those nesting shapes; the extractors flatten it to the same canonical pair.

Lazy import of ``rclpy`` — if the package isn't installed (or the framework
interpreter isn't the ROS-blessed one), ``start()`` logs a warning and returns
False, and ``grab_pose()`` returns None. Construction never raises; failure
modes (missing package, init error, unknown ``msg_kind``) all yield
``grab_pose() -> None``. Callers treat "no odom" the same as "no pose", which
keeps the observation fallback chain intact — identical to ``Ros2Camera``.

**Where the odom topic comes from — the SLAM responsibility boundary.**
This class is a pure *consumer*: it only subscribes to a topic and caches the
latest pose. It does NOT run any localization / mapping / pose-estimation
itself. The pose published on the odom topic must be produced on the **robot
side** by an external SLAM / odometry stack that the integrator deploys
alongside the framework — for example:

  * a LiDAR SLAM node (cartographer, slam_toolbox, FAST-LIO, LIO-SAM, …),
  * a visual-inertial odometry node (VINS-Fusion, ORB-SLAM3, RTAB-Map, …),
  * a wheel-encoder + IMU fusion node (robot_localization EKF),
  * or any other node that publishes ``nav_msgs/Odometry`` /
    ``geometry_msgs/PoseStamped`` / ``PoseWithCovarianceStamped``.

Bring that stack up **before / independently of** the framework (launch its
nodes, feed it the robot's sensor topics, let it converge), then point
``ros2_odom_topic`` at the topic it publishes. If SLAM isn't running or hasn't
converged, no message is published and ``grab_pose()`` simply returns ``None``
— the framework keeps working, just without an external pose feed (same
"no data" fallback as a missing camera).
"""

from __future__ import annotations

import importlib
import logging
import math
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Pure-python pose extractors (no rclpy — accept any object with the right
# attributes, so a ``SimpleNamespace`` works in tests). Each returns a
# canonical ``(xyz_m: np.ndarray(3,) float64, quat_xyzw: np.ndarray(4,) float64)``
# pair, or ``None`` on missing/malformed fields. Never raises.
# =============================================================================
def _quat_to_yaw_deg(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw (deg) from a quaternion (x, y, z, w) — planar-robot heading.

    Pure-``math`` (no scipy) so the cross-vendor module stays dependency-light;
    the standard formula ``yaw = atan2(2(wz+xy), 1 - 2(yy+zz))``.
    """
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def _extract_pose(pose: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Read position + orientation off a ``geometry_msgs/Pose``-shaped object.

    Returns ``(xyz_m, quat_xyzw)`` or ``None`` if any field is missing / the
    quaternion components aren't numeric. Never raises.
    """
    pos = getattr(pose, "position", None)
    orn = getattr(pose, "orientation", None)
    if pos is None or orn is None:
        return None
    try:
        xyz = np.array(
            [float(pos.x), float(pos.y), float(pos.z)],
            dtype=np.float64,
        )
        quat = np.array(
            [float(orn.x), float(orn.y), float(orn.z), float(orn.w)],
            dtype=np.float64,
        )
    except (AttributeError, TypeError, ValueError):
        return None
    return xyz, quat


def _extract_odometry(msg: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """``nav_msgs/Odometry``: pose lives under ``msg.pose.pose``."""
    pose_with_cov = getattr(msg, "pose", None)
    if pose_with_cov is None:
        return None
    return _extract_pose(getattr(pose_with_cov, "pose", None))


def _extract_pose_stamped(msg: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """``geometry_msgs/PoseStamped``: pose lives under ``msg.pose``."""
    return _extract_pose(getattr(msg, "pose", None))


def _extract_pose_with_covariance_stamped(msg: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """``geometry_msgs/PoseWithCovarianceStamped``: pose under ``msg.pose.pose``."""
    pwc = getattr(msg, "pose", None)
    if pwc is None:
        return None
    return _extract_pose(getattr(pwc, "pose", None))


# ``msg_kind`` → (ROS2 module, message type name, extractor). Add a row here
# to support another pose-carrying message type.
_MSG_KINDS: dict[str, tuple[str, str, Any]] = {
    "odometry": ("nav_msgs.msg", "Odometry", _extract_odometry),
    "pose_stamped": ("geometry_msgs.msg", "PoseStamped", _extract_pose_stamped),
    "pose_with_covariance_stamped": (
        "geometry_msgs.msg",
        "PoseWithCovarianceStamped",
        _extract_pose_with_covariance_stamped,
    ),
}


class Ros2Odom:
    """One ROS2 odometry stream, exposed as a synchronous pose reader.

    Mirrors ``Ros2Camera``'s surface/lifecycle so the two are interchangeable
    in spirit behind the piper (and future) adapters.

    Lifecycle:
      * ``__init__`` only stores config.
      * ``start()`` creates the node + subscription + spin thread. Idempotent.
      * ``stop()`` tears them down. Idempotent.
      * ``grab_pose()`` returns ``None`` until ``start()`` succeeds AND the first
        odom message has arrived, then the latest pose dict per call. Never raises.

    The message type is chosen by ``msg_kind`` (see ``_MSG_KINDS``). The topic
    is optional — a driver that doesn't configure one simply never constructs
    a ``Ros2Odom``, and ``grab_pose()`` on a never-started instance is ``None``.
    """

    def __init__(
        self,
        odom_topic: str,
        *,
        msg_kind: str = "odometry",
        log_prefix: str = "[ROS2]",
    ) -> None:
        self._odom_topic = odom_topic
        self._log_prefix = log_prefix
        # Construction must never raise (parity with Ros2Camera): an unknown
        # msg_kind degrades to "odometry" + warning rather than ValueError.
        self._msg_kind = msg_kind
        if msg_kind not in _MSG_KINDS:
            logger.warning(
                "%s unknown msg_kind=%r (expected one of %s); falling back to 'odometry'.",
                self._log_prefix,
                msg_kind,
                sorted(_MSG_KINDS),
            )
            self._msg_kind = "odometry"

        self._node: Any = None  # rclpy.node.Node once started
        self._executor: Any = None  # SingleThreadedExecutor once started
        self._spin_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_xyz: np.ndarray | None = None
        self._latest_quat: np.ndarray | None = None
        self._owns_rclpy = False  # whether WE called rclpy.init() (so we shutdown)

    # ----------------------------------------------------------------- state
    @property
    def is_running(self) -> bool:
        """True once ``start()`` has created the node + spin thread."""
        return self._node is not None

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> bool:
        """Create the node + subscription and start the spin thread.

        Idempotent. Returns True on success, False on any failure (with a
        warning logged). Never raises — missing rclpy / init errors all degrade
        to ``grab_pose() -> None``.
        """
        if self._node is not None:
            return True
        mod_name, type_name, _ = _MSG_KINDS[self._msg_kind]
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node

            msg_mod = importlib.import_module(mod_name)
            msg_type = getattr(msg_mod, type_name)
        except ImportError:
            logger.warning(
                "%s rclpy/%s not available — skipping odometry. "
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
            self._node = Node("jiuwensymbiosis_ros2_odom")
            self._node.create_subscription(msg_type, self._odom_topic, self._on_odom, 10)
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._spin_thread = threading.Thread(target=self._executor.spin, name="ros2_odom_spin", daemon=True)
            self._spin_thread.start()
            logger.info(
                "%s ROS2 odometry ready (topic=%s, msg_kind=%s).",
                self._log_prefix,
                self._odom_topic,
                self._msg_kind,
            )
            return True
        except Exception as e:
            logger.warning(
                "%s ROS2 odometry init failed (%s); continuing without odometry.",
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

    # ------------------------------------------------------------ subscription cb
    def _on_odom(self, msg: Any) -> None:
        _, _, extractor = _MSG_KINDS[self._msg_kind]
        result = extractor(msg)
        if result is None:
            return
        xyz, quat = result
        with self._lock:
            self._latest_xyz = xyz
            self._latest_quat = quat

    # -------------------------------------------------------------- pose grab
    def grab_pose(self) -> dict | None:
        """Return the latest pose as a dict, or ``None``.

        Non-blocking. Returns ``None`` if no odom message has arrived yet
        (e.g. before ``start()`` or before any message is received).

        Dict schema (raw ROS units — meters + quaternion xyzw, plus a
        convenience ``yaw_deg`` for planar-robot heading):

            {"x","y","z"} (m), {"qx","qy","qz","qw"}, "yaw_deg" (deg)
        """
        with self._lock:
            xyz = self._latest_xyz
            quat = self._latest_quat
        if xyz is None or quat is None:
            return None
        qx, qy, qz, qw = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        return {
            "x": float(xyz[0]),
            "y": float(xyz[1]),
            "z": float(xyz[2]),
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "qw": qw,
            "yaw_deg": _quat_to_yaw_deg(qx, qy, qz, qw),
        }

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
        self._executor = None
        self._spin_thread = None
        with self._lock:
            self._latest_xyz = None
            self._latest_quat = None
