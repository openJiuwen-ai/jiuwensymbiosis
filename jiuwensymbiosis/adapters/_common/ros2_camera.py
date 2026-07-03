# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""ROS2 camera wrapper — robot-agnostic, same surface as ``RealSenseCamera``.

Bridges the **async** ROS2 pub/sub model (messages arrive only while an
executor is spinning) to the **synchronous** ``grab_frames()`` contract that
``CameraDriver`` (and thus ``get_observation`` / ``grab_rgb`` / ``get_image``)
expect: one non-blocking call returning the latest ``(rgb_uint8, depth_m)``
pair, or ``None``.

Bridge design:
  * ``start()`` lazily imports ``rclpy`` + ``sensor_msgs``, creates a node with
    subscriptions to an RGB topic (and optional depth / camera_info topics),
    then runs a ``SingleThreadedExecutor`` in a daemon thread.
  * Each subscription callback converts the incoming ``sensor_msgs/Image`` to a
    numpy array via the pure-numpy ``_ros_to_rgb`` / ``_ros_to_depth_m``
    helpers (no ``cv_bridge`` — it is broken against numpy>=2 in this env) and
    stores it in a ``threading.Lock``-guarded "latest frame" slot.
  * ``grab_frames()`` reads those slots without blocking; returns ``None``
    until both an RGB and a depth frame have arrived (and their shapes match).

Lazy import of ``rclpy`` — if the package isn't installed (or the framework
interpreter isn't the ROS-blessed one), ``start()`` logs a warning and returns
False, and ``grab_frames()`` returns None. Construction never raises; failure
modes (missing package, init error) all yield ``grab_frames() -> None``.
Callers treat "no camera" the same as "no frames", which keeps the
"ok=False, reason=no_camera" fallback chain intact — identical to
``RealSenseCamera``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Pure-numpy encoding converters (no cv_bridge — it conflicts with numpy>=2).
# ``msg`` is a ``sensor_msgs.msg.Image``-shaped object: we only read
# ``encoding`` / ``data`` / ``height`` / ``width``, so a ``SimpleNamespace``
# works in tests without rclpy.
# =============================================================================
def _ros_to_rgb(msg: Any) -> np.ndarray | None:
    """Convert a sensor_msgs/Image (bgr8/rgb8/rgba8/bgra8) to HxWx3 RGB uint8.

    Returns None for unrecognized encodings or malformed messages (missing
    attributes, byte length not matching ``height*width*channels``). Never raises.
    """
    enc = str(getattr(msg, "encoding", "")).lower()
    try:
        h = int(msg.height)
        w = int(msg.width)
        arr = np.frombuffer(msg.data, dtype=np.uint8)
    except (AttributeError, TypeError, ValueError):
        return None
    try:
        if enc == "rgb8":
            return arr.reshape(h, w, 3).copy()
        if enc == "bgr8":
            bgr = arr.reshape(h, w, 3)
            return bgr[:, :, ::-1].copy()  # match RealSenseCamera's BGR→RGB
        if enc in ("rgba8", "bgra8"):
            ch = arr.reshape(h, w, 4)[:, :, :3]  # drop alpha
            if enc == "bgra8":
                return ch[:, :, ::-1].copy()
            return ch.copy()
    except ValueError:
        # Buffer length doesn't match h*w*channels — malformed frame, skip it.
        return None
    return None


def _ros_to_depth_m(msg: Any, depth_scale_m: float) -> np.ndarray | None:
    """Convert a sensor_msgs/Image depth to HxW float32 meters.

    Handles ``16uc1`` (integer depth; multiply by ``depth_scale_m`` — typically
    0.001 for millimetre devices like RealSense) and ``32fc1`` (already meters).
    Returns None for unrecognized encodings or malformed messages. Never raises.
    """
    enc = str(getattr(msg, "encoding", "")).lower()
    try:
        h = int(msg.height)
        w = int(msg.width)
        scale = float(depth_scale_m)
        if enc == "16uc1":
            raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
            return raw.astype(np.float32) * scale
        if enc == "32fc1":
            return np.frombuffer(msg.data, dtype=np.float32).reshape(h, w).copy()
    except (AttributeError, TypeError, ValueError):
        # Missing attribute, non-numeric scale, or buffer/shape mismatch.
        return None
    return None


class Ros2Camera:
    """One ROS2 image stream pair (RGB + optional depth), exposed as a camera.

    Mirrors ``RealSenseCamera``'s surface so the two are drop-in interchangeable
    behind ``CameraDriver`` and ``PiperLowLevel``'s ``camera_source`` switch.

    Lifecycle:
      * ``__init__`` only stores config.
      * ``start()`` creates the node + subscriptions + spin thread. Idempotent.
      * ``stop()`` tears them down. Idempotent.
      * ``grab_frames()`` returns ``None`` until ``start()`` succeeds AND both an
        RGB and a depth frame have arrived, then the latest ``(rgb, depth_m)``
        per call. Never raises.

    Intrinsics come from either a live ``CameraInfo`` subscription
    (``camera_info_topic``) or an explicit 3x3 matrix passed at construction
    (``intrinsics`` — e.g. loaded from a calibration file). The property
    returns whichever was most recently available.
    """

    def __init__(
        self,
        rgb_topic: str,
        depth_topic: str | None = None,
        *,
        depth_scale_m: float = 0.001,
        camera_info_topic: str | None = None,
        intrinsics: list[float] | np.ndarray | None = None,
        log_prefix: str = "[ROS2]",
    ) -> None:
        self._rgb_topic = rgb_topic
        self._depth_topic = depth_topic
        self._depth_scale_m = float(depth_scale_m)
        self._camera_info_topic = camera_info_topic
        self._log_prefix = log_prefix
        self._intrinsics: np.ndarray | None = None
        if intrinsics is not None:
            # Construction must never raise (parity with RealSenseCamera): a malformed
            # intrinsics (len != 9) degrades to None + warning rather than ValueError.
            try:
                self._intrinsics = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
            except (ValueError, TypeError) as e:
                logger.warning(
                    "%s ignoring malformed intrinsics (expected 9 elements, got %s): %s",
                    self._log_prefix,
                    np.asarray(intrinsics, dtype=object).shape,
                    e,
                )
                self._intrinsics = None

        self._node: Any = None  # rclpy.node.Node once started
        self._executor: Any = None  # SingleThreadedExecutor once started
        self._spin_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_rgb: np.ndarray | None = None
        self._latest_depth_m: np.ndarray | None = None
        self._owns_rclpy = False  # whether WE called rclpy.init() (so we shutdown)

    # ----------------------------------------------------------------- state
    @property
    def is_running(self) -> bool:
        """True once ``start()`` has created the node + spin thread."""
        return self._node is not None

    @property
    def intrinsics(self) -> np.ndarray | None:
        """3x3 K matrix. From CameraInfo subscription or constructor arg."""
        return self._intrinsics

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> bool:
        """Create the node + subscriptions and start the spin thread.

        Idempotent. Returns True on success, False on any failure (with a
        warning logged). Never raises — missing rclpy / init errors all degrade
        to ``grab_frames() -> None``.
        """
        if self._node is not None:
            return True
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from sensor_msgs.msg import CameraInfo, Image
        except ImportError:
            logger.warning(
                "%s rclpy/sensor_msgs not available — skipping camera. "
                "Source /opt/ros/<distro>/setup.bash (under a ROS2-compatible "
                "interpreter) to enable.",
                self._log_prefix,
            )
            return False
        try:
            if not rclpy.ok():
                rclpy.init(args=None)
                self._owns_rclpy = True
            self._node = Node("jiuwensymbiosis_ros2_camera")
            self._node.create_subscription(Image, self._rgb_topic, self._on_rgb, 10)
            if self._depth_topic:
                self._node.create_subscription(Image, self._depth_topic, self._on_depth, 10)
            if self._camera_info_topic:
                self._node.create_subscription(CameraInfo, self._camera_info_topic, self._on_camera_info, 10)
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._spin_thread = threading.Thread(target=self._executor.spin, name="ros2_camera_spin", daemon=True)
            self._spin_thread.start()
            logger.info(
                "%s ROS2 camera ready (rgb=%s, depth=%s, camera_info=%s).",
                self._log_prefix,
                self._rgb_topic,
                self._depth_topic or "(none)",
                self._camera_info_topic or "(none)",
            )
            return True
        except Exception as e:
            logger.warning(
                "%s ROS2 camera init failed (%s); continuing without camera.",
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
    def _on_rgb(self, msg: Any) -> None:
        rgb = _ros_to_rgb(msg)
        if rgb is None:
            return
        with self._lock:
            self._latest_rgb = rgb

    def _on_depth(self, msg: Any) -> None:
        depth = _ros_to_depth_m(msg, self._depth_scale_m)
        if depth is None:
            return
        with self._lock:
            self._latest_depth_m = depth

    def _on_camera_info(self, msg: Any) -> None:
        k = getattr(msg, "k", None)
        if k is None or len(k) != 9:
            return
        with self._lock:
            self._intrinsics = np.asarray(k, dtype=np.float64).reshape(3, 3)

    # -------------------------------------------------------------- frame grab
    def grab_frames(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the latest ``(rgb_uint8, depth_m_float32)`` pair, or None.

        Non-blocking. Returns None if no RGB or depth frame has arrived yet
        (e.g. before ``start()`` or before any message is received), or if the
        RGB and depth resolutions don't match (they are expected to be aligned
        externally — typically via ``aligned_depth_to_color``).
        """
        with self._lock:
            rgb = self._latest_rgb
            depth = self._latest_depth_m
        if rgb is None or depth is None:
            return None
        if rgb.shape[:2] != depth.shape:
            return None
        return rgb, depth

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
            self._latest_rgb = None
            self._latest_depth_m = None
