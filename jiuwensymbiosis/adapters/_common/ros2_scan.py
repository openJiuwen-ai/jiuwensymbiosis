# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""ROS2 laser-scan wrapper — robot-agnostic, mirrors ``Ros2Odom``.

Bridges the **async** ROS2 pub/sub model (messages arrive only while an
executor is spinning) to a **synchronous** ``grab_scan()`` contract: one
non-blocking call returning the latest ``LaserScan`` as a plain dict
``{ranges, angle_min, angle_max, range_max, range_min}``, or ``None``.

This dict shape is exactly what NEUPAN's ``planner.scan_to_point(state,
scan, ...)`` expects (see ``neupan/neupan.py:scan_to_point``), so a nav
driver can feed the scan straight into the planner with no conversion.

Bridge design (same as ``Ros2Odom`` / ``Ros2Camera``):
  * ``start()`` lazily imports ``rclpy`` + ``sensor_msgs``, creates a node
    with a subscription to the scan topic, then runs a
    ``SingleThreadedExecutor`` in a daemon thread.
  * The subscription callback copies the scan fields into a
    ``threading.Lock``-guarded "latest scan" slot via the pure-python
    ``_extract_scan`` helper (no rclpy needed to test — a
    ``SimpleNamespace`` works).
  * ``grab_scan()`` reads that slot without blocking; returns ``None``
    until the first message has arrived.

Lazy import of ``rclpy`` — if the package isn't installed (or the framework
interpreter isn't the ROS-blessed one), ``start()`` logs a warning and returns
False, and ``grab_scan()`` returns None. Construction never raises; failure
modes (missing package, init error) all yield ``grab_scan() -> None``. Callers
treat "no scan" the same as "no obstacles", which keeps the nav loop's fallback
chain intact — identical to ``Ros2Odom``.

**Where the scan comes from — the sensor responsibility boundary.**
This class is a pure *consumer*: it only subscribes to a topic and caches the
latest scan. It does NOT run any lidar driver / point-cloud conversion itself.
The scan published on the topic must be produced on the **robot side** by an
external lidar driver (or a depth-to-scan converter) that the integrator
deploys alongside the framework. Bring that driver up **before / independently
of** the framework, then point ``ros2_scan_topic`` at the topic it publishes.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


def _extract_scan(msg: Any) -> dict | None:
    """Read a ``sensor_msgs/LaserScan``-shaped object into a plain dict.

    Returns ``{ranges, angle_min, angle_max, range_max, range_min}`` or
    ``None`` if any required field is missing / non-numeric. Never raises.
    ``ranges`` is copied to a plain ``list[float]`` so callers (and tests)
    never hold an rclpy-typed array.
    """
    try:
        ranges_raw = getattr(msg, "ranges", None)
        if ranges_raw is None:
            return None
        ranges = [float(r) for r in ranges_raw]
        angle_min = float(msg.angle_min)
        angle_max = float(msg.angle_max)
        range_min = float(msg.range_min)
        range_max = float(msg.range_max)
    except (AttributeError, TypeError, ValueError):
        return None
    return {
        "ranges": ranges,
        "angle_min": angle_min,
        "angle_max": angle_max,
        "range_min": range_min,
        "range_max": range_max,
    }


class Ros2Scan:
    """One ROS2 laser-scan stream, exposed as a synchronous scan reader.

    Mirrors ``Ros2Odom``'s surface/lifecycle so the two are interchangeable in
    spirit behind the adapter driver.

    Lifecycle:
      * ``__init__`` only stores config.
      * ``start()`` creates the node + subscription + spin thread. Idempotent.
      * ``stop()`` tears them down. Idempotent.
      * ``grab_scan()`` returns ``None`` until ``start()`` succeeds AND the first
        scan message has arrived, then the latest scan dict per call. Never raises.
    """

    def __init__(
        self,
        scan_topic: str,
        *,
        log_prefix: str = "[ROS2]",
    ) -> None:
        self._scan_topic = scan_topic
        self._log_prefix = log_prefix
        self._node: Any = None  # rclpy.node.Node once started
        self._executor: Any = None  # SingleThreadedExecutor once started
        self._spin_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_scan: dict | None = None
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
        to ``grab_scan() -> None``.
        """
        if self._node is not None:
            return True
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from sensor_msgs.msg import LaserScan
        except ImportError:
            logger.warning(
                "%s rclpy/sensor_msgs not available — skipping laser scan. "
                "Source /opt/ros/<distro>/setup.bash (under a ROS2-compatible "
                "interpreter) to enable.",
                self._log_prefix,
            )
            return False
        try:
            if not rclpy.ok():
                rclpy.init(args=None)
                self._owns_rclpy = True
            self._node = Node("jiuwensymbiosis_ros2_scan")
            self._node.create_subscription(LaserScan, self._scan_topic, self._on_scan, 10)
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._spin_thread = threading.Thread(
                target=self._executor.spin,
                name="ros2_scan_spin",
                daemon=True,
            )
            self._spin_thread.start()
            logger.info(
                "%s ROS2 laser scan ready (topic=%s).",
                self._log_prefix,
                self._scan_topic,
            )
            return True
        except Exception as e:
            logger.warning(
                "%s ROS2 laser scan init failed (%s); continuing without scan.",
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
    def _on_scan(self, msg: Any) -> None:
        scan = _extract_scan(msg)
        if scan is None:
            return
        with self._lock:
            self._latest_scan = scan

    # -------------------------------------------------------------- scan grab
    def grab_scan(self) -> dict | None:
        """Return the latest scan dict, or ``None``.

        Non-blocking. Returns ``None`` if no scan message has arrived yet
        (e.g. before ``start()`` or before any message is received).

        Dict schema (raw ROS units — radians + meters):

            {"ranges": list[float], "angle_min": float, "angle_max": float,
             "range_min": float, "range_max": float}
        """
        with self._lock:
            return self._latest_scan

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
            self._latest_scan = None
