# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""ROS2 velocity-command publisher — robot-agnostic, mirrors ``Ros2Odom``.

Bridges a **synchronous** ``publish_twist(vx, vy, wz)`` write contract to the
**async** ROS2 pub/sub model (a publisher only flushes while an executor is
spinning): one non-blocking call publishing a ``geometry_msgs/Twist`` (or
``TwistStamped``) on a configurable topic. Returns True on publish, False if
the publisher isn't running (rclpy missing / start() failed). Never raises.

Bridge design (same as ``Ros2Odom`` / ``Ros2Camera`` / ``Ros2Scan``):
  * ``start()`` lazily imports ``rclpy`` + the configured message class, creates
    a node with a publisher on the cmd_vel topic, then runs a
    ``SingleThreadedExecutor`` in a daemon thread.
  * ``publish_twist(vx, vy, wz)`` fills ``linear.x/y`` + ``angular.z`` and
    publishes; ``TwistStamped`` wraps the same ``Twist`` under ``.twist``.

The message type is chosen by ``msg_kind`` (the "Twist 等" the config asks for):
  * ``"twist"``         → ``geometry_msgs/msg/Twist``
  * ``"twist_stamped"`` → ``geometry_msgs/msg/TwistStamped`` (same Twist + header)

Lazy import of ``rclpy`` — if the package isn't installed (or the framework
interpreter isn't the ROS-blessed one), ``start()`` logs a warning and returns
False, and ``publish_twist()`` returns False. Construction never raises;
failure modes (missing package, init error, unknown ``msg_kind``) all yield
``publish_twist() -> False``. Callers treat "no publisher" the same as "no
motion", which keeps the nav loop's fallback chain intact.

**Where the velocity goes — the motion-execution boundary.**
This class is a pure *writer*: it only publishes a Twist on a topic. It does
NOT execute the motion itself. The chassis must respond to that topic via an
external bridge the integrator deploys alongside the framework — e.g. a Go2
``cmd_vel_bridge`` that translates ``/cmd_vel`` into the vendor SDK's sport
command. Bring that bridge up **before / independently of** the framework,
then point ``ros2_cmd_vel_topic`` at the topic it reads.
"""

from __future__ import annotations

import importlib
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


# ``msg_kind`` → (ROS2 module, message type name). Add a row here to support
# another velocity-carrying message type.
_CMD_VEL_MSG_KINDS: dict[str, tuple[str, str]] = {
    "twist": ("geometry_msgs.msg", "Twist"),
    "twist_stamped": ("geometry_msgs.msg", "TwistStamped"),
}


class Ros2CmdVel:
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
        to ``publish_twist() -> False``.
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
