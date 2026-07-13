# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Low-level Unitree Go2 driver — quadruped mobile base (mecanum kinematics).

Route A navigation (NEUPAN + ROS2 cmd_vel):
  * **Chassis motion** → the NEUPAN planner computes ``(vx, vy, omega)`` for the
    mecanum kinematics (3-dim action — vy is real, unlike a diff base); this
    driver publishes it on the cmd_vel topic via ``Ros2CmdVel``. A robot-side
    ``cmd_vel_bridge`` (deployed by the integrator, NOT in the framework) turns
    the Twist into the SDK sport command. NEUPAN is user-installed (lazy import
    in ``connect()``); missing → ``move_to_pose_blocking`` raises with install
    guidance (the nav path's primary capability).
  * **Images** → ``Ros2Camera`` (reused from ``adapters/_common``). Lazy rclpy;
    missing → ``grab_frames()`` returns None (vision degrades, base still moves).
  * **Odometry** → ``Ros2Odom`` (reused from ``adapters/_common``). Lazy rclpy;
    missing → ``get_odom_pose()`` returns None.
  * **Laser scan** → ``Ros2Scan`` (reused from ``adapters/_common``). Lazy rclpy;
    missing → ``get_scan()`` returns None. Required by the NEUPAN nav loop.

The ``unitree_sdk2py`` path (Cyclone DDS) is **softened**: the SDK is still
initialized when present (kept for a future SDK-direct motion path), but a
missing SDK no longer raises — it degrades to a warning, because the
NEUPAN/cmd_vel nav path does not need it.

Frame conventions:
  * The base pose (``get_pose``) is the 2D planar pose from odometry:
    ``x, y`` (meters, base/map frame) + ``yaw`` (deg). It is wrapped into a
    ``SimpleNamespace(x, y, z=0, rx=0, ry=0, rz=yaw_deg)`` to satisfy the
    ``RobotDriver`` 6-DoF pose shape — the Env verb ``get_flange_pose`` only
    passes it through; the Api layer decides what the numbers mean.
  * NEUPAN speaks its own state convention ``[x, y, theta]`` with **theta in
    radians** (our odom ``yaw_deg`` is converted at the nav-loop boundary).
  * ``tool_offset_mm`` is 0.0 (a base has no flange→tip offset).

Construction never raises (parity with ``Ros2Camera`` / ``Ros2Odom``). ROS2
side failures degrade to "not started"; a missing neupan package raises in
``connect()`` (nav's primary capability) — distinct from per-call motion
errors, which surface at ``move_to_pose_blocking``.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from jiuwensymbiosis.adapters._common.ros2_camera import Ros2Camera
from jiuwensymbiosis.adapters._common.ros2_cmd_vel import Ros2CmdVel
from jiuwensymbiosis.adapters._common.ros2_odom import Ros2Odom
from jiuwensymbiosis.adapters._common.ros2_scan import Ros2Scan

logger = logging.getLogger(__name__)

# ``ChannelFactoryInitialize`` is a process-wide one-shot Cyclone DDS init:
# it binds a DDS participant to the host NIC. Re-invoking on an already-
# initialized participant errors out. Tracked here so multiple driver
# instances (or reconnect after ``disconnect``) skip the redundant call.
# Only a process restart resets DDS (the SDK has no clean shutdown).
_dds_factory_initialized: bool = False


class UnitreeGo2Driver:
    """Unitree Go2 mobile-base driver: NEUPAN nav + ROS2 cmd_vel/camera/odom.

    Route A navigation: the NEUPAN planner computes ``(vx, vy, omega)`` for the
    mecanum kinematics, this driver publishes it on the cmd_vel topic, and a
    robot-side ``cmd_vel_bridge`` turns it into the SDK sport command. The SDK
    path (``unitree_sdk2py``) is **softened**: present → initialized, missing →
    degraded to a warning (it is not needed for the NEUPAN/cmd_vel nav path).

    Implements the ``RobotDriver`` Protocol (motion + pose + close) plus the
    ``CameraDriver`` sibling (``intrinsics`` / ``grab_frames``) and
    ``get_odom_pose`` / ``get_scan`` accessors.
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
        # ROS2 velocity command (optional; required by the NEUPAN nav loop)
        ros2_cmd_vel_topic: str | None = None,
        ros2_cmd_vel_msg_kind: str = "twist",
        # ROS2 laser scan (optional; required by the NEUPAN nav loop)
        ros2_scan_topic: str | None = None,
        # NEUPAN nav (optional; None → move raises at call time with install guidance)
        neupan_config_path: str | None = None,
        neupan_arrive_threshold_m: float = 0.1,
        neupan_max_collision_count: int = 100,
        neupan_control_hz: float = 10.0,
    ) -> None:
        self._lock = threading.RLock()
        self._max_linear = float(max_linear_speed_mps)
        self._max_angular = float(max_angular_speed_radps)
        self._neupan_config_path = neupan_config_path
        self._neupan_arrive_threshold_m = float(neupan_arrive_threshold_m)
        self._neupan_max_collision_count = int(neupan_max_collision_count)
        self._neupan_control_hz = float(neupan_control_hz)
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

        # --- velocity command publisher (optional; required by the NEUPAN nav
        #     loop). Route A: NEUPAN computes (vx, vy, omega), this publishes the
        #     Twist to cmd_vel; a robot-side ``cmd_vel_bridge`` turns it into the
        #     SDK sport command. SDK motion is NOT used for navigation.
        self._cmd_vel: Ros2CmdVel | None = None
        if ros2_cmd_vel_topic:
            self._cmd_vel = Ros2CmdVel(
                cmd_vel_topic=ros2_cmd_vel_topic,
                msg_kind=ros2_cmd_vel_msg_kind,
                log_prefix="[Go2]",
            )

        # --- laser scan (optional; required by the NEUPAN nav loop)
        self._scan: Ros2Scan | None = None
        if ros2_scan_topic:
            self._scan = Ros2Scan(scan_topic=ros2_scan_topic, log_prefix="[Go2]")

        # --- NEUPAN planner (lazily built in connect() so __init__ stays cheap
        #     and never imports torch/cvxpy at construction time)
        self._planner: Any = None  # neupan.neupan instance once connect()

    # ============================================================== lifecycle
    def connect(self) -> None:
        """Start ROS2 subscriptions + (softened) SDK init + build NEUPAN planner.

        Idempotent. ROS2 side failures (missing rclpy) degrade to None / not
        running, never raise. The SDK init is **softened**: a missing
        ``unitree_sdk2py`` no longer raises (the SDK motion path is unavailable,
        but the NEUPAN/cmd_vel nav path works without it). NEUPAN import also
        happens here; a missing neupan package raises (motion's primary
        capability for the nav path), with a clear install message.
        """
        if self._connected:
            return
        # ROS2 side: start() returns False on missing rclpy — degrade, don't raise.
        if self._camera is not None and not self._camera.start():
            logger.warning("[Go2] ROS2 camera not started — vision degraded to None.")
        if self._odom is not None and not self._odom.start():
            logger.warning("[Go2] ROS2 odometry not started — odom degraded to None.")
        if self._cmd_vel is not None and not self._cmd_vel.start():
            logger.warning("[Go2] ROS2 cmd_vel not started — nav will raise at call time.")
        if self._scan is not None and not self._scan.start():
            logger.warning("[Go2] ROS2 laser scan not started — scan degraded to None.")

        # SDK side: softened. Missing unitree_sdk2py no longer raises — the SDK
        # motion path is just unavailable, while the NEUPAN/cmd_vel nav path
        # works without it. The SDK init code is retained for the future
        # SDK-direct motion path (still a TODO seam in _move_to_xy_yaw's docstring).
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        except ImportError:
            logger.warning(
                "[Go2] unitree_sdk2py not installed — SDK motion path unavailable; "
                "NEUPAN/cmd_vel nav path still works. "
                'Install via `pip install -e ".[unitree]"` to enable SDK motion.'
            )
        else:
            try:
                global _dds_factory_initialized
                if not _dds_factory_initialized:
                    ChannelFactoryInitialize(0, self._network_interface)
                    _dds_factory_initialized = True
                self._sdk = True  # marker: DDS participant ready
                logger.info(
                    "[Go2] chassis SDK initialized (interface=%s).",
                    self._network_interface or "(default)",
                )
            except Exception as exc:  # SDK init failure is non-fatal now (nav doesn't need it)
                logger.warning("[Go2] chassis SDK init failed (%s) — SDK motion path unavailable.", exc)
                self._sdk = None

        # NEUPAN planner: lazily built only when a config path is configured.
        # Missing neupan package is a hard error for the nav path (motion's
        # primary capability under route A) — same posture as ubetech.
        if self._neupan_config_path is not None:
            try:
                from neupan import neupan as _neupan_mod
            except ImportError as exc:
                raise RuntimeError(
                    "[Go2] neupan not installed. Install from your local NeuPAN "
                    "repository (`pip install -e .`; pulls torch/cvxpy/scipy). "
                    'See README "NEUPAN navigation".'
                ) from exc
            try:
                self._planner = _neupan_mod.init_from_yaml(self._neupan_config_path)
                logger.info("[Go2] NEUPAN planner loaded (config=%s).", self._neupan_config_path)
            except Exception as exc:  # init_from_yaml failure is a real motion error
                raise RuntimeError(f"[Go2] NEUPAN planner init failed: {exc}") from exc
        self._connected = True

    def close(self) -> None:
        """Stop cmd_vel/camera/odom/scan + release SDK. Idempotent, best-effort."""
        if self._cmd_vel is not None:
            try:
                self._cmd_vel.stop()
            except Exception as e:  # best-effort cmd_vel teardown; log + continue
                logger.debug("[Go2] cmd_vel stop failed during teardown: %s", e)
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
        if self._scan is not None:
            try:
                self._scan.stop()
            except Exception as e:  # best-effort scan teardown; log + continue
                logger.debug("[Go2] scan stop failed during teardown: %s", e)
        self._planner = None
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

    def get_scan(self) -> dict | None:
        """Latest ROS2 laser scan dict, or None (no scan backend / no message)."""
        return self._scan.grab_scan() if self._scan is not None else None

    # ============================================================== camera
    def grab_frames(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Grab (rgb, depth_m) from the ROS2 camera, or None if no camera."""
        return self._camera.grab_frames() if self._camera is not None else None

    # ============================================================== motion (NEUPAN + ROS2 cmd_vel)
    def home(self) -> None:
        """Drive the base back to the configured home (x, y, yaw). Blocking."""
        x, y, yaw = self._home_xy_yaw
        self._move_to_xy_yaw(x, y, yaw)

    def move_to_pose_blocking(self, pose: Any, *args: Any, **kwargs: Any) -> None:
        """Blocking planar move to <pose> (x, y, rz=yaw). z/rx/ry ignored.

        ``pose`` is a mapping or object with ``x``, ``y`` (meters) and optional
        ``rz`` / ``r`` (deg). Route A: the NEUPAN planner computes the velocity,
        this driver publishes it via the cmd_vel topic, and a robot-side
        ``cmd_vel_bridge`` turns it into the SDK sport command. Vendor extensions
        (``sync_timeout_s``, etc.) ride in ``*args``/``**kwargs`` after it,
        matching the ``RobotDriver`` Protocol signature.
        """
        x = float(getattr(pose, "x", 0.0))
        y = float(getattr(pose, "y", 0.0))
        rz = float(getattr(pose, "rz", getattr(pose, "r", 0.0)))
        self._move_to_xy_yaw(x, y, rz)

    # ============================================================== private
    def _move_to_xy_yaw(self, target_x: float, target_y: float, target_yaw_deg: float) -> None:
        """Drive toward (target_x, target_y) via the NEUPAN planner + cmd_vel.

        Mecanum-base navigation loop: each tick read odom + scan, ask the NEUPAN
        planner for ``(vx, vy, omega)`` (3-dim action for the mecanum kinematics
        — vy is NOT zero, unlike a diff base), clamp to the configured max
        speeds, and publish via cmd_vel. The loop exits when NEUPAN reports
        ``arrive`` OR the pure distance to the goal drops below
        ``neupan_arrive_threshold_m`` (the distance check supplements NEUPAN's
        own ``check_curve_arrive``, whose path-index condition often fails
        after obstacle-avoidance deviation — same lesson as
        ``go2_point_nav.py``). ``target_yaw_deg`` is accepted for the ``home()``
        call shape but NEUPAN derives the goal heading itself from
        ``atan2(dy, dx)``.

        Raises ``RuntimeError`` if cmd_vel isn't running, the planner isn't
        built, or collisions persist past ``neupan_max_collision_count``.
        """
        if not self._connected or self._cmd_vel is None or not self._cmd_vel.is_running:
            raise RuntimeError("[Go2] cmd_vel not running (call connect() first; needs rclpy).")
        if self._planner is None:
            raise RuntimeError(
                "[Go2] NEUPAN planner not built (no neupan_config_path; "
                "set it in config + `pip install -e .` in your NeuPAN repo)."
            )
        for v, name in ((target_x, "x"), (target_y, "y"), (target_yaw_deg, "yaw")):
            if not math.isfinite(v):
                raise ValueError(f"[Go2] non-finite {name}={v}")

        logger.info(
            "[Go2] nav → (x=%.3f m, y=%.3f m) [NEUPAN + cmd_vel]",
            target_x,
            target_y,
        )
        try:
            self._nav_loop(target_x, target_y)
        finally:
            # Unconditional stop: never leave a velocity hanging after the loop,
            # whether it exited via arrival, collision-cap, or exception.
            self._cmd_vel.publish_twist(0.0, 0.0, 0.0)

    def _nav_loop(self, target_x: float, target_y: float) -> None:
        """NEUPAN control loop (mecanum 3-dim). Caller guarantees cmd_vel + planner."""
        planner = self._planner
        cmd_vel = self._cmd_vel  # bound locally: caller (_move_to_xy_yaw) guards non-None
        if cmd_vel is None:
            raise RuntimeError("[Go2] cmd_vel not running (call connect() first; needs rclpy).")
        period = 1.0 / self._neupan_control_hz if self._neupan_control_hz > 0 else 0.1
        threshold = self._neupan_arrive_threshold_m
        max_collision = self._neupan_max_collision_count

        # Wait for an initial pose so the planner has a real start state.
        start = self._odom_state()
        if start is None:
            raise RuntimeError("[Go2] no odometry yet — cannot start nav (start SLAM/odom first).")

        # Goal heading follows the start→goal direction.
        goal_theta = math.atan2(target_y - start[1, 0], target_x - start[0, 0])
        goal = np.array([[target_x], [target_y], [goal_theta]])
        planner.update_initial_path_from_goal(start, goal)
        # Override NEUPAN's threshold so it actually controls arrival (its
        # index-based check dominates otherwise).
        try:
            planner.ipath.arrive_threshold = threshold
        except (AttributeError, TypeError):
            logger.debug("[Go2] planner has no settable ipath.arrive_threshold; using distance check only.")

        collision_count = 0
        while True:
            loop_start = time.monotonic()

            state = self._odom_state()
            scan = self.get_scan()
            # Need both a fresh pose and a scan to plan; otherwise hold (no
            # velocity) and wait for data rather than planning blind.
            if state is None or scan is None or not scan.get("ranges"):
                cmd_vel.publish_twist(0.0, 0.0, 0.0)
                self._sleep_remaining(loop_start, period)
                continue

            points = planner.scan_to_point(
                state, scan, scan_offset=[0, 0, 0], angle_range=[-math.pi, math.pi], down_sample=2
            )
            action, info = planner(state, points, None)

            # Arrival: NEUPAN's flag OR pure distance (index-based check is unreliable).
            dist = float(np.hypot(state[0, 0] - target_x, state[1, 0] - target_y))
            if info.get("arrive") or dist < threshold:
                logger.info("[Go2] nav arrived (dist=%.3f m, arrive=%s).", dist, info.get("arrive"))
                return

            if info.get("stop"):
                collision_count += 1
                cmd_vel.publish_twist(0.0, 0.0, 0.0)
                if collision_count >= max_collision:
                    raise RuntimeError(
                        f"[Go2] nav aborted: collisions persisted {collision_count} cycles "
                        f"(min_distance={float(planner.min_distance):.3f} m)."
                    )
            else:
                collision_count = 0
                # Mecanum model: action is [vx, vy, omega] (3-dim). Unlike a diff
                # base, vy is real and published.
                vx = float(action[0, 0])
                vy = float(action[1, 0])
                wz = float(action[2, 0])
                # Safety envelope: clamp to configured hardware limits before publish.
                vx = max(-self._max_linear, min(self._max_linear, vx))
                vy = max(-self._max_linear, min(self._max_linear, vy))
                wz = max(-self._max_angular, min(self._max_angular, wz))
                cmd_vel.publish_twist(vx, vy, wz)

            self._sleep_remaining(loop_start, period)

    def _odom_state(self) -> np.ndarray | None:
        """Current pose as a NEUPAN-shaped ``[[x],[y],[theta_rad]]`` column, or None."""
        odom = self.get_odom_pose()
        if odom is None:
            return None
        return np.array([[float(odom["x"])], [float(odom["y"])], [math.radians(float(odom["yaw_deg"]))]])

    @staticmethod
    def _sleep_remaining(loop_start: float, period: float) -> None:
        """Sleep the unused fraction of the control period (busy-loop guard)."""
        elapsed = time.monotonic() - loop_start
        remaining = period - elapsed
        if remaining > 0:
            time.sleep(remaining)
