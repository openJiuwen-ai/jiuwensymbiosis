# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cruzr environment wrapper."""

from __future__ import annotations

import logging
from typing import Any, Optional

from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
from jiuwensymbiosis.adapters.cruzr.geometry import LIFTER_LIMITS
from jiuwensymbiosis.env.base import BaseRobotEnv, RobotObservation

logger = logging.getLogger(__name__)


class CruzrEnv(BaseRobotEnv):
    """Cruzr robot controlled through ROS 2 joint commands."""

    capabilities = frozenset(
        {
            "motion.joint",
            "vision.camera",
            "vision.depth",
            "vision.detection",
            "motion.base",
            "motion.base_servo",
            "motion.lift",
            "motion.waist",
            "motion.goal",
            "vision.search",
            "grasp.dual_arm",
            "planning.reachability",
        }
    )
    name = "cruzr"

    # Waist first: it is the one that carries depth, so a hit there answers "where exactly"
    # and the base can square up straight away. The head only ever answers "which way".
    cameras = ("waist", "head")

    def __init__(self, cfg: CruzrConfig) -> None:
        """Store config; ROS 2 driver is created on connect."""
        self.cfg = cfg
        self._inner: Optional[Any] = None
        self._connected = False
        self._warm_thread: Optional[Any] = None
        self._warm_ik_thread: Optional[Any] = None
        self._warm_camera_thread: Optional[Any] = None
        # Read by RecoveryRail / WorldState. There is no cheap hardware probe for it
        # (read_hand_ft blocks ~2 s per arm), so CruzrApi writes it at the clamp/release
        # points it already knows about instead.
        self.holding_payload: bool = False
        # Safety envelope read by SafetyRail. The URDF lifter range the grasp/place
        # planner already filters its own candidates against, so declaring it rejects a
        # hallucinated set_lift_pose before the ramp starts without narrowing any target
        # the planner can produce.
        self.lift_limits = dict(LIFTER_LIMITS)
        # Expose URDF + arm chains for the framework planning.reachability capability.
        self.urdf_path = getattr(cfg, "urdf_path", None) or None
        self.arm_chains = {"left": ("base_link", cfg.left_arm_leaf),
                           "right": ("base_link", cfg.right_arm_leaf)}

    @property
    def low_level(self) -> Any:
        """Return the underlying ``CruzrLowLevel`` driver."""
        if self._inner is None:
            raise RuntimeError("CruzrEnv is not connected.")
        return self._inner

    def connect(self) -> None:
        """Connect to ROS 2 and create command publisher."""
        if self._connected:
            return
        from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrLowLevel

        self._inner = CruzrLowLevel(self.cfg)
        self._connected = True
        self._warm_self_collision_async()
        self._warm_ik_async()
        self._warm_camera_async()
        logger.info("CruzrEnv connected (command_topic=%s)", self.cfg.command_topic)

    def _warm_self_collision_async(self) -> None:
        """Build the pin+coal self-collision model in the background at connect so the
        first ``dual_arm_grasp`` doesn't pay the multi-second build in the dead time
        between the base arriving at the box and the arms closing. Search + approach
        (~15-20 s of driving) overlap the build; by grasp time the model is cached
        (``dual_arm_grasp`` joins this thread before its collision check). Daemon → never
        blocks connect/shutdown; a build failure degrades inside ``self_collision``
        (the collision check just disables, never crashes), so best-effort here too.
        """
        import threading

        urdf = getattr(self.cfg, "urdf_path", "")
        if not urdf:
            return
        pkg = getattr(self.cfg, "urdf_package_dir", None)

        def _warm() -> None:
            try:
                from jiuwensymbiosis.kinematics import self_collision
                self_collision.available(urdf, pkg)
            except Exception as exc:  # noqa: BLE001  # best-effort warm-up; grasp rebuilds if needed
                logger.debug("self-collision warm-up skipped: %s", exc)

        self._warm_thread = threading.Thread(
            target=_warm, name="cruzr-warm-collision", daemon=True)
        self._warm_thread.start()

    def _warm_ik_async(self) -> None:
        """Pre-build the pinocchio IK model in the background at connect so the first grasp's IK
        solve doesn't pay ``buildModelFromUrdf``. Daemon, best-effort (IK falls back to legacy DLS
        when pin is absent); mirrors ``_warm_self_collision_async``.
        """
        import threading

        urdf = getattr(self.cfg, "urdf_path", "")
        if not urdf:
            return

        def _warm() -> None:
            try:
                from jiuwensymbiosis.kinematics import ik_pinocchio
                ik_pinocchio.warm(urdf)
            except Exception as exc:  # noqa: BLE001  # best-effort warm-up; IK rebuilds/falls back if needed
                logger.debug("IK warm-up skipped: %s", exc)

        self._warm_ik_thread = threading.Thread(
            target=_warm, name="cruzr-warm-ik", daemon=True)
        self._warm_ik_thread.start()

    def _warm_camera_async(self) -> None:
        """Grab one waist + one head frame in the background at connect so the first real detect
        doesn't pay the camera worker's rclpy/DDS-discovery + TF-fill cold start. Daemon,
        best-effort — frames are discarded; a failed grab just leaves the cold start for later.
        """
        import threading

        inner = self._inner
        if inner is None:
            return

        def _warm() -> None:
            try:
                inner.grab_frames(camera="waist")
            except Exception as exc:  # noqa: BLE001  # best-effort; real grab retries the cold start
                logger.debug("waist camera warm-up skipped: %s", exc)
            try:
                inner.grab_head_frame()
            except Exception as exc:  # noqa: BLE001  # best-effort; real grab retries the cold start
                logger.debug("head camera warm-up skipped: %s", exc)

        self._warm_camera_thread = threading.Thread(
            target=_warm, name="cruzr-warm-camera", daemon=True)
        self._warm_camera_thread.start()

    def disconnect(self) -> None:
        """Close ROS 2 resources."""
        if not self._connected:
            return
        try:
            close = getattr(self._inner, "close", None)
            if callable(close):
                close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("CruzrEnv disconnect failed: %s", exc)
        self._inner = None
        self._connected = False

    def get_observation(self) -> RobotObservation:
        """Return latest known joint positions, if the state topic is available."""
        if self._inner is None:
            return RobotObservation()
        joints_by_name = {}
        try:
            joints_by_name = self._inner.get_joint_positions()
        except Exception as exc:  # noqa: BLE001
            logger.debug("CruzrEnv.get_joint_positions failed: %s", exc)
        return RobotObservation(
            joints=list(joints_by_name.values()) if joints_by_name else None,
            extra={
                "joint_positions": joints_by_name,
                "command_topic": self.cfg.command_topic,
                "state_topic": self.cfg.state_topic,
            },
        )

    def reset(self) -> None:
        """Return the default arm to home."""
        if self._inner is not None:
            self._inner.home(arm=self.cfg.default_arm)

    # --- mobility / perception verbs ---
    # Each one is spelled out: BaseRobotEnv declares them all, so an inherited
    # ``NotImplementedError`` — not a driver call — is what a missing verb gets.

    def navigate_relative(self, dx_m: float, dy_m: float = 0.0, dyaw_rad: float = 0.0) -> dict:
        """Turn then translate, using the driver's default odom-servo gains."""
        return self.low_level.navigate_relative(float(dx_m), float(dy_m), float(dyaw_rad))

    def navigate_arc(self, radius_m: float, dyaw_rad: float) -> dict:
        """One constant-curvature arc (differential wheels + odom curvature servo)."""
        return self.low_level.navigate_arc(float(radius_m), float(dyaw_rad))

    def start_base_drive(self, **kwargs: Any) -> Any:
        """Start the non-blocking forward-drive worker; returns its handle."""
        return self.low_level.start_base_drive(**kwargs)

    def base_drive_running(self, handle: Any) -> bool:
        """Whether the forward-drive worker behind ``handle`` is still rolling."""
        return bool(self.low_level.base_drive_running(handle))

    def steer_base_drive(self, handle: Any, bearing_rad: float) -> None:
        """Feed a live bearing to the running drive so it curves toward the target."""
        self.low_level.steer_base_drive(handle, float(bearing_rad))

    def hold_base_drive(self, handle: Any) -> None:
        """Pause the wheels of a running drive (target lost) without ending it."""
        self.low_level.hold_base_drive(handle)

    def stop_base_drive(self, handle: Any) -> dict:
        """Stop the drive worker and reap its result."""
        return self.low_level.stop_base_drive(handle) or {}

    def set_lifter(self, q_lifter: dict[str, float]) -> Any:
        """Command the three torso lifter joints to absolute angles (rad)."""
        return self.low_level.set_lifter(q_lifter)

    def grab_calibrated_frame(self, camera: str | None = None) -> Any:
        """One waist-RGBD frame as a ``CameraFrame`` (rgb + depth + K + base←cam), or None."""
        from jiuwensymbiosis.perception.frame import CameraFrame

        frames = self.low_level.grab_frames(camera=camera or "waist")
        if frames is None:
            return None
        rgb, depth_m, intrinsics, tf_base_cam = frames
        return CameraFrame(rgb=rgb, depth_m=depth_m, intrinsics=intrinsics, tf_base_cam=tf_base_cam)
