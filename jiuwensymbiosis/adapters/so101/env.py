# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""``So101Env`` — adapter from ``BaseRobotEnv`` to ``So101Driver``.

The always-available capabilities are ``motion.cartesian`` + ``motion.joint``
+ ``grasp.parallel``. Vision capabilities are added only when a camera is
configured; opening a configured camera is a fail-closed connection requirement,
so the milestone-A no-camera config does not expose unusable vision tools.

Properties are read-only (setters raise ``AttributeError``) per the BaseRobotEnv
override contract: mypy forbids a read-only property overriding a read-write one
without an explicit setter.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from jiuwensymbiosis.adapters.so101.config import So101Config
from jiuwensymbiosis.adapters.so101.geometry import So101Pose
from jiuwensymbiosis.env.base import BaseRobotEnv, RobotObservation

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jiuwensymbiosis.adapters.so101.lowlevel import So101Driver
    from jiuwensymbiosis.env.protocol import RobotDriver

logger = logging.getLogger(__name__)


class So101Env(BaseRobotEnv):
    """SO-101 5-DoF arm + parallel gripper (LeRobot ``SOFollower``)."""

    # Keep the class-level superset for the static adapter validator.  The
    # instance-level value is narrowed in __init__/connect according to the
    # configured and actually opened camera.
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
    _BASE_CAPABILITIES = frozenset({"motion.cartesian", "motion.joint", "grasp.parallel"})
    name = "so101"

    def __init__(self, cfg: So101Config) -> None:
        """Store config; driver is None until connect()."""
        self.cfg = cfg
        self._inner: So101Driver | None = None
        self._connected = False
        self.capabilities = self._capabilities_for_config()

    def _capabilities_for_config(self) -> frozenset[str]:
        """Return capabilities implied by the declarative config.

        ``camera_serial=None`` is the explicit milestone-A/no-camera mode.  A
        configured camera supplies RGB, depth and the adapter's detection tools;
        the latter still fail closed at call time if hand-eye calibration or the
        detector is unavailable.
        """
        caps = set(self._BASE_CAPABILITIES)
        if self.cfg.camera_serial:
            caps.update({"vision.camera", "vision.depth", "vision.detection"})
        return frozenset(caps)

    def _capabilities_for_driver(self, driver: Any) -> frozenset[str]:
        """Confirm vision capabilities from the connected driver's camera state."""
        caps = set(self._capabilities_for_config())
        if self.cfg.camera_serial and getattr(driver, "camera_available", True) is False:
            caps.difference_update({"vision.camera", "vision.depth", "vision.detection"})
        return frozenset(caps)

    # --- controlled penetration point (read-only) ---------------------------
    @property
    def low_level(self) -> RobotDriver | None:
        """The underlying driver, or None before connect()."""
        return self._inner

    @low_level.setter
    def low_level(self, _: RobotDriver | None) -> None:
        raise AttributeError("So101Env.low_level is read-only (binds to self._inner via connect/disconnect)")

    # --- safety envelope (SafetyRail reads these) ----------------------------
    @property
    def z_min_safe(self) -> float | None:
        """Tip/control-frame Z floor (mm) from the driver, else config."""
        if self._inner is not None:
            return float(self._inner.z_min_safe)
        return float(self.cfg.z_min_safe_mm)

    @z_min_safe.setter
    def z_min_safe(self, _: float | None) -> None:
        raise AttributeError("So101Env.z_min_safe is read-only (computed from driver/config)")

    @property
    def workspace_bounds(self) -> tuple[float, float, float, float] | None:
        """XY workspace bounds ``(xmin, ymin, xmax, ymax)`` in mm from config."""
        return self.cfg.workspace_bounds

    @workspace_bounds.setter
    def workspace_bounds(self, _: tuple[float, float, float, float] | None) -> None:
        raise AttributeError("So101Env.workspace_bounds is read-only (computed from config)")

    @property
    def joint_limits(self) -> dict[str, tuple[float, float]] | None:
        """Joint soft limits (deg), always keyed over ``ARM_JOINT_ORDER`` (5 items).

        Rebuilt fresh each access so SafetyRail's ``len(q) == len(names)`` check
        and the ``q[i]`` index labels stay stable regardless of dict ordering.
        """
        from jiuwensymbiosis.adapters.so101.lowlevel import ARM_JOINT_ORDER

        limits = self.cfg.joint_limits
        if limits is None:
            return None
        # Re-insert in ARM_JOINT_ORDER for stable indexing.
        return {name: limits[name] for name in ARM_JOINT_ORDER}

    @joint_limits.setter
    def joint_limits(self, _: dict[str, tuple[float, float]] | None) -> None:
        raise AttributeError("So101Env.joint_limits is read-only (computed from config)")

    # --- robot body constants -----------------------------------------------
    @property
    def home_pose(self) -> So101Pose | None:
        """FK(home_joints_deg) control-frame pose, or None before connect."""
        if self._inner is not None:
            return self._inner.home_pose
        return None

    @home_pose.setter
    def home_pose(self, _: Any) -> None:
        raise AttributeError("So101Env.home_pose is read-only (read from driver)")

    @property
    def tool_offset_mm(self) -> float:
        """Flange-to-tip offset (mm); milestone A fixed 0.0."""
        if self._inner is not None:
            return float(self._inner.tool_offset_mm)
        return 0.0

    @tool_offset_mm.setter
    def tool_offset_mm(self, _: float) -> None:
        raise AttributeError("So101Env.tool_offset_mm is read-only (milestone A fixed 0.0)")

    # ----------------------------------------------------------------- connect
    def connect(self) -> None:
        """Instantiate the driver from config and connect it atomically.

        On any failure the env stays disconnected (``_inner`` left None) and
        the driver's idempotent close is invoked so no partial state leaks.
        """
        if self._connected:
            return
        from jiuwensymbiosis.adapters.so101.lowlevel import So101Driver

        driver = So101Driver(self.cfg)
        try:
            driver.connect()
        except Exception:
            # Idempotent teardown; do not leak a partially-connected driver.
            try:
                driver.close()
            except Exception as exc:  # noqa: BLE001 - best-effort
                logger.warning("So101Env: driver.close() after failed connect raised %s", exc)
            self._inner = None
            self._connected = False
            raise
        self._inner = driver
        self._connected = True
        self.capabilities = self._capabilities_for_driver(driver)
        logger.info("So101Env connected (port=%s)", self.cfg.port)

    def disconnect(self) -> None:
        """Idempotent, best-effort driver teardown."""
        if not self._connected:
            return
        try:
            # `_inner` is non-None here: set True only after assignment in connect().
            self._inner.close()  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 - disconnect is best-effort
            logger.warning("So101Env disconnect failed: %s", exc)
        self._inner = None
        self._connected = False
        self.capabilities = self._capabilities_for_config()

    # -------------------------------------------------------------- observation
    def get_observation(self) -> RobotObservation:
        """Collect pose + joint + gripper + (optional) RGB-D state."""
        if self._inner is None:
            return RobotObservation()
        rgb: np.ndarray | None = None
        depth: np.ndarray | None = None
        try:
            frames = self._inner.grab_frames()  # CameraDriver sibling protocol
            if frames is not None:
                rgb, depth = frames
        except Exception as exc:  # noqa: BLE001 - camera read best-effort
            logger.debug("So101Env.grab_frames failed: %s", exc)
        pose: dict[str, float] | None = None
        try:
            p = self._inner.get_pose()
            pose = {"x": p.x, "y": p.y, "z": p.z, "rx": p.rx, "ry": p.ry, "rz": p.rz}
        except Exception as exc:  # noqa: BLE001 - pose read best-effort
            logger.debug("So101Env.get_pose failed: %s", exc)
            pose = None
        joints: list[float] | None = None
        try:
            joints = list(self._inner.get_angles())
        except Exception as exc:  # noqa: BLE001 - joint read best-effort
            logger.debug("So101Env.get_angles failed: %s", exc)
            joints = None
        gripper: float | None = None
        try:
            gripper = float(self._inner.get_gripper_position())
        except Exception as exc:  # noqa: BLE001 - gripper read best-effort
            logger.debug("So101Env.get_gripper_position failed: %s", exc)
            gripper = None
        return RobotObservation(
            pose=pose,
            joints=joints,
            rgb=rgb,
            depth=depth,
            extra={
                "z_min_safe": self.z_min_safe,
                "gripper_state": gripper,  # grasp.parallel-capability-gated
            },
        )

    def get_angles(self) -> list[float]:
        """Read the 5 arm joint angles (deg) in ``ARM_JOINT_ORDER``; raise if down."""
        if self._inner is None:
            raise RuntimeError("So101Env.get_angles: env not connected.")
        return list(self._inner.get_angles())

    # --- cartesian entry point: normalize any pose-like object to So101Pose -----
    def move_to_flange(self, pose: Any) -> None:
        """Dispatch a flange-frame Cartesian move to the driver.

        ``So101Driver.move_to_pose_blocking`` requires a :class:`So101Pose`; the
        generic capability mixins (:meth:`MotionMixin.goto_xyzr`,
        :meth:`MotionMixin.move_direction`) hand a ``SimpleNamespace`` here, so
        normalize a complete mapping or attribute-bag pose into a ``So101Pose``
        before delegating. Missing coordinates are rejected instead of silently
        becoming zero-valued hardware targets. A ``So101Pose`` is passed through.
        """
        if isinstance(pose, So101Pose):
            target = pose
        else:
            if isinstance(pose, Mapping):
                missing = [name for name in ("x", "y", "z", "rx", "ry") if name not in pose]
                if "rz" not in pose and "r" not in pose:
                    missing.append("rz (or r)")
                if missing:
                    raise TypeError(f"SO-101 pose mapping missing required fields: {', '.join(missing)}.")
                x, y, z = pose["x"], pose["y"], pose["z"]
                rx, ry = pose["rx"], pose["ry"]
                rz = pose["rz"] if "rz" in pose else pose["r"]
            else:
                missing = [name for name in ("x", "y", "z", "rx", "ry") if not hasattr(pose, name)]
                if not hasattr(pose, "rz") and not hasattr(pose, "r"):
                    missing.append("rz (or r)")
                if missing:
                    raise TypeError(f"SO-101 pose object missing required fields: {', '.join(missing)}.")
                x, y, z = pose.x, pose.y, pose.z
                rx, ry = pose.rx, pose.ry
                rz = pose.rz if hasattr(pose, "rz") else pose.r
            target = So101Pose(
                x=float(x),
                y=float(y),
                z=float(z),
                rx=float(rx),
                ry=float(ry),
                rz=float(rz),
            )
        self._require_driver().move_to_pose_blocking(target)
