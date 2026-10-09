# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``PiperEnv`` — adapter from ``BaseRobotEnv`` to ``PiperLowLevel``.

Wraps the driver (``low_level``), exposes ``connect``/``disconnect``/
``get_observation`` plus the safety contract (``z_min_safe`` /
``workspace_bounds``). Cartesian input is normalized to the driver's pose type;
other motion/end-effector operations use the inherited Env verbs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from jiuwensymbiosis.adapters.piper.config import PiperConfig
from jiuwensymbiosis.adapters.piper.geometry import flange_pose_to_tip, normalize_flange_pose
from jiuwensymbiosis.env.base import BaseRobotEnv, RobotObservation

if TYPE_CHECKING:
    from jiuwensymbiosis.env.protocol import CartesianDriver

logger = logging.getLogger(__name__)


class PiperEnv(BaseRobotEnv):
    """6-DoF AgileX Piper + parallel gripper + optional wrist RealSense."""

    capabilities = frozenset(
        {
            "motion.cartesian",
            "motion.joint",
            "motion.servo",
            "grasp.parallel",
            "vision.camera",
            "vision.depth",
            "vision.detection",
        }
    )
    name = "piper"
    # The Piper SDK speaks degrees (``PiperJointAngles``, ``move_joint_blocking(q)``), so
    # ``joint_limits`` and the observed joints are degrees too.
    _joint_units = "deg"
    # ``PiperApi.goto_xyzr`` offers only ``top_down`` (its Literal says so), so the default
    # and the whole enum are the same single value.
    _default_orientation_policy = "top_down"

    def __init__(self, cfg: PiperConfig) -> None:
        """Store config; driver is None until connect()."""
        self.cfg = cfg
        self._inner: CartesianDriver | None = None  # PiperLowLevel
        self._connected = False

    @property
    def low_level(self) -> CartesianDriver | None:
        """The underlying low-level driver (PiperLowLevel), or None before connect()."""
        return self._inner

    @low_level.setter
    def low_level(self, value: CartesianDriver | None) -> None:
        """Bind a driver before connect() — the seam a smoke test or a simulator uses.

        Once one is bound, only connect/disconnect may rebind it: that is the invariant the
        binding protects, and it holds whether the driver came from connect() or from here.
        """
        if self._inner is not None:
            raise AttributeError("PiperEnv.low_level is already bound — connect/disconnect owns rebinding")
        self._inner = value

    @property
    def z_min_safe(self) -> float | None:
        """Tip-frame Z floor (mm): from the live driver if connected, else config."""
        if self._inner is not None:
            return float(self._inner.z_min_safe)
        cfg_val = getattr(self.cfg, "z_min_safe_mm", None)
        return float(cfg_val) if cfg_val is not None else None

    @z_min_safe.setter
    def z_min_safe(self, _: float | None) -> None:
        raise AttributeError("PiperEnv.z_min_safe is read-only (computed from driver/config)")

    @property
    def joint_names(self) -> list[str] | None:
        """``j1..j6`` in chain order — read off ``PiperJointAngles`` so it cannot drift.

        The shared ``move_joint`` action addresses joints by NAME; the Piper SDK takes a
        vector, and this is the ordering the Env converts between them with. Derived rather
        than hard-coded because the dataclass IS the vendor's order.
        """
        from dataclasses import fields

        from jiuwensymbiosis.adapters.piper.lowlevel import PiperJointAngles

        return [f.name for f in fields(PiperJointAngles)]

    @joint_names.setter
    def joint_names(self, _: list[str] | None) -> None:
        raise AttributeError("PiperEnv.joint_names is read-only (the vendor joint dataclass)")

    @property
    def workspace_bounds(self) -> tuple[float, float, float, float] | None:
        """XY workspace bounds ``(xmin, ymin, xmax, ymax)`` in mm from config, or None."""
        c = self.cfg
        raw = (
            getattr(c, "x_min_mm", None),
            getattr(c, "y_min_mm", None),
            getattr(c, "x_max_mm", None),
            getattr(c, "y_max_mm", None),
        )
        if any(v is None for v in raw):
            return None
        return cast("tuple[float, float, float, float]", raw)

    @workspace_bounds.setter
    def workspace_bounds(self, _: tuple[float, float, float, float] | None) -> None:
        raise AttributeError("PiperEnv.workspace_bounds is read-only (computed from config)")

    @property
    def joint_limits(self) -> dict[str, tuple[float, float]] | None:
        """Joint soft limits (deg) from config, or None when unconfigured."""
        return getattr(self.cfg, "joint_limits", None)

    @joint_limits.setter
    def joint_limits(self, _: dict[str, tuple[float, float]] | None) -> None:
        raise AttributeError("PiperEnv.joint_limits is read-only (computed from config)")

    @property
    def home_pose(self):
        """FLANGE home target (vendor Pose object), or None before connect."""
        if self._inner is not None:
            return self._inner.home_pose
        return None

    @home_pose.setter
    def home_pose(self, _: Any) -> None:
        raise AttributeError("PiperEnv.home_pose is read-only (read from driver)")

    @property
    def tool_offset_mm(self) -> float:
        """Local +Z tool extension (mm), from the driver or config before connect."""
        if self._inner is not None:
            return float(self._inner.tool_offset_mm)
        return float(getattr(self.cfg, "tool_offset_mm", 0.0))

    @tool_offset_mm.setter
    def tool_offset_mm(self, _: float) -> None:
        raise AttributeError("PiperEnv.tool_offset_mm is read-only (computed from driver/config)")

    # ----------------------------------------------------------------- connect
    def connect(self) -> None:
        """Instantiate and connect the PiperLowLevel driver from config."""
        if self._connected:
            return
        from jiuwensymbiosis.adapters.piper.lowlevel import PiperLowLevel

        kwargs: dict[str, Any] = dict(  # noqa: C408  # mutable builder, conditionally extended below
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
        # Always forward the config fallbacks: the driver falls back to them when the
        # calibration carries no object anchor, which a calib_path no longer rules out.
        # They are inert when the calibration does carry one.
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
        """Close the low-level driver and mark as disconnected only on success."""
        if self._inner is None:
            self._connected = False
            return
        # Retain the driver and connected state if any underlying resource failed
        # to close. RobotSession can then report the failure and a later call can
        # retry the driver's idempotent close path.
        self._inner.close()
        self._inner = None
        self._connected = False

    def home(self) -> None:
        """Move to the configured home target; home_use_init_pose selects the startup pose."""
        self._require_cartesian().home()

    def move_to_flange(self, pose: Any) -> None:
        """Normalize a complete flange pose before dispatching to the driver.

        Accept native FlangePose values, mappings (including read-only ones),
        and complete x/y/z/rx/ry/rz attribute objects. r aliases yaw; rz wins.
        Missing fields are rejected rather than filled with hardware targets.
        """
        target = normalize_flange_pose(pose)
        self._require_cartesian().move_to_pose_blocking(target)

    # -------------------------------------------------------------- observation
    def get_observation(self) -> RobotObservation:
        """Publish TIP/FLANGE from one sample together, or neither if conversion fails."""
        if self._inner is None:
            return RobotObservation()
        rgb: np.ndarray | None = None
        depth: np.ndarray | None = None
        try:
            frames = self._inner.grab_frames()  # type: ignore[attr-defined]  # CameraDriver sibling protocol
            if frames is not None:
                rgb, depth = frames
        except Exception as exc:  # noqa: BLE001 - camera read best-effort
            logger.debug("PiperEnv.grab_frames failed: %s", exc)
        pose: dict | None = None
        flange_pose: dict | None = None
        try:
            p = self._inner.get_pose()
            sample_flange = {"x": p.x, "y": p.y, "z": p.z, "rx": p.rx, "ry": p.ry, "rz": p.rz}
            sample_tip = flange_pose_to_tip(p, self.tool_offset_mm)
            flange_pose, pose = sample_flange, sample_tip
        except Exception as exc:  # noqa: BLE001 - pose read/conversion best-effort
            logger.debug("PiperEnv pose read/conversion failed: %s", exc)
        joints: list[float] | None = None
        try:
            a = self._inner.get_angles()  # type: ignore[attr-defined]  # JointDriver sibling protocol
            joints = list(a.as_tuple())
        except Exception:  # noqa: BLE001 - joint read best-effort
            joints = None
        return RobotObservation(
            pose=pose,
            joints=joints,
            rgb=rgb,
            depth=depth,
            extra={
                "flange_pose": flange_pose,
                "z_min_safe": self.z_min_safe,
                # GripperDriver sibling protocol; grasp.parallel-capability-gated
                "gripper_state": (
                    self._inner.gripper_state  # type: ignore[attr-defined]
                    if "grasp.parallel" in self.capabilities
                    else None
                ),
            },
        )

    def get_angles(self) -> Any:
        """Read joint angles from the driver; raise if not connected."""
        if self._inner is None:
            raise RuntimeError("PiperEnv.get_angles: env not connected.")
        return self._inner.get_angles()  # type: ignore[attr-defined]  # JointDriver sibling protocol
