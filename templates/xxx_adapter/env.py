# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""XxxEnv — hardware abstraction wrapping XxxDriver.

Follow the *driver delegation* pattern:
  - env.connect()    → creates self.low_level = XxxDriver(...)
  - env.disconnect() → calls self.low_level.disconnect(); sets to None
  - env.get_observation() → reads from self.low_level

See docs/hardware-porting-guide.md Step 3 for details.
"""

from __future__ import annotations

# TODO: Replace with your actual driver
from jiuwensymbiosis.adapters.xxx.lowlevel import XxxDriver

from jiuwensymbiosis.env.base import BaseRobotEnv, RobotObservation


class XxxEnv(BaseRobotEnv):
    """Hardware environment for the Xxx robot.

    Capabilities — declare what this robot body actually supports.
    Must be a subset of ``KNOWN_CAPABILITIES`` (env/base.py:35).
    Add capabilities your hardware supports; remove those it doesn't.
    """

    capabilities = frozenset(
        {
            "motion.cartesian",  # Cartesian end-effector commands
            # "motion.joint",       # [选填] Joint-space commands
            # "grasp.suction",      # [选填] Suction cup
            # "grasp.parallel",     # [选填] Parallel gripper
            # "vision.camera",      # [选填] RGB image stream
            # "vision.depth",       # [选填] Depth stream
            # "vision.detection",   # [选填] Object detection (needs detector service)
        }
    )
    name: str = "xxx"

    def __init__(self, cfg) -> None:
        """Store config reference; do NOT connect yet (deferred to connect())."""
        self._cfg = cfg
        self.low_level: XxxDriver | None = None  # ← driver delegate

    # ---------------------------------------------------------------- lifecycle

    def connect(self) -> None:
        """Open hardware connection. Must be idempotent."""
        if self.low_level is not None:
            return  # already connected
        # TODO: Pass real connection parameters from cfg
        self.low_level = XxxDriver()
        self.low_level.connect()

    def disconnect(self) -> None:
        """Release hardware. Must be idempotent and safe at any state."""
        if self.low_level is None:
            return  # already disconnected
        try:
            self.low_level.disconnect()
        finally:
            self.low_level = None

    # --------------------------------------------------------------- observation

    def get_observation(self) -> RobotObservation:
        """Best-effort snapshot. Should NOT raise on transient sensor gaps."""
        ll = self.low_level
        if ll is None:
            return RobotObservation()  # connected → return empty

        # Pose
        try:
            p = ll.get_pose()
            pose = {
                "x": p.x,
                "y": p.y,
                "z": p.z,
                "rx": getattr(p, "rx", 0.0),
                "ry": getattr(p, "ry", 0.0),
                "rz": getattr(p, "rz", 0.0),
            }
        except Exception:
            pose = None

        # Camera frames (only if vision.camera capability declared)
        rgb = None
        depth = None
        if "vision.camera" in self.capabilities:
            try:
                frames = ll.grab_frames()
                if frames is not None:
                    rgb, depth = frames
            except Exception:
                pass

        return RobotObservation(pose=pose, rgb=rgb, depth=depth)

    # ----------------------------------------------------------- optional overrides

    def reset(self) -> None:
        """Bring robot back to safe pose. Default: no-op. Override as needed."""
        pass

    def emergency_stop(self) -> None:
        """Software-level halt. Default: no-op. Hardware E-stop must be physical."""
        pass

    # ---------------------------------------------------------- safety boundaries

    @property
    def z_min_safe(self) -> float:
        """Z floor (mm) — SafetyRail reads this automatically.
        Expose from config so users can adjust via YAML.
        """
        return float(self._cfg.z_min_safe_mm)

    @property
    def workspace_bounds(self) -> tuple | None:
        """XY workspace bounds or None. SafetyRail reads this automatically."""
        cfg = self._cfg
        if cfg.x_min_mm is not None:
            return (cfg.x_min_mm, cfg.y_min_mm, cfg.x_max_mm, cfg.y_max_mm)
        return None

    # ---------------------------------------------------- robot body constants

    @property
    def home_pose(self):
        """Home pose (vendor Pose object) from the driver, or None before connect."""
        if self.low_level is not None:
            return self.low_level.home_pose
        return None

    @property
    def tool_offset_mm(self) -> float:
        """Flange-to-tip offset (mm) from the driver, or 0 before connect."""
        if self.low_level is not None:
            return float(self.low_level.tool_offset_mm)
        return 0.0
