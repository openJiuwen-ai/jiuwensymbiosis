# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``UbetechCruzrS2Api`` — UBTECH Cruzr S2 mobile-base capability API.

Mobile-base form factor: ``MotionMixin`` (chassis x/y/yaw) + ``VisionMixin``
(camera stream via ``get_image``). ``goto_xyzr`` is overridden for planar base
semantics — ``x, y`` are target base-frame meters, ``r`` is target yaw (deg),
and ``z`` is ignored (a mobile base doesn't actuate world-frame Z).

The high-level vision methods (``get_grasp_info_simple`` / ``pixel_to_base_xyz``
/ ``analyze_scene``) stay abstract (raise ``NotImplementedError``): a mobile
base has no arm to grasp with, so the grasp-geometry pipeline isn't wired up.
``get_image`` uses the working default (raw frame grab) so the camera stream is
available for VLM scene inspection. ``validate_adapter`` flags the abstract
methods as WARN (A-10) — that's the expected "capability declared, high-level
method not implemented" signal for this form factor.
"""

from __future__ import annotations

from typing import Any

from jiuwensymbiosis.adapters.ubetech_cruzr_s2.env import UbetechCruzrS2Env
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.api.decorators import robot_tool
from jiuwensymbiosis.api.mixins import MotionMixin, VisionMixin
from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger(__name__)


class UbetechCruzrS2Api(
    MotionMixin,
    VisionMixin,
    BaseRobotApi,  # always last
):
    """UBTECH Cruzr S2 mobile-base API — chassis motion + camera stream."""

    def __init__(
        self,
        env: UbetechCruzrS2Env,
        *,
        detector_service_url: str = "http://127.0.0.1:8114",
        z_correction_mm: float = 0.0,
        grasp_z_offset_mm: float = -25.0,
        chip_thickness_mm: float = 75.0,
    ) -> None:
        """Initialize UbetechCruzrS2Api with env and (optional) detection geometry constants."""
        super().__init__(env)
        self._detector_service_url = detector_service_url
        self._z_correction_mm = float(z_correction_mm)
        self._grasp_z_offset_mm = float(grasp_z_offset_mm)
        self._chip_thickness_mm = float(chip_thickness_mm)

    # ================================================================ Motion
    # ``home`` is inherited from MotionMixin (delegates to env.home()).

    @robot_tool(desc="Get current base pose (meters for x/y, deg for yaw).")
    def get_pose(self) -> dict:
        """Get current mobile-base pose (x, y meters + yaw deg from odometry)."""
        p = self.env.get_flange_pose()
        return {
            "x": p.x,
            "y": p.y,
            "z": p.z,
            "rx": p.rx,
            "ry": p.ry,
            "rz": p.rz,
        }

    @robot_tool(desc="Get the home base pose constants (read-only).")
    def get_home_pose(self) -> dict:
        """Get home pose constants (read-only)."""
        p = self.env.home_pose
        return {
            "x": p.x,
            "y": p.y,
            "z": p.z,
            "rx": p.rx,
            "ry": p.ry,
            "rz": p.rz,
            "r": p.rz,
        }

    @robot_tool(
        desc=(
            "Drive the base to absolute (x, y[, r]) in meters/deg, base frame. "
            "r is target yaw (deg); if omitted, current yaw is preserved. "
            "z is ignored (planar base — world-frame Z is not actuated)."
        ),
        tags=["motion"],
    )
    def goto_xyzr(self, x: float, y: float, z: float = 0.0, r: float | None = None) -> None:
        """Drive the base to a planar target pose.

        ``x, y`` are target base-frame meters, ``r`` is target yaw (deg). ``z``
        is accepted for API shape parity with the arm ``goto_xyzr`` but ignored
        (a mobile base doesn't move in world-frame Z). ``rx, ry`` are fixed
        at 0 (planar) and never commanded.
        """
        if r is None:
            cur = self.env.get_flange_pose()
            r = getattr(cur, "rz", getattr(cur, "r", 0.0))
        # 6-DoF-shaped pose; the driver's move_to_pose_blocking only uses x/y/rz.
        pose = type("Pose", (), {"x": float(x), "y": float(y), "z": float(z), "rx": 0.0, "ry": 0.0, "rz": float(r)})()
        self.env.move_to_flange(pose)

    # ================================================================ Vision
    # ``get_image`` is inherited from VisionMixin (raw frame grab via env.grab_rgb).
    # The high-level detection methods stay abstract (see module docstring).

    def get_grasp_info_simple(self, object_name: str) -> dict:
        """Not implemented on a mobile base (no arm to grasp with)."""
        raise NotImplementedError("UbetechCruzrS2Api.get_grasp_info_simple: no arm on this form factor")

    def pixel_to_base_xyz(self, u: float, v: float, depth_m: float) -> dict:
        """Not implemented on a mobile base (no hand-eye calibration wired up)."""
        raise NotImplementedError("UbetechCruzrS2Api.pixel_to_base_xyz: no calibration on this form factor")

    def analyze_scene(self, object_name: str | None = None) -> dict:
        """Not implemented — override to wire up a VLM scene-analysis backend."""
        raise NotImplementedError("UbetechCruzrS2Api.analyze_scene: not implemented")

    # ``get_image`` returns Any from the inherited VisionMixin default; keep the
    # property-shaped accessor explicit so mypy has a typed surface.
    def get_image(self) -> Any:
        """Latest RGB frame, or None if no camera (inherited VisionMixin default)."""
        return self.env.grab_rgb()
