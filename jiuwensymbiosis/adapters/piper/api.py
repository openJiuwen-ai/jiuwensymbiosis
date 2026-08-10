# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``PiperApi`` — 6-DoF AgileX Piper + parallel gripper + open-vocab vision.

Design notes:
  * Agent-facing tool surface keeps the 4-DoF view (``goto_xyzr(x, y, z, r)``)
    where ``r`` becomes ``rz`` and ``rx, ry`` default to "tool pointing down".
    top-down pick prompts reuse the existing tool shape verbatim.
  * Full 6-DoF access for tilted picks is via ``goto_pose``.
  * Parallel gripper (``open_gripper`` / ``close_gripper``) drives the piper
    ``GripperCtrl``; v1 uses two-state open/close (width/force args accepted but
    the configured open-width is used — richer control lives in the lowlevel).
  * Vision: open-vocabulary detection (GroundingDINO + SAM2) on the wrist
    RealSense + 6-DoF eye-in-hand reprojection
    ``tf_base_cam = tf_base_flange(GetArmEndPose) @ tf_flange_cam``.

``_TOOL_DOWN_RX/RY`` defines the Euler "tool pointing straight down" orientation.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from jiuwensymbiosis.env.protocol import PiperFullDriver

from jiuwensymbiosis.adapters.piper.env import PiperEnv
from jiuwensymbiosis.adapters.piper.geometry import FlangePose, pixel_and_depth_to_base_xyz
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.api.decorators import robot_tool
from jiuwensymbiosis.api.mixins import (
    JointMotionMixin,
    MotionMixin,
    ParallelGripperMixin,
    VisionMixin,
)

logger = logging.getLogger(__name__)

# piper 的"工具竖直朝下"(ry=0)在工作区高处不可达、且够不到桌面物体顶面；
# 真机标定(2026-06-08)：略倾 ry≈30 才能在抓取高度可达。tip↔flange 因此带水平分量。
_TOOL_DOWN_RX = 180.0
_TOOL_DOWN_RY = 30.0


class PiperApi(
    MotionMixin,
    JointMotionMixin,
    ParallelGripperMixin,
    VisionMixin,
    BaseRobotApi,
):
    """6-DoF AgileX Piper + parallel gripper + open-vocab wrist vision."""

    def __init__(
        self,
        env: PiperEnv,
        *,
        detector_service_url: str = "http://127.0.0.1:8114",
        default_object_name: str = "object",
        z_correction_mm: float = 0.0,
        grasp_z_offset_mm: float = -25.0,
        place_z_offset_mm: float = 75.0,
    ) -> None:
        """Initialize PiperApi with env, detector service URL, and grasp geometry constants."""
        super().__init__(env)
        self._detector_service_url = detector_service_url
        self._seg_fn: Callable[..., list[dict[str, Any]]] | None = None
        self._default_object = default_object_name
        # Constant base-frame Z correction added to detections (see PiperConfig).
        self._z_correction_mm = float(z_correction_mm)
        # Offset from the detected TOP to the deterministic grasp point (see PiperConfig).
        self._grasp_z_offset_mm = float(grasp_z_offset_mm)
        # Stack place offset above a target's top (see PiperConfig).
        self._place_z_offset_mm = float(place_z_offset_mm)

    # ============================================================  Motion
    # ``home`` is inherited from MotionMixin (delegates to env.home()).

    @robot_tool(desc="Get current TIP pose (mm/deg, base frame).")
    def get_pose(self) -> dict:
        p = self.env.get_flange_pose()
        tool_off = self.env.tool_offset_mm
        return {
            "x": p.x,
            "y": p.y,
            "z": p.z - tool_off,
            "rx": p.rx,
            "ry": p.ry,
            "rz": p.rz,
        }

    @robot_tool(desc="Get raw flange pose (diagnostic; prefer get_pose for task code).")
    def get_flange_pose(self) -> dict:
        p = self.env.get_flange_pose()
        return {"x": p.x, "y": p.y, "z": p.z, "rx": p.rx, "ry": p.ry, "rz": p.rz}

    @robot_tool(desc="Get the home pose constants (read-only).")
    def get_home_pose(self) -> dict:
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
            "Move the tip to absolute (x, y, z[, r]) in mm/deg, base frame. "
            "Tool defaults to pointing straight down (rx=180, ry=0); r becomes rz. "
            "When calibration is loaded, z is in TIP frame (tool offset is added "
            "internally before commanding the flange). For arbitrary tilt, use goto_pose."
        ),
        tags=["motion"],
    )
    def goto_xyzr(self, x: float, y: float, z: float, r: float | None = None) -> None:
        if r is None:
            r = self.env.get_flange_pose().rz
        # Tilted tool (ry=_TOOL_DOWN_RY): the tip sits tool_offset_mm along the tool
        # axis below AND ahead of the flange. flange = tip + tool_offset_mm·(sin ry in
        # +X, cos ry in +Z).  (The +X sign matches the touch calibration: flange is
        # behind the tip.)
        tool_offset_mm = self.env.tool_offset_mm
        ry_rad = math.radians(_TOOL_DOWN_RY)
        flange_x = x + tool_offset_mm * math.sin(ry_rad)
        flange_z = z + tool_offset_mm * math.cos(ry_rad)
        logger.info(
            "[PiperApi] goto_xyzr TIP=(%.2f, %.2f, %.2f, rz=%.2f) -> flange=(%.2f, %.2f, %.2f, ry=%.1f)",
            x,
            y,
            z,
            r,
            flange_x,
            y,
            flange_z,
            _TOOL_DOWN_RY,
        )
        self.env.move_to_flange(FlangePose(flange_x, y, flange_z, _TOOL_DOWN_RX, _TOOL_DOWN_RY, float(r)))

    def servo_to_tip(self, pose: dict) -> None:
        """NON-BLOCKING servo command toward a TIP-frame pose (real-time loop).

        Mirrors ``goto_xyzr``'s tip→flange conversion (tilted tool ``ry``, tool
        offset) but issues the command via the env's non-blocking
        ``servo_to_flange`` instead of the blocking ``move_to_flange``. The
        real-time ``ServoController`` calls this each tick; ``get_pose`` (also
        TIP frame) is its matching pose reader, so the loop stays frame-
        consistent. ``pose`` keys: ``x/y/z`` (mm) + optional ``r``/``rz`` (deg).
        """
        x = float(pose["x"])
        y = float(pose["y"])
        z = float(pose["z"])
        r = pose.get("r", pose.get("rz"))
        if r is None:
            r = self.env.get_flange_pose().rz
        tool_offset_mm = self.env.tool_offset_mm
        ry_rad = math.radians(_TOOL_DOWN_RY)
        flange_x = x + tool_offset_mm * math.sin(ry_rad)
        flange_z = z + tool_offset_mm * math.cos(ry_rad)
        self.env.servo_to_flange(
            {
                "x": flange_x,
                "y": y,
                "z": flange_z,
                "rx": _TOOL_DOWN_RX,
                "ry": _TOOL_DOWN_RY,
                "rz": float(r),
            }
        )

    @robot_tool(
        desc="Full 6-DoF move (x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg). "
        "z is FLANGE frame (no tool-offset compensation).",
        tags=["motion"],
        input_params={
            "type": "object",
            "properties": {
                "pose": {
                    "type": "object",
                    "properties": {
                        "x_mm": {"type": "number"},
                        "y_mm": {"type": "number"},
                        "z_mm": {"type": "number"},
                        "rx_deg": {"type": "number"},
                        "ry_deg": {"type": "number"},
                        "rz_deg": {"type": "number"},
                    },
                    "required": ["x_mm", "y_mm", "z_mm", "rx_deg", "ry_deg", "rz_deg"],
                }
            },
            "required": ["pose"],
        },
    )
    def goto_pose(self, pose: FlangePose) -> None:
        if isinstance(pose, dict):
            pose = FlangePose(**pose)
        logger.info("[PiperApi] goto_pose -> %s", pose.as_tuple())
        self.env.move_to_flange(pose)

    # ============================================================  Joint
    # ``move_joint`` is inherited from JointMotionMixin (delegates to env.move_joint()).

    # ============================================================  Gripper
    # ``open_gripper`` / ``close_gripper`` are inherited from ParallelGripperMixin
    # (delegate to env.set_end_effector()); v1 uses the configured width/effort and
    # accepts width_mm/force_n only for API parity.

    # ============================================================  Vision
    def _project_pixel_to_base_raw(self, u: float, v: float, depth_m: float) -> np.ndarray:
        """Eye-in-hand RAW projection: ``tf_base_flange(live) @ tf_flange_cam``.

        The vendor-specific seam VisionMixin delegates to; applies NO xy/z
        correction (the shared geometry owns that). Reads the live flange pose,
        so the arm must be settled when this runs.
        """
        ll = self._ll()
        if ll.tf_flange_cam is None:
            raise RuntimeError("get_grasp_info_simple needs a loaded calibration (set calib_path in YAML).")
        calib = ll.calibration
        intrinsics = calib.get("intrinsics") if calib is not None else None
        if intrinsics is None:
            intrinsics = ll.intrinsics
        if intrinsics is None:
            raise RuntimeError("camera intrinsics unavailable (no calibration, no live camera)")
        p = self.env.get_flange_pose()
        flange_pose = FlangePose(p.x, p.y, p.z, p.rx, p.ry, p.rz)
        return pixel_and_depth_to_base_xyz((u, v), depth_m, flange_pose, ll.tf_flange_cam, intrinsics)

    def _grasp_debug_extra(self) -> dict:
        """Piper eye-in-hand debug context (frame model + live flange pose)."""
        p = self.env.get_flange_pose()
        return {
            "flange_pose_6dof": [p.x, p.y, p.z, p.rx, p.ry, p.rz],
            "frame_model": "piper_eye_in_hand_tf_base_flange@tf_flange_cam",
        }

    # ``get_grasp_info_simple`` / ``pixel_to_base_xyz`` / ``get_image`` are inherited
    # from VisionMixin, which drives the ``_project_pixel_to_base_raw`` seam above.

    @robot_tool(
        desc="Run a higher-level scene analysis grounded on object_name. "
        "Returns detection counts + top scores; useful for quick sanity checks."
    )
    def analyze_scene(self, object_name: str | None = None) -> dict:
        target = object_name or self._default_object
        rgb = self.get_image()
        if rgb is None:
            return {"ok": False, "reason": "no_camera"}
        self._ensure_detector()
        if self._seg_fn is None:
            return {"ok": False, "reason": "detector_unavailable"}
        try:
            results = self._seg_fn(rgb, text_prompt=target)
        except Exception as exc:  # noqa: BLE001 - surface detector failure as ok=False
            return {"ok": False, "reason": str(exc)}
        scores = sorted((float(r.get("score", 0.0)) for r in results), reverse=True)
        return {
            "ok": True,
            "object": target,
            "n_detections": len(results),
            "top_scores": scores[:5],
        }

    # ---------------------------------------------------------------- helpers
    def _ll(self) -> PiperFullDriver:
        """The vendor driver, for vision/calibration reads only (motion/gripper go via ``self.env``).

        The returned object satisfies RobotDriver + JointDriver + CameraDriver +
        GripperDriver + VisionDriver. Callers accessing vision-specific attributes
        (``tf_flange_cam``, ``calibration``, ``intrinsics``, ``grab_frames``)
        should be aware that these come from the composite driver protocol.
        """
        ll = self.env.low_level
        if ll is None:
            raise RuntimeError("PiperApi: env not connected. Call session.connect() / use `with session:`.")
        return cast("PiperFullDriver", ll)
