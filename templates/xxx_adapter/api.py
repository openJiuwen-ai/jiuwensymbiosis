# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""XxxApi — capability-mixin implementation for the Xxx robot.

Inherits from the Mixins that match your hardware capabilities
(see the Capability ↔ Mixin table in docs/hardware-porting-guide.md §3.2)
and overrides every abstract @robot_tool method.

Key patterns shown here:
  - Motion / end-effector use the Env verbs (``self.env.home() /
    move_to_flange() / move_joint() / set_end_effector()``).
  - Robot body constants (``home_pose``, ``tool_offset_mm``) use Env properties
    (``self.env.home_pose`` / ``self.env.tool_offset_mm``).
  - Vision calibration data uses ``self.env.low_level`` (the ``RobotDriver``
    protocol) — this is a controlled penetration for sensor-specific data that
    does not belong on the Env body abstraction.
  - @robot_tool decorators provide hardware-specific descriptions.
  - Every method returns ``{"ok": True/False, ...}`` shape.
"""

from __future__ import annotations

from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.api.decorators import robot_tool
from jiuwensymbiosis.api.mixins import (
    MotionMixin,
    # JointMotionMixin,     # [选填] Uncomment if your robot supports joint motion
    # SuctionMixin,         # [选填] Uncomment for suction end-effector
    # ParallelGripperMixin, # [选填] Uncomment for parallel gripper
    # VisionMixin,          # [选填] Uncomment for vision+detection
)


class XxxApi(
    MotionMixin,
    # JointMotionMixin,
    # SuctionMixin,
    # ParallelGripperMixin,
    # VisionMixin,
    BaseRobotApi,  # always last
):
    """Robot API for Xxx — TODO: replace with your robot description."""

    # If your Api.__init__ needs extra parameters beyond env (e.g. detector URL,
    # calibration constants), declare them here. The session builder passes them
    # via ``api_kwargs_from_cfg``.

    # ================================================================ Motion

    @robot_tool(
        desc="Return Xxx to the configured home pose (safe upper height).",
        tags=["motion"],
    )
    def home(self) -> None:
        """Return to the home pose (motion command → Env verb)."""
        self.env.home()

    @robot_tool(desc="Get current TIP pose (mm/deg, base frame).")
    def get_pose(self) -> dict:
        """Get current end-effector pose."""
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

    @robot_tool(desc="Get the home pose constants (read-only).")
    def get_home_pose(self) -> dict:
        """Get home pose constants (read-only)."""
        return {
            "x": self.env.home_pose.x,
            "y": self.env.home_pose.y,
            "z": self.env.home_pose.z,
            "rx": self.env.home_pose.rx,
            "ry": self.env.home_pose.ry,
            "rz": self.env.home_pose.rz,
        }

    @robot_tool(
        desc=(
            "Move the TIP to absolute (x, y, z[, r]) in mm/deg, base frame. If r is omitted, current r is preserved."
        ),
        tags=["motion"],
    )
    def goto_xyzr(self, x: float, y: float, z: float, r: float | None = None) -> None:
        """Move tip to target Cartesian pose. tip↔flange geometry stays in the api layer."""
        if r is None:
            r = self.env.get_flange_pose().rz
        pose = type("Pose", (), {"x": x, "y": y, "z": z, "rx": 180.0, "ry": 0.0, "rz": r})()
        self.env.move_to_flange(pose)

    # ============================================================= Joint [选填]
    # Uncomment if your robot supports joint-space motion:
    #
    # @robot_tool(desc="Move to joint configuration q (degrees).", tags=["motion"])
    # def move_joint(self, q: list[float]) -> None:
    #     self.env.move_joint(q)

    # ============================================================= Suction [选填]
    # Uncomment for suction end-effector:
    #
    # @robot_tool(desc="Turn suction ON.", tags=["grasp"])
    # def activate_suction(self) -> dict:
    #     self.env.set_end_effector(True)
    #     return {"ok": True, "state": "on"}
    #
    # @robot_tool(desc="Turn suction OFF.", tags=["grasp"])
    # def deactivate_suction(self) -> dict:
    #     self.env.set_end_effector(False)
    #     return {"ok": True, "state": "off"}

    # ============================================================ Gripper [选填]
    # Uncomment for parallel gripper:
    #
    # @robot_tool(desc="Close the parallel gripper.", tags=["grasp"])
    # def close_gripper(self, force_n: Optional[float] = None) -> dict:
    #     self.env.set_end_effector(True)
    #     return {"ok": True, "state": "closed"}
    #
    # @robot_tool(desc="Open the parallel gripper.", tags=["grasp"])
    # def open_gripper(self, width_mm: float = 80.0) -> dict:
    #     self.env.set_end_effector(False)
    #     return {"ok": True, "state": "open"}

    # ============================================================= Vision [选填]
    # Uncomment for vision-enabled robot. Requires:
    #   1. GroundingDINO+SAM2 detection server running (see _common/detector_sidecar)
    #   2. Camera calibration (hand-eye + intrinsics)
    #   3. Driver implementing grab_frames()
    #
    # from jiuwensymbiosis.adapters._common.detector_client import init_detector
    # from jiuwensymbiosis.adapters._common.vision import detect_and_centroid, apply_xy_correction
    #
    # def __init__(self, env, *, detector_service_url="http://127.0.0.1:8114",
    #              z_correction_mm=0.0, grasp_z_offset_mm=-25.0, chip_thickness_mm=75.0):
    #     super().__init__(env)
    #     self._detector_service_url = detector_service_url
    #     self._z_correction_mm = float(z_correction_mm)
    #     self._grasp_z_offset_mm = float(grasp_z_offset_mm)
    #     self._chip_thickness_mm = float(chip_thickness_mm)
    #     self._seg_fn = None
    #
    # def _ensure_detector(self):
    #     if self._seg_fn is not None:
    #         return
    #     self._seg_fn = init_detector(self._detector_service_url)
    #
    # @robot_tool(desc="Detect object + back-project to base XYZ. Returns "
    #             '{"ok": bool, "position": [x,y,z], "grasp_z": float, "score": float, ...}')
    # def get_grasp_info_simple(self, object_name: str) -> dict:
    #     ll = self.env.low_level
    #     frames = ll.grab_frames()
    #     if frames is None:
    #         return {"ok": False, "reason": "no_camera"}
    #     rgb, depth_img_m = frames
    #     self._ensure_detector()
    #     det = detect_and_centroid(rgb=rgb, depth_img_m=depth_img_m,
    #                               seg_fn=self._seg_fn, object_name=object_name,
    #                               tcp_at_grab=self.env.get_flange_pose())
    #     if not det.get("ok"):
    #         return det
    #     # TODO: pixel→base back-projection (per-robot geometry)
    #     # xyz_raw = pixel_and_depth_to_base_xyz(...)
    #     # xyz_final, _ = apply_xy_correction(xyz_raw, ...)
    #     return {"ok": False, "reason": "back_projection_not_implemented"}
    #
    # @robot_tool(desc="Convert pixel (u,v) at depth_m to base XYZ mm.")
    # def pixel_to_base_xyz(self, u: float, v: float, depth_m: float) -> dict:
    #     raise NotImplementedError("pixel_to_base_xyz requires calibration")
    #
    # @robot_tool(desc="Grab latest RGB frame.")
    # def get_image(self) -> Any:
    #     return self.env.grab_rgb()
    #
    # @robot_tool(desc="Run scene analysis grounded on object_name.")
    # def analyze_scene(self, object_name: Optional[str] = None) -> dict:
    #     return {"ok": False, "reason": "not_implemented"}
