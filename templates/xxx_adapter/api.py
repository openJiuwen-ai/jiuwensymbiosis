# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""XxxApi — capability-mixin implementation for the Xxx robot.

Inherits from the Mixins that match your hardware capabilities
(see docs/zh/how-to/port-hardware-adapter.md)
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
    # Uncomment for a vision-enabled robot. Requires:
    #   1. GroundingDINO+SAM2 detection server running (see _common/detector_sidecar)
    #   2. Camera calibration (hand-eye + intrinsics)
    #   3. Driver implementing grab_frames()
    #
    # VisionMixin already implements get_grasp_info_simple / pixel_to_base_xyz /
    # get_image / _ensure_detector on top of the shared grasp geometry. A new
    # adapter supplies ONLY (a) the geometry constants via __init__ (or accepts
    # the mixin class defaults) and (b) the projection seam below.
    #
    # from jiuwensymbiosis.utils.geometry import apply_transform, pixel_and_depth_to_camera_xyz
    #
    # def __init__(self, env, *, detector_service_url="http://127.0.0.1:8114",
    #              z_correction_mm=0.0, grasp_z_offset_mm=-25.0, place_z_offset_mm=75.0):
    #     super().__init__(env)
    #     self._detector_service_url = detector_service_url
    #     self._z_correction_mm = float(z_correction_mm)
    #     self._grasp_z_offset_mm = float(grasp_z_offset_mm)
    #     self._place_z_offset_mm = float(place_z_offset_mm)
    #     self._seg_fn = None
    #
    # def _project_pixel_to_base_raw(self, u, v, depth_m):
    #     # The one vendor-specific step: pixel + depth -> RAW base-frame XYZ (mm).
    #     # Apply NO xy/z correction here — the shared geometry owns that.
    #     #   eye-in-hand:  tf_base_cam = pose_to_tf(self.env.get_flange_pose()) @ tf_flange_cam
    #     #   eye-to-hand:  tf_base_cam = <constant T_base_cam from calibration>
    #     ll = self.env.low_level
    #     intrinsics = (ll.calibration or {}).get("intrinsics") or ll.intrinsics
    #     p_cam = pixel_and_depth_to_camera_xyz((u, v), depth_m, intrinsics)
    #     return apply_transform(tf_base_cam, p_cam)  # per-robot tf_base_cam
    #
    # @robot_tool(desc="Run scene analysis grounded on object_name.")
    # def analyze_scene(self, object_name: Optional[str] = None) -> dict:
    #     return {"ok": False, "reason": "not_implemented"}
