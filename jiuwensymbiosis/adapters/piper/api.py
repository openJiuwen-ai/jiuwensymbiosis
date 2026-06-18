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
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

import numpy as np

if TYPE_CHECKING:
    from jiuwensymbiosis.adapters._common.protocol import PiperFullDriver

from jiuwensymbiosis.adapters._common.detector_client import init_detector
from jiuwensymbiosis.adapters._common.vision import (
    apply_xy_correction,
    detect_and_centroid,
    dump_grasp_debug,
)
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
        chip_thickness_mm: float = 75.0,
    ) -> None:
        """Initialize PiperApi with env, detector service URL, and grasp geometry constants."""
        super().__init__(env)
        self._detector_service_url = detector_service_url
        self._seg_fn: Optional[Callable[..., list[dict[str, Any]]]] = None
        self._default_object = default_object_name
        # Constant base-frame Z correction added to detections (see PiperConfig).
        self._z_correction_mm = float(z_correction_mm)
        # Offset from the detected TOP to the deterministic grasp point (see PiperConfig).
        self._grasp_z_offset_mm = float(grasp_z_offset_mm)
        # Stack place offset above a target's top (see PiperConfig).
        self._chip_thickness_mm = float(chip_thickness_mm)

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
    def goto_xyzr(self, x: float, y: float, z: float, r: Optional[float] = None) -> None:
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
    @robot_tool(
        desc="Pixel (u,v) at depth_m (meters) → base-frame XYZ in mm. " "Requires a loaded calibration.",
    )
    def pixel_to_base_xyz(self, u: float, v: float, depth_m: float) -> dict:
        ll = self._ll()
        if ll.tf_flange_cam is None:
            raise RuntimeError("pixel_to_base_xyz needs a loaded calibration (set calib_path in YAML).")
        calib = ll.calibration
        intrinsics = calib.get("intrinsics") if calib is not None else None
        if intrinsics is None:
            intrinsics = ll.intrinsics
        if intrinsics is None:
            raise RuntimeError("camera intrinsics unavailable (no calibration, no live camera)")
        p = self.env.get_flange_pose()
        flange_pose = FlangePose(p.x, p.y, p.z, p.rx, p.ry, p.rz)
        xyz = pixel_and_depth_to_base_xyz((u, v), depth_m, flange_pose, ll.tf_flange_cam, intrinsics)
        if calib is not None:
            xyz, _desc = apply_xy_correction(
                np.asarray(xyz, dtype=np.float64),
                xy_transform=calib.get("xy_transform"),
                xy_correction_mm=calib.get("xy_correction_mm"),
            )
        return {"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])}

    @robot_tool(
        desc="Live open-vocab detection of object_name + depth + calibration → base XYZ. Returns "
        '{"ok": bool, "object": str, "position": [x,y,z]_mm, "grasp_z": float, '
        '"grasp_position": [x,y,z]_mm, "place_z": float, "place_position": [x,y,z]_mm, '
        '"score": float, "pixel_uv": [u,v], "depth_m": float}.'
    )
    def get_grasp_info_simple(self, object_name: str) -> dict:
        ll = self._ll()
        frames = ll.grab_frames()
        if frames is None:
            return {"ok": False, "reason": "no_camera"}
        rgb, depth_img_m = frames

        tcp_at_grab = self.env.get_flange_pose()
        self._ensure_detector()
        det = detect_and_centroid(
            rgb=rgb,
            depth_img_m=depth_img_m,
            seg_fn=self._seg_fn,
            object_name=object_name,
            tcp_at_grab=_PoseShim(tcp_at_grab),
        )
        if not det.get("ok"):
            return det

        u, v, depth_m = det["u"], det["v"], det["depth_m"]
        best = det["best"]
        img_w, img_h = det["img_shape"]
        mask_h, mask_w = det["mask_shape"]

        if ll.tf_flange_cam is None:
            raise RuntimeError("get_grasp_info_simple needs a loaded calibration (set calib_path in YAML).")
        calib = ll.calibration
        intrinsics = calib.get("intrinsics") if calib is not None else None
        intrinsics_src = "calib"
        if intrinsics is None:
            intrinsics = ll.intrinsics
            intrinsics_src = "live"
        if intrinsics is None:
            raise RuntimeError("camera intrinsics unavailable (no calibration, no live camera)")

        tcp_at_proj = self.env.get_flange_pose()
        if tcp_at_proj.as_tuple() != tcp_at_grab.as_tuple():
            logger.warning(
                "[grasp-debug] flange pose moved between frame grab and projection! " "grab=%s proj=%s",
                tcp_at_grab.as_tuple(),
                tcp_at_proj.as_tuple(),
            )
        flange_pose = FlangePose(
            tcp_at_proj.x,
            tcp_at_proj.y,
            tcp_at_proj.z,
            tcp_at_proj.rx,
            tcp_at_proj.ry,
            tcp_at_proj.rz,
        )
        xyz_raw = pixel_and_depth_to_base_xyz(
            (u, v),
            depth_m,
            flange_pose,
            ll.tf_flange_cam,
            intrinsics,
        )

        xy_transform = calib.get("xy_transform") if calib is not None else None
        xy_corr = calib.get("xy_correction_mm") if (calib is not None and xy_transform is None) else None
        xyz_final, corr_desc = apply_xy_correction(
            xyz_raw,
            xy_transform=xy_transform,
            xy_correction_mm=xy_corr,
        )
        if self._z_correction_mm:
            xyz_final = np.asarray(xyz_final, dtype=np.float64).copy()
            xyz_final[2] += self._z_correction_mm
            corr_desc = f"{corr_desc}+z{self._z_correction_mm:+.0f}"

        intrinsics_flat = np.asarray(intrinsics, dtype=float).reshape(-1)
        logger.info(
            "[grasp-debug] K_src=%s flange_pose=(%.2f, %.2f, %.2f, %.2f, %.2f, %.2f) "
            "raw_xyz_mm=(%.2f, %.2f, %.2f) corr=%s final_xyz_mm=(%.2f, %.2f, %.2f)",
            intrinsics_src,
            *flange_pose.as_tuple(),
            float(xyz_raw[0]),
            float(xyz_raw[1]),
            float(xyz_raw[2]),
            corr_desc,
            float(xyz_final[0]),
            float(xyz_final[1]),
            float(xyz_final[2]),
        )

        try:
            dump_grasp_debug(
                rgb=rgb,
                object_name=object_name,
                best=best,
                u=u,
                v=v,
                depth_m=depth_m,
                tcp_grab=_PoseShim(tcp_at_grab),
                tcp_proj=_PoseShim(tcp_at_proj),
                xyz_raw=xyz_raw,
                xyz_final=xyz_final,
                xy_corr=xy_corr,
                xy_transform=xy_transform,
                intrinsics_src=intrinsics_src,
                intrinsics=intrinsics_flat.tolist(),
                img_shape=(img_w, img_h),
                mask_shape=(mask_w, mask_h),
                extra_info={
                    "flange_pose_6dof": list(flange_pose.as_tuple()),
                    "frame_model": "piper_eye_in_hand_tf_base_flange@tf_flange_cam",
                },
            )
        except Exception as exc:  # noqa: BLE001 - debug dump must never break a grasp
            logger.debug("[grasp-debug] dump failed: %s", exc)

        # Deterministic grasp + stack-place geometry, computed HERE (perception side)
        # so the agent never does z math:
        #   grasp_z = top + grasp_z_offset_mm  (descend HERE to grasp the body),
        #             clamped to the safety floor;
        #   place_z = top + chip_thickness_mm  (descend HERE to release a held object
        #             ON TOP of this object, so the held object's bottom rests on this top).
        top_z = float(xyz_final[2])
        z_floor = self.env.z_min_safe
        grasp_z = top_z + self._grasp_z_offset_mm
        if z_floor is not None:
            grasp_z = max(grasp_z, float(z_floor))
        place_z = top_z + self._chip_thickness_mm
        x_f, y_f = float(xyz_final[0]), float(xyz_final[1])
        logger.info(
            "[PiperApi] %s: pos=(%.1f, %.1f, %.1f) grasp_z=%.1f place_z=%.1f score=%.2f",
            object_name,
            x_f,
            y_f,
            top_z,
            grasp_z,
            place_z,
            best["score"],
        )
        return {
            "ok": True,
            "object": object_name,
            "position": [x_f, y_f, top_z],
            "grasp_z": grasp_z,
            "grasp_position": [x_f, y_f, grasp_z],
            "place_z": place_z,
            "place_position": [x_f, y_f, place_z],
            "score": float(best["score"]),
            "pixel_uv": [u, v],
            "depth_m": depth_m,
        }

    # ``get_image`` is inherited from VisionMixin (grabs frames via env.low_level).

    @robot_tool(
        desc="Run a higher-level scene analysis grounded on object_name. "
        "Returns detection counts + top scores; useful for quick sanity checks."
    )
    def analyze_scene(self, object_name: Optional[str] = None) -> dict:
        target = object_name or self._default_object
        rgb = self.get_image()
        if rgb is None:
            return {"ok": False, "reason": "no_camera"}
        self._ensure_detector()
        if self._seg_fn is None:
            return {"ok": False, "reason": "detector_unavailable"}
        try:
            results = self._seg_fn(rgb, text_prompt=target)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": str(exc)}
        scores = sorted((float(r.get("score", 0.0)) for r in results), reverse=True)
        return {
            "ok": True,
            "object": target,
            "n_detections": len(results),
            "top_scores": scores[:5],
        }

    # ---------------------------------------------------------------- helpers
    def _ll(self) -> "PiperFullDriver":
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

    def _ensure_detector(self) -> None:
        """Lazy-init the detector segmentation function if not already bound."""
        if self._seg_fn is not None:
            return
        try:
            self._seg_fn = init_detector(self._detector_service_url)
            logger.info("[PiperApi] detector client bound to %s", self._detector_service_url)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[PiperApi] detector init failed (%s); detection tools will return ok=False.",
                exc,
            )


class _PoseShim:
    """Exposes a piper pose with an ``r`` alias for debug helpers
    (``detect_and_centroid`` / ``dump_grasp_debug``) that log ``pose.x/y/z/r``."""

    __slots__ = ("x", "y", "z", "rx", "ry", "rz", "r")

    def __init__(self, pose) -> None:
        """Copy pose fields + alias rz as r, debug helpers."""
        self.x = pose.x
        self.y = pose.y
        self.z = pose.z
        self.rx = pose.rx
        self.ry = pose.ry
        self.rz = pose.rz
        self.r = pose.rz
