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
from jiuwensymbiosis.api import defaults
from jiuwensymbiosis.api.actions import (
    ANALYZE_SCENE,
    CLOSE_GRIPPER,
    GET_GRASP_INFO_SIMPLE,
    GET_HOME_POSE,
    GET_IMAGE,
    GET_POSE,
    GOTO_XYZR,
    MOVE_DIRECTION,
    MOVE_JOINT,
    OPEN_GRIPPER,
    PIXEL_TO_BASE_XYZ,
    implements,
)
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.api.decorators import robot_tool
from jiuwensymbiosis.perception.detector_client import init_detector
from jiuwensymbiosis.perception.vision import (
    GraspFailure,
    GraspResult,
    apply_xy_correction,
    detect_and_centroid,
    dump_grasp_debug,
)

logger = logging.getLogger(__name__)

# piper 的"工具竖直朝下"(ry=0)在工作区高处不可达、且够不到桌面物体顶面；
# 真机标定(2026-06-08)：略倾 ry≈30 才能在抓取高度可达。tip↔flange 因此带水平分量。
_TOOL_DOWN_RX = 180.0
_TOOL_DOWN_RY = 30.0


def _flange_pose_from_dict(pose: dict) -> FlangePose:
    """Build a ``FlangePose`` from the shared ``goto_pose`` payload.

    The contract's keys are ``x/y/z`` (mm) + ``rx/ry/rz`` (deg) — the same ones
    SafetyRail unpacks to bounds-check the move. This vendor type spells them
    ``x_mm`` / ``rx_deg``, so translate; the vendor spelling is still accepted so
    existing callers holding a FlangePose-shaped dict keep working.
    """
    if "x_mm" in pose:
        return FlangePose(**pose)
    return FlangePose(
        x_mm=float(pose["x"]), y_mm=float(pose["y"]), z_mm=float(pose["z"]),
        rx_deg=float(pose["rx"]), ry_deg=float(pose["ry"]), rz_deg=float(pose["rz"]),
    )


class PiperApi(BaseRobotApi):
    """6-DoF AgileX Piper + parallel gripper + open-vocab wrist vision.

    Every action this body offers is declared below, so this file IS the capability
    list. The ones whose implementation is the generic Env delegation forward to
    ``api.defaults``; the rest are Piper geometry (tilted tool, tip↔flange offset)
    or Piper vision (wrist eye-in-hand calibration).
    """

    # Marker capabilities: real things this body can do that no ACTION advertises, so the
    # class attr is the only way to declare them (see BaseRobotApi.capabilities). Leaving one
    # out is not cosmetic — the agent gates on ``api.capabilities & env.capabilities``, so an
    # ability the hardware has and the api forgets to claim is silently switched off.
    #   motion.servo  — servo_to_tip streams non-blocking tip targets, which is what lets the
    #                   fast path FOLLOW a target that keeps moving instead of aiming once.
    #   vision.depth   — the wrist camera returns aligned depth.
    capability = {"motion.servo", "vision.depth"}

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
    # ``home`` comes from BaseRobotApi — every body owes a safe posture.

    @implements(MOVE_DIRECTION)
    def move_direction(self, direction: str, distance_mm: float) -> dict:
        return defaults.move_direction(self, direction, distance_mm)

    @implements(GET_POSE)
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

    @implements(GET_HOME_POSE)
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

    @implements(GOTO_XYZR)
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

    # The FLANGE-frame twin of get_flange_pose, and body-only for the same reason: "where the
    # flange is" depends on how long this robot's tool happens to be, so a plan written against
    # it does not survive being moved. Task code wants the TIP — goto_pose / goto_xyzr. This one
    # is for bring-up, calibration and tool changes, where the flange IS the thing you mean.
    # Not in the shared vocabulary, so the planner never sees it; SafetyRail still does.
    @robot_tool(
        desc="Move the FLANGE to an absolute pose (diagnostic/calibration; prefer goto_pose for task code).",
        capability="motion.cartesian",
        tags=["motion"],
    )
    def goto_flange_pose(self, pose: FlangePose) -> None:
        if isinstance(pose, dict):
            pose = _flange_pose_from_dict(pose)
        logger.info("[PiperApi] goto_flange_pose -> %s", pose.as_tuple())
        self.env.move_to_flange(pose)

    # ============================================================  Joint
    @implements(MOVE_JOINT)
    def move_joint(self, q: list[float]) -> None:
        return defaults.move_joint(self, q)

    # ============================================================  Gripper
    # v1 is two-state: the configured width/effort is used and width_mm / force_n are
    # accepted for contract parity only.
    @implements(OPEN_GRIPPER)
    def open_gripper(self, width_mm: float = 80.0) -> dict:
        return defaults.open_gripper(self, width_mm)

    @implements(CLOSE_GRIPPER)
    def close_gripper(self, force_n: float | None = None) -> dict:
        return defaults.close_gripper(self, force_n)

    # ============================================================  Vision
    @implements(PIXEL_TO_BASE_XYZ)
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

    @implements(GET_GRASP_INFO_SIMPLE)
    def get_grasp_info_simple(self, object_name: str) -> GraspResult | GraspFailure:
        ll = self._ll()
        frames = ll.grab_frames()
        if frames is None:
            return {"ok": False, "reason": "no_camera", "object": object_name}
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
            # detect_and_centroid returns plain dict; structurally a GraspFailure
            return det  # type: ignore[return-value]

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
                "[grasp-debug] flange pose moved between frame grab and projection! grab=%s proj=%s",
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
        #   place_z = top + place_z_offset_mm  (descend HERE to release a held object
        #             ON TOP of this object, so the held object's bottom rests on this top).
        top_z = float(xyz_final[2])
        z_floor = self.env.z_min_safe
        grasp_z = top_z + self._grasp_z_offset_mm
        if z_floor is not None:
            grasp_z = max(grasp_z, float(z_floor))
        place_z = top_z + self._place_z_offset_mm
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

    @implements(GET_IMAGE)
    def get_image(self):
        return defaults.get_image(self)

    @implements(ANALYZE_SCENE)
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
        # The shared action means "every instance", so list them. This body has no
        # per-instance depth, so an entry carries score + pixel only; a planner still
        # learns HOW MANY there are, which is what drives a multi-target loop.
        objects = [{"object": target, "score": float(r.get("score", 0.0)),
                    "pixel_uv": r.get("center") or r.get("pixel_uv")} for r in results]
        return {
            "ok": True,
            "object": target,
            "count": len(objects),
            "objects": objects,
            "n_detections": len(results),
            "top_scores": scores[:5],
        }

    # ---------------------------------------------------------------- helpers
    def _ll(self) -> PiperFullDriver:
        """The vendor driver, for vision/calibration reads only (motion/gripper go via ``self.env``).

        The returned object satisfies CartesianDriver + JointDriver + CameraDriver +
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
        except Exception as exc:  # noqa: BLE001 - detector init best-effort; tools degrade
            logger.warning(
                "[PiperApi] detector init failed (%s); detection tools will return ok=False.",
                exc,
            )


class _PoseShim:
    """Exposes a piper pose with an ``r`` alias for debug helpers
    (``detect_and_centroid`` / ``dump_grasp_debug``) that log ``pose.x/y/z/r``.
    """

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
