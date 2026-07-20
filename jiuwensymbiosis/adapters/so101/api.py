# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""``So101Api`` — capability-mixed API for the SO-101 adapter.

Capability composition: ``MotionMixin`` + ``JointMotionMixin``
+ ``ParallelGripperMixin`` + ``VisionMixin`` + ``BaseRobotApi``. Vision is
desktop-fixed eye-to-hand (milestone B): a RealSense D405 bolted to the desk,
NOT the wrist. The hand-eye calibration therefore solves ``T_base_cam`` (a
CONSTANT camera-in-base transform), so projection does NOT read the flange pose
per step — ``p_base = T_base_cam @ p_cam`` — unlike piper's eye-in-hand
``tf_base_flange @ tf_flange_cam``.

Two overrides fix contract mismatches with the LeRobot percentage gripper:

- ``open_gripper`` / ``close_gripper`` are re-decorated with
  ``input_params={"type": "object", "properties": {}}`` so the LLM-facing tool
  has NO parameters (the SO-101 gripper is two-state percentage, not mm/N).
  Python keeps the mixin-compatible ``width_mm``/``force_n`` params but ignores
  them — no fake unit conversion.
- ``goto_pose`` uses a FLAT signature ``(x, y, z, rx, ry, rz)`` because
  ``SafetyRail.before_tool_call`` reads only top-level ``args.get("x"/"y"/"z")``
  and does NOT unpack a nested ``pose={...}``. A flat shape is the only way the
  rail's Z/XY boundary check fires automatically.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from jiuwensymbiosis.adapters.so101.geometry import So101Pose
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.api.decorators import robot_tool
from jiuwensymbiosis.api.mixins import (
    JointMotionMixin,
    MotionMixin,
    ParallelGripperMixin,
    VisionMixin,
)
from jiuwensymbiosis.perception.detector_client import init_detector
from jiuwensymbiosis.perception.vision import (
    GraspFailure,
    GraspResult,
    apply_xy_correction,
    detect_and_centroid,
    dump_grasp_debug,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jiuwensymbiosis.adapters.so101.env import So101Env

logger = logging.getLogger(__name__)

__all__ = ["So101Api"]


class So101Api(
    MotionMixin,
    JointMotionMixin,
    ParallelGripperMixin,
    VisionMixin,
    BaseRobotApi,
):
    """SO-101 5-DoF arm + parallel gripper + desktop eye-to-hand vision."""

    def __init__(
        self,
        env: So101Env,
        *,
        detector_service_url: str = "http://127.0.0.1:8114",
        z_correction_mm: float = 0.0,
        grasp_z_offset_mm: float = -25.0,
        chip_thickness_mm: float = 75.0,
    ) -> None:
        super().__init__(env)
        self._detector_service_url = detector_service_url
        self._seg_fn: Callable[..., list[dict[str, Any]]] | None = None
        self._z_correction_mm = float(z_correction_mm)
        self._grasp_z_offset_mm = float(grasp_z_offset_mm)
        self._chip_thickness_mm = float(chip_thickness_mm)

    # --- gripper overrides (two-state percentage, no mm/N params) ------------
    @robot_tool(
        desc="Open the SO-101 gripper to the configured fully-open percentage position.",
        capability="grasp.parallel",
        input_params={"type": "object", "properties": {}},
        tags=["grasp"],
    )
    def open_gripper(self, width_mm: float = 80.0) -> dict:
        """Open the gripper. ``width_mm`` is accepted for API parity and ignored —
        the SO-101 gripper is a two-state percentage actuator (no width control).
        Maps to ``env.set_end_effector(False)`` -> driver sends ``gripper_open_pos``.
        """
        del width_mm  # ignored: SO-101 is two-state, no width calibration yet.
        self.env.set_end_effector(False)
        return {"ok": True, "state": "open"}

    @robot_tool(
        desc="Close the SO-101 gripper to the configured fully-closed percentage position.",
        capability="grasp.parallel",
        input_params={"type": "object", "properties": {}},
        tags=["grasp"],
    )
    def close_gripper(self, force_n: float | None = None) -> dict:
        """Close the gripper. ``force_n`` is accepted for API parity and ignored —
        the SO-101 gripper is a two-state percentage actuator (no force control).
        Maps to ``env.set_end_effector(True)`` -> driver sends ``gripper_close_pos``.
        """
        del force_n  # ignored: SO-101 is two-state, no force calibration yet.
        self.env.set_end_effector(True)
        return {"ok": True, "state": "closed"}

    # --- cartesian overrides -------------------------------------------------
    @robot_tool(
        desc=(
            "Move the SO-101 control frame to absolute (x, y, z[, r]) in mm/deg, "
            "base frame. Position is strongly constrained; orientation is "
            "best-effort (5-DoF underactuated arm). If r is omitted, the current "
            "roll about Z is preserved. Default approach is top-down (rx=180, ry=0)."
        ),
        capability="motion.cartesian",
        tags=["motion"],
    )
    def goto_xyzr(self, x: float, y: float, z: float, r: float | None = None) -> None:
        """Move the control frame to ``(x, y, z[, r])`` mm/deg, base frame.

        SO-101 is a 5-DoF underactuated arm: position is the strong constraint,
        orientation is best-effort (the IK solver weights position higher via
        ``ik_orientation_weight``). If ``r`` is omitted, preserve the current
        roll about Z.
        """
        if r is None:
            cur = self.env.get_flange_pose()
            r = getattr(cur, "rz", getattr(cur, "r", 0.0))
        # Milestone A: top-down approach (rx=180, ry=0) like the default mixin,
        # but routed through the flat goto_pose so the same IK path is used.
        self.goto_pose(x=float(x), y=float(y), z=float(z), rx=180.0, ry=0.0, rz=float(r))

    @robot_tool(
        desc=(
            "Move the SO-101 control frame to absolute (x, y, z, rx, ry, rz) "
            "in mm/deg, base frame. Position is strongly constrained; orientation "
            "is best-effort (5-DoF underactuated arm). SafetyRail checks x/y/z "
            "bounds before this runs."
        ),
        capability="motion.cartesian",
        tags=["motion"],
    )
    def goto_pose(
        self,
        x: float,
        y: float,
        z: float,
        rx: float,
        ry: float,
        rz: float,
    ) -> None:
        """Move to an absolute Cartesian pose via flat parameters.

        Flat (not nested ``pose={...}``) so ``SafetyRail.before_tool_call`` can
        read ``x``/``y``/``z`` from top-level tool args and apply Z/XY bounds.
        """
        self.env.move_to_flange(
            So101Pose(
                x=float(x),
                y=float(y),
                z=float(z),
                rx=float(rx),
                ry=float(ry),
                rz=float(rz),
            )
        )

    # ============================================================  Vision
    # Desktop-fixed eye-to-hand: ``tf_base_cam`` is a CONSTANT (camera-in-base),
    # so projection is ``p_base = tf_base_cam @ p_cam`` — NO flange read per step
    # (unlike piper eye-in-hand ``tf_base_flange @ tf_flange_cam``).
    def _ll(self):
        """The driver, for vision/calibration reads only."""
        ll = self.env.low_level
        if ll is None:
            raise RuntimeError("So101Api: env not connected. Call session.connect() / use `with session:`.")
        return ll

    def _resolve_intrinsics(self, ll: Any) -> tuple[Any, str]:
        """Intrinsics from calibration (preferred) else live camera."""
        calib = getattr(ll, "calibration", None)
        intrinsics = calib.get("intrinsics") if calib is not None else None
        if intrinsics is not None:
            return intrinsics, "calib"
        intrinsics = getattr(ll, "intrinsics", None)
        if intrinsics is None:
            raise RuntimeError("camera intrinsics unavailable (no calibration, no live camera)")
        return intrinsics, "live"

    def _ensure_detector(self) -> None:
        """Lazy-init the detector segmentation function if not already bound."""
        if self._seg_fn is not None:
            return
        try:
            self._seg_fn = init_detector(self._detector_service_url)
            logger.info("[So101Api] detector client bound to %s", self._detector_service_url)
        except Exception as exc:  # noqa: BLE001 - detector init best-effort
            logger.warning("[So101Api] detector init failed (%s); detection tools will return ok=False.", exc)

    @robot_tool(
        desc="Pixel (u,v) at depth_m (meters) -> base-frame XYZ in mm. Requires a loaded eye-to-hand calibration.",
    )
    def pixel_to_base_xyz(self, u: float, v: float, depth_m: float) -> dict:
        """Back-project (u, v, depth_m) -> base-frame XYZ (mm).

        Eye-to-hand: ``p_base = tf_base_cam @ pixel_and_depth_to_camera_xyz(uv, depth, K)``
        where ``tf_base_cam`` is a constant (camera fixed to the desk). Applies
        the calibration's ``xy_transform``/``xy_correction_mm`` when present.
        """
        from jiuwensymbiosis.utils.geometry import apply_transform, pixel_and_depth_to_camera_xyz

        ll = self._ll()
        if ll.tf_base_cam is None:
            raise RuntimeError("pixel_to_base_xyz needs a loaded eye-to-hand calibration (set calib_path in YAML).")
        intrinsics, _src = self._resolve_intrinsics(ll)
        xyz_raw = apply_transform(
            ll.tf_base_cam,
            pixel_and_depth_to_camera_xyz((float(u), float(v)), float(depth_m), intrinsics),
        )
        calib = getattr(ll, "calibration", None)
        if calib is not None:
            xyz_raw, _desc = apply_xy_correction(
                np.asarray(xyz_raw, dtype=np.float64),
                xy_transform=calib.get("xy_transform"),
                xy_correction_mm=calib.get("xy_correction_mm"),
            )
        return {"x": float(xyz_raw[0]), "y": float(xyz_raw[1]), "z": float(xyz_raw[2])}

    @robot_tool(
        desc="Live open-vocab detection of object_name + depth + eye-to-hand calibration -> base XYZ. Returns "
        '{"ok": bool, "object": str, "position": [x,y,z]_mm, "grasp_z": float, '
        '"grasp_position": [x,y,z]_mm, "place_z": float, "place_position": [x,y,z]_mm, '
        '"score": float, "pixel_uv": [u,v], "depth_m": float}.'
    )
    def get_grasp_info_simple(self, object_name: str) -> GraspResult | GraspFailure:
        """Detect an object and return its 3D grasp/place geometry (eye-to-hand).

        Pipeline: grab -> detect_and_centroid -> eye-to-hand back-project ->
        xy-correct -> grasp/place geometry. ``tf_base_cam`` is a constant, so
        projection does NOT read the flange pose (the camera is desk-fixed).
        """
        from types import SimpleNamespace

        from jiuwensymbiosis.utils.geometry import apply_transform, pixel_and_depth_to_camera_xyz

        ll = self._ll()
        frames = ll.grab_frames()
        if frames is None:
            return {"ok": False, "reason": "no_camera", "object": object_name}
        rgb, depth_img_m = frames

        self._ensure_detector()
        det = detect_and_centroid(
            rgb=rgb,
            depth_img_m=depth_img_m,
            seg_fn=self._seg_fn,
            object_name=object_name,
            tcp_at_grab=SimpleNamespace(x=0.0, y=0.0, z=0.0, r=0.0),
        )
        if not det.get("ok"):
            return det  # type: ignore[return-value]

        u, v, depth_m = det["u"], det["v"], det["depth_m"]
        best = det["best"]
        img_w, img_h = det["img_shape"]
        mask_h, mask_w = det["mask_shape"]

        if ll.tf_base_cam is None:
            raise RuntimeError("get_grasp_info_simple needs a loaded eye-to-hand calibration (set calib_path in YAML).")
        intrinsics, intrinsics_src = self._resolve_intrinsics(ll)

        # Eye-to-hand: constant T_base_cam, no flange read.
        xyz_raw = apply_transform(
            ll.tf_base_cam,
            pixel_and_depth_to_camera_xyz((u, v), depth_m, intrinsics),
        )
        calib = getattr(ll, "calibration", None)
        xy_transform = calib.get("xy_transform") if calib is not None else None
        xy_corr = calib.get("xy_correction_mm") if (calib is not None and xy_transform is None) else None
        xyz_final, corr_desc = apply_xy_correction(
            np.asarray(xyz_raw, dtype=np.float64),
            xy_transform=xy_transform,
            xy_correction_mm=xy_corr,
        )
        if self._z_correction_mm:
            xyz_final = np.asarray(xyz_final, dtype=np.float64).copy()
            xyz_final[2] += self._z_correction_mm
            corr_desc = f"{corr_desc}+z{self._z_correction_mm:+.0f}"

        logger.info(
            "[grasp-debug] K_src=%s eye_to_hand T_base_cam "
            "raw_xyz_mm=(%.2f, %.2f, %.2f) corr=%s "
            "final_xyz_mm=(%.2f, %.2f, %.2f)",
            intrinsics_src,
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
                tcp_grab=SimpleNamespace(x=0.0, y=0.0, z=0.0, r=0.0),
                tcp_proj=SimpleNamespace(x=0.0, y=0.0, z=0.0, r=0.0),
                xyz_raw=np.asarray(xyz_raw, dtype=np.float64),
                xyz_final=np.asarray(xyz_final, dtype=np.float64),
                xy_corr=xy_corr,
                xy_transform=xy_transform,
                intrinsics_src=intrinsics_src,
                intrinsics=np.asarray(intrinsics, dtype=float).reshape(-1).tolist(),
                img_shape=(img_w, img_h),
                mask_shape=(mask_w, mask_h),
                extra_info={
                    "eye_to_hand": True,
                    "frame_model": "so101_eye_to_hand_T_base_cam",
                },
            )
        except Exception as exc:  # noqa: BLE001 - debug dump must never break a grasp
            logger.debug("[grasp-debug] dump failed: %s", exc)

        top_z = float(xyz_final[2])
        z_floor = self.env.z_min_safe
        grasp_z = top_z + self._grasp_z_offset_mm
        if z_floor is not None:
            grasp_z = max(grasp_z, float(z_floor))
        place_z = top_z + self._chip_thickness_mm
        x_f, y_f = float(xyz_final[0]), float(xyz_final[1])
        logger.info(
            "[So101Api] %s: pos=(%.1f, %.1f, %.1f) grasp_z=%.1f place_z=%.1f score=%.2f",
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
    def analyze_scene(self, object_name: str | None = None) -> dict:
        """Scene analysis grounded on ``object_name`` (detection counts + scores)."""
        target = object_name or "object"
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
