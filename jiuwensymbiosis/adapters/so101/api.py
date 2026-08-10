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
- ``goto_pose`` takes a nested ``pose: So101Pose`` value object (six fields
  describing one Cartesian pose). ``SafetyRail.before_tool_call`` unpacks the
  nested ``pose`` object to read ``x``/``y``/``z`` for Z-floor / XY-bound
  checks, so the rail still fires automatically without top-level scalars.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from jiuwensymbiosis.adapters.so101.geometry import So101Pose
from jiuwensymbiosis.adapters.so101.lowlevel import So101PreDispatchError
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.api.decorators import robot_tool
from jiuwensymbiosis.api.mixins import (
    JointMotionMixin,
    MotionMixin,
    ParallelGripperMixin,
    VisionMixin,
)
from jiuwensymbiosis.utils.geometry import apply_transform, pixel_and_depth_to_camera_xyz

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

    # The low-level servo maintains a previous-planned-pose / previous-IK-seed
    # chain. Tell the generic fast controller to slew from its previous command
    # as well, so the outer loop does not re-anchor each waypoint to encoder FK.
    servo_slew_from_last_command = True
    # SO-101 is underactuated (5 DoF): orientation remains a best-effort IK
    # command, but fast-path completion is based on live XYZ only. This avoids
    # requiring an independently unattainable rz while preserving the generic
    # angular reached check for every other adapter.
    servo_reached_angular_keys: tuple[str, ...] = ()
    # At the SO-101's configured 20 Hz rate, per-tick INFO logs are affordable
    # and essential for distinguishing planned convergence from live TCP lag.
    servo_log_ticks = True

    def __init__(
        self,
        env: So101Env,
        *,
        detector_service_url: str = "http://127.0.0.1:8114",
        z_correction_mm: float = 0.0,
        grasp_z_offset_mm: float = -25.0,
        place_z_offset_mm: float = 75.0,
        floor_margin_mm: float = 0.0,
    ) -> None:
        super().__init__(env)
        self._detector_service_url = detector_service_url
        self._seg_fn: Callable[..., list[dict[str, Any]]] | None = None
        self._z_correction_mm = float(z_correction_mm)
        self._grasp_z_offset_mm = float(grasp_z_offset_mm)
        self._place_z_offset_mm = float(place_z_offset_mm)
        # Extra clearance above (z_min_safe + this) enforced by the shared grasp
        # geometry; wired from cfg.minimum_floor_margin_mm (SO-101 desk model).
        self._floor_margin_mm = float(floor_margin_mm)

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
        return self._gripper_response("open")

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
        return self._gripper_response("closed")

    def _gripper_response(self, default_state: str) -> dict:
        """Merge driver gripper detail into the API response."""
        response: dict[str, Any] = {"ok": True, "state": default_state}
        detail = self.env.last_gripper_result
        if detail:
            response.update(detail)
        return response

    def is_grasp_confirmed(self, result: Mapping[str, Any] | None = None) -> bool:
        """Whether the last close stopped on object contact.

        This is a private fast-runner hook, deliberately not a ``robot_tool``.
        Reaching the configured fully-closed position means the gripper closed
        on empty space; only the driver's conservative contact state confirms a
        payload.
        """
        detail = result if isinstance(result, Mapping) else self.env.last_gripper_result
        return bool(detail and detail.get("ok") is not False and detail.get("state") == "contact")

    def retreat_home(self) -> None:
        """Return home through the payload-aware SO-101 retreat path."""
        self._ll().retreat_home()

    def recovery_home(self) -> None:
        """Return home through the SO-101 recovery retreat path."""
        self._ll().recovery_home()

    # --- cartesian overrides -------------------------------------------------
    @robot_tool(
        desc=(
            "Move the SO-101 control frame to absolute (x, y, z[, r]) in mm/deg, base frame. "
            "orientation_policy is preserve (keep live orientation), grasp (use the calibrated "
            "grasp orientation), or top_down (legacy rx=180, ry=0). If omitted, the configured "
            "policy is used; r overrides the selected rz."
        ),
        capability="motion.cartesian",
        tags=["motion"],
    )
    def goto_xyzr(
        self,
        x: float,
        y: float,
        z: float,
        r: float | None = None,
        orientation_policy: str | None = None,
    ) -> None:
        """Move the control frame to ``(x, y, z[, r])`` mm/deg, base frame.

        SO-101 is a 5-DoF underactuated arm: position is the strong constraint,
        orientation is best-effort (the IK solver weights position higher via
        ``ik_orientation_weight``). General motion defaults to preserving the
        live orientation instead of silently imposing a top-down pose.
        """
        rx, ry, rz = self._resolve_orientation(orientation_policy, rz_override=r)
        self.goto_pose(So101Pose(x=float(x), y=float(y), z=float(z), rx=rx, ry=ry, rz=rz))

    def servo_to_tip(self, pose: dict) -> bool:
        """Issue one non-blocking servo command toward a TIP-frame pose.

        SO-101 milestone-A geometry has ``tool_offset_mm == 0`` and uses the
        same base/control frame for the tip and flange.  The real-time runner
        uses ``rz`` internally to match :meth:`get_pose`; ``r`` remains an
        accepted compatibility alias for callers using the SCARA convention.
        """
        if not isinstance(pose, Mapping):
            raise TypeError(f"servo_to_tip pose must be a mapping, got {type(pose).__name__}.")
        x = float(pose["x"])
        y = float(pose["y"])
        z = float(pose["z"])
        policy = pose.get("orientation_policy")
        explicit_rz = pose.get("rz", pose.get("r"))
        rx, ry, rz = self._resolve_orientation(
            policy,
            rz_override=explicit_rz,
            rx_override=pose.get("rx"),
            ry_override=pose.get("ry"),
        )
        dispatched = self.env.servo_to_flange({"x": x, "y": y, "z": z, "rx": rx, "ry": ry, "rz": rz})
        return dispatched is not False

    def _resolve_orientation(
        self,
        policy: Any,
        *,
        rz_override: Any = None,
        rx_override: Any = None,
        ry_override: Any = None,
    ) -> tuple[float, float, float]:
        """Resolve a complete orientation without imposing an accidental pose."""
        cur = self.env.get_flange_pose()
        current = (
            float(getattr(cur, "rx", 180.0)),
            float(getattr(cur, "ry", 0.0)),
            float(getattr(cur, "rz", getattr(cur, "r", 0.0))),
        )
        selected = policy
        if selected is None:
            selected = getattr(self.env.cfg, "cartesian_orientation_policy", "preserve")
        if not isinstance(selected, str):
            raise So101PreDispatchError(f"orientation_policy must be a string, got {type(selected).__name__}.")
        selected = selected.strip().lower()

        if selected == "preserve":
            base = current
        elif selected == "top_down":
            base = (180.0, 0.0, current[2])
        elif selected == "grasp":
            configured = getattr(self.env.cfg, "grasp_orientation", None)
            if configured is None:
                raise So101PreDispatchError(
                    "orientation_policy='grasp' requires cfg.grasp_orientation with calibrated rx/ry/rz values."
                )
            try:
                base = (float(configured["rx"]), float(configured["ry"]), float(configured["rz"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise So101PreDispatchError("cfg.grasp_orientation must contain numeric rx/ry/rz values.") from exc
        else:
            raise So101PreDispatchError(
                f"orientation_policy must be one of ['grasp', 'preserve', 'top_down'], got {selected!r}."
            )

        try:
            values = (
                base[0] if rx_override is None else float(rx_override),
                base[1] if ry_override is None else float(ry_override),
                base[2] if rz_override is None else float(rz_override),
            )
        except (TypeError, ValueError) as exc:
            raise So101PreDispatchError("orientation overrides must be numeric.") from exc
        if not all(np.isfinite(value) for value in values):
            raise So101PreDispatchError(f"resolved orientation must be finite, got {values!r}.")
        return values

    @robot_tool(
        desc=(
            "Move the SO-101 control frame to absolute (x, y, z, rx, ry, rz) "
            "in mm/deg, base frame. Position is strongly constrained; orientation "
            "is best-effort (5-DoF underactuated arm). SafetyRail checks x/y/z "
            "bounds before this runs (reads the nested pose object)."
        ),
        capability="motion.cartesian",
        tags=["motion"],
        input_params={
            "type": "object",
            "properties": {
                "pose": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "z": {"type": "number"},
                        "rx": {"type": "number"},
                        "ry": {"type": "number"},
                        "rz": {"type": "number"},
                    },
                    "required": ["x", "y", "z", "rx", "ry", "rz"],
                }
            },
            "required": ["pose"],
        },
    )
    def goto_pose(self, pose: So101Pose) -> None:
        """Move to an absolute Cartesian pose.

        Accepts a ``So101Pose`` or a dict with ``x/y/z/rx/ry/rz`` keys (the
        LLM-facing tool schema and ``RobotControlTool`` deliver a dict/JSON
        object). SafetyRail unpacks the nested ``pose`` object to read
        ``x``/``y``/``z`` for Z-floor / XY-bound checks before this runs.
        """
        if isinstance(pose, dict):
            pose = So101Pose(**pose)
        logger.info(
            "[So101Api] goto_pose -> (x=%.2f, y=%.2f, z=%.2f, rx=%.2f, ry=%.2f, rz=%.2f)",
            pose.x,
            pose.y,
            pose.z,
            pose.rx,
            pose.ry,
            pose.rz,
        )
        self.env.move_to_flange(pose)

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

    def _project_pixel_to_base_raw(self, u: float, v: float, depth_m: float) -> np.ndarray:
        """Eye-to-hand RAW projection: ``p_base = tf_base_cam @ p_cam`` (constant T_base_cam).

        The vendor-specific seam VisionMixin delegates to; applies NO xy/z
        correction (the shared geometry owns that). ``tf_base_cam`` is desk-fixed,
        so this does NOT read the flange pose.
        """
        ll = self._ll()
        if ll.tf_base_cam is None:
            raise RuntimeError("get_grasp_info_simple needs a loaded eye-to-hand calibration (set calib_path in YAML).")
        intrinsics, _src = self._resolve_intrinsics(ll)
        return apply_transform(
            ll.tf_base_cam,
            pixel_and_depth_to_camera_xyz((float(u), float(v)), float(depth_m), intrinsics),
        )

    def _grasp_debug_tcp(self) -> Any:
        """Eye-to-hand: the flange pose is irrelevant to projection, so the debug
        dump records zeros (matches the historical SO-101 grasp-debug artifacts)."""
        from types import SimpleNamespace

        return SimpleNamespace(x=0.0, y=0.0, z=0.0, r=0.0)

    def _grasp_debug_extra(self) -> dict:
        """SO-101 eye-to-hand debug marker."""
        return {"eye_to_hand": True, "frame_model": "so101_eye_to_hand_T_base_cam"}

    def get_grasp_tracking_sample(self, object_name: str) -> dict[str, Any]:
        """Private SO101 fast-path sample including mask/depth-quality metadata.

        This method intentionally has no ``@robot_tool`` decorator: it is an
        adapter-to-runner hook, not an LLM/API action. It reuses the shared
        VisionMixin pipeline (so the public :meth:`get_grasp_info_simple` result
        is unchanged) and appends SO-101-private tracking fields.
        """
        result, intermediates = self._grasp_info_with_intermediates(object_name)
        if intermediates is not None:
            result.update(
                self._tracking_metadata(
                    best=intermediates["best"],
                    depth_img_m=intermediates["depth_img_m"],
                    u=float(intermediates["u"]),
                    v=float(intermediates["v"]),
                )
            )
        return result

    @staticmethod
    def _tracking_metadata(
        *,
        best: Mapping[str, Any],
        depth_img_m: np.ndarray,
        u: float,
        v: float,
    ) -> dict[str, Any]:
        """Build SO101-private mask/depth quality fields."""
        # Use the same 11x11 image-grid window as the centroid depth lookup.
        # A hand/gripper crossing that window produces a large percentile span
        # and is rejected before contaminated XYZ replaces the trusted target.
        dep_h, dep_w = depth_img_m.shape[:2]
        cu, cv, radius = int(round(u)), int(round(v)), 5
        x0, x1 = max(0, cu - radius), min(dep_w, cu + radius + 1)
        y0, y1 = max(0, cv - radius), min(dep_h, cv + radius + 1)
        patch = np.asarray(depth_img_m[y0:y1, x0:x1], dtype=np.float64)
        valid = patch[(patch > 0.0) & np.isfinite(patch)]
        if valid.size:
            p10, p90 = np.percentile(valid, [10.0, 90.0])
            depth_span_mm = float((p90 - p10) * 1000.0)
            valid_ratio = float(valid.size) / float(patch.size)
        else:  # detect_and_centroid already guards this; keep fail-closed.
            depth_span_mm = float("inf")
            valid_ratio = 0.0
        return {
            "_tracking_mask": np.asarray(best["mask"], dtype=bool).copy(),
            "_tracking_depth_span_mm": depth_span_mm,
            "_tracking_valid_depth_ratio": valid_ratio,
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
