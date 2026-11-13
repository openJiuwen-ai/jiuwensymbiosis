# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""``So101Api`` — what the SO-101 does, action by action.

Every method binds one entry of the shared action vocabulary with
``@implements(SPEC)``; the generic ones forward to ``api.defaults``. Vision is
desktop-fixed eye-to-hand (milestone B): a RealSense D405 bolted to the desk,
NOT the wrist. The hand-eye calibration therefore solves ``T_base_cam`` (a
CONSTANT camera-in-base transform), so projection does NOT read the flange pose
per step — ``p_base = T_base_cam @ p_cam`` — unlike piper's eye-in-hand
``tf_base_flange @ tf_flange_cam``.

Two places where this body departs from the generic implementation:

- ``open_gripper`` / ``close_gripper`` keep the contract's ``width_mm`` /
  ``force_n`` params but ignore them — the SO-101 gripper is two-state
  percentage, not mm/N. The contract already calls both a HINT, so this needs no
  per-body caveat.
- ``goto_pose`` takes a nested ``pose: So101Pose`` value object (six fields
  describing one Cartesian pose). ``SafetyRail.before_tool_call`` unpacks the
  nested ``pose`` object to read ``x``/``y``/``z`` for Z-floor / XY-bound
  checks, so the rail still fires automatically without top-level scalars.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from jiuwensymbiosis.adapters.so101.geometry import So101Pose
from jiuwensymbiosis.adapters.so101.lowlevel import So101PreDispatchError
from jiuwensymbiosis.agent.cancel import RunCancelled
from jiuwensymbiosis.api import defaults
from jiuwensymbiosis.api.actions import (
    ANALYZE_SCENE,
    CLOSE_GRIPPER,
    GET_GRASP_INFO_SIMPLE,
    GET_HOME_POSE,
    GET_IMAGE,
    GET_POSE,
    GOTO_POSE,
    GOTO_XYZR,
    MOVE_DIRECTION,
    MOVE_JOINT,
    OPEN_GRIPPER,
    PIXEL_TO_BASE_XYZ,
    implements,
)
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.api.reachability import Reachability
from jiuwensymbiosis.contracts import GraspFailure, GraspResult
from jiuwensymbiosis.errors import InferenceServiceError
from jiuwensymbiosis.perception.config import DetectorConfig
from jiuwensymbiosis.perception.detector_client import (
    check_detection_completion,
    create_detector_client,
    init_detector,
    segment_image,
)
from jiuwensymbiosis.perception.vision import (
    annotate_detection_overlay,
    apply_xy_correction,
    detect_and_centroid,
    dump_grasp_debug,
    mask_pixels_and_depths,
    select_top_surface_grasp,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jiuwensymbiosis.adapters.so101.env import So101Env

logger = logging.getLogger(__name__)

__all__ = ["So101Api"]


class So101Api(BaseRobotApi):
    """SO-101 5-DoF arm + parallel gripper + desktop eye-to-hand vision.

    Every action this body offers is declared below, so this file IS the capability
    list. Generic ones forward to ``api.defaults``; the rest are SO-101 specifics
    (orientation policy, two-state percentage gripper, eye-to-hand calibration).
    """

    # Marker capabilities: real things this body can do that no ACTION advertises, so the
    # class attr is the only way to declare them (see BaseRobotApi.capabilities). Leaving one
    # out is not cosmetic — the agent gates on ``api.capabilities & env.capabilities``, so an
    # ability the hardware has and the api forgets to claim is silently switched off.
    #   motion.servo        — servo_to_tip streams non-blocking tip targets, so the fast path
    #                         can FOLLOW a moving target rather than aim once.
    #   vision.eye_to_hand  — the camera is bolted to the desk, not the wrist, so a detection
    #                         is already an ABSOLUTE base-frame point; tracking uses that
    #                         directly instead of the eye-in-hand relative correction.
    #   vision.depth        — the D405 returns aligned depth.
    # planning.reachability is NOT declared here: it is derived from whether the body can
    # actually answer (a judge on the Api ∩ a URDF + arm_chains on the Env). This body holds
    # the generic judge but its Env exposes no URDF, so the judge degrades to "unknown" —
    # declaring it by hand used to make that claim anyway.
    capability = {"motion.servo", "vision.eye_to_hand", "vision.depth"}

    # ---- generic actions: the Env delegation is the whole implementation ----
    # Planning-time proprioception: read by the planner (agent/run.py), never by the LLM.
    def check_reachable(self, target: Any) -> bool | None:
        return self._reach.check_reachable(target)

    def describe_reach(self) -> dict | None:
        return self._reach.describe_reach()

    @implements(GET_POSE)
    def get_pose(self) -> dict:
        return defaults.get_pose(self)

    @implements(GET_HOME_POSE)
    def get_home_pose(self) -> dict:
        return defaults.get_home_pose(self)

    @implements(MOVE_DIRECTION)
    def move_direction(self, direction: str, distance_mm: float) -> dict:
        return defaults.move_direction(self, direction, distance_mm)

    @implements(MOVE_JOINT)
    def move_joint(self, targets: dict[str, float]) -> Any:
        return defaults.move_joint(self, targets)

    @implements(GET_IMAGE)
    def get_image(self):
        return defaults.get_image(self)

    # The low-level servo maintains a previous-planned-pose / previous-IK-seed
    # chain. Tell the generic fast controller to slew from its previous command
    # as well, so the outer loop does not re-anchor each waypoint to encoder FK.
    servo_slew_from_last_command = True
    # SO-101 is underactuated (5 DoF): orientation remains a best-effort IK
    # command, but fast-path completion is based on live XYZ only. This avoids
    # requiring an independently unattainable rz while preserving the generic
    # angular reached check for every other adapter.
    servo_reached_angular_keys: tuple[str, ...] = ()
    # At the SO-101's configured 5 Hz rate, per-tick INFO logs are affordable
    # and essential for distinguishing planned convergence from live TCP lag.
    servo_log_ticks = True

    def __init__(
        self,
        env: So101Env,
        *,
        detector_service_url: str | None = "http://127.0.0.1:8114",
        detector_client: Any = None,
        z_correction_mm: float = 0.0,
        grasp_z_offset_mm: float = -25.0,
        place_z_offset_mm: float = 75.0,
        grasp_top_surface_enabled: bool = False,
        grasp_top_band_mm: float = 15.0,
        grasp_top_percentile: float = 90.0,
        grasp_min_points: int = 30,
        grasp_mask_erode_px: int = 2,
    ) -> None:
        super().__init__(env)
        # Held, not inherited: any body that ships a URDF gets the same judge.
        self._reach = Reachability(self)
        self._detector_service_url = detector_service_url
        self._seg_fn: Callable[..., list[dict[str, Any]]] | None = detector_client
        self._z_correction_mm = float(z_correction_mm)
        self._grasp_z_offset_mm = float(grasp_z_offset_mm)
        self._place_z_offset_mm = float(place_z_offset_mm)
        # Top-surface grasp point (opt-in via cfg; default off keeps centroid behaviour).
        self._grasp_top_surface_enabled = bool(grasp_top_surface_enabled)
        self._grasp_top_band_mm = float(grasp_top_band_mm)
        self._grasp_top_percentile = float(grasp_top_percentile)
        self._grasp_min_points = int(grasp_min_points)
        self._grasp_mask_erode_px = int(grasp_mask_erode_px)
        # Last annotated detection image (RGB ndarray), for the GUI to show per
        # detection step. Built only on the public tool path, never in the
        # tracking hot loop. Consumed via pop_last_detection_overlay.
        self._last_detection_overlay: Any = None

    # --- gripper overrides (two-state percentage, no mm/N params) ------------
    @implements(OPEN_GRIPPER)
    def open_gripper(self, width_mm: float = 80.0) -> dict:
        """Open the gripper. ``width_mm`` is accepted for API parity and ignored —
        the SO-101 gripper is a two-state percentage actuator (no width control).
        Maps to ``env.set_end_effector(False)`` -> driver sends ``gripper_open_pos``.
        """
        del width_mm  # ignored: SO-101 is two-state, no width calibration yet.
        self.env.set_end_effector(False)
        return self._gripper_response("open")

    @implements(CLOSE_GRIPPER)
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

        This is a private fast-runner hook, deliberately not an action.
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
    @implements(GOTO_XYZR)
    def goto_xyzr(
        self,
        x: float,
        y: float,
        z: float,
        r: float | None = None,
        orientation_policy: Literal["preserve", "top_down", "grasp"] | None = None,
    ) -> None:
        """Move the control frame to ``(x, y, z[, r])`` mm/deg, base frame.

        SO-101 is a 5-DoF underactuated arm: position is the strong constraint,
        orientation is best-effort (the IK solver weights position higher via
        ``ik_orientation_weight``). General motion defaults to preserving the
        live orientation instead of silently imposing a top-down pose — the
        ``Literal`` above is what puts that choice, and this body's default, in
        front of the planner instead of leaving it a config-file surprise.
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

    @implements(GOTO_POSE)
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

    def _ensure_detector(self) -> None:
        """Bind the legacy URL only for directly constructed, unmanaged APIs."""
        if self._seg_fn is None:
            self._seg_fn = (
                create_detector_client(DetectorConfig())
                if self._detector_service_url is None
                else init_detector(self._detector_service_url, request_scoped=True)
            )

    @implements(PIXEL_TO_BASE_XYZ)
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

    def _project_mask_pixels_to_base_raw(self, us: np.ndarray, vs: np.ndarray, depths_m: np.ndarray) -> np.ndarray:
        """Eye-to-hand batch projection: ``(N,3)`` base points via the constant ``tf_base_cam``.

        The batch form of the single-pixel projection in :meth:`_compute_grasp_info` —
        one constant ``tf_base_cam @ p_cam`` applied to the whole masked point cloud
        (no flange read). Used by the top-surface grasp path.
        """
        from jiuwensymbiosis.utils.geometry import apply_transform, pixels_and_depths_to_camera_xyz

        ll = self._ll()
        if ll.tf_base_cam is None:
            raise RuntimeError("top-surface grasp needs a loaded eye-to-hand calibration (set calib_path in YAML).")
        intrinsics, _src = self._resolve_intrinsics(ll)
        return apply_transform(ll.tf_base_cam, pixels_and_depths_to_camera_xyz(us, vs, depths_m, intrinsics))

    def _project_base_to_pixel(self, xyz_base: Any) -> tuple[float, float] | None:
        """Eye-to-hand inverse projection: base-frame XYZ (mm) → pixel (u, v), or None.

        Marks the actual top-surface grasp point on the GUI diagnostic overlay.
        ``p_cam = inv(tf_base_cam) @ p_base``; pixel via the pinhole intrinsics.
        Returns None when calibration/intrinsics are missing or the point is behind
        the camera (non-positive depth).
        """
        from jiuwensymbiosis.utils.geometry import apply_transform, invert_transform

        ll = self._ll()
        if ll.tf_base_cam is None:
            return None
        intrinsics, _src = self._resolve_intrinsics(ll)
        tf_cam_base = invert_transform(np.asarray(ll.tf_base_cam, dtype=np.float64))
        p_cam = apply_transform(tf_cam_base, np.asarray(xyz_base, dtype=np.float64))
        z = float(p_cam[2])
        if not np.isfinite(z) or z <= 0.0:
            return None
        k = np.asarray(intrinsics, dtype=np.float64)
        u = float(k[0, 0]) * float(p_cam[0]) / z + float(k[0, 2])
        v = float(k[1, 1]) * float(p_cam[1]) / z + float(k[1, 2])
        return (u, v)

    def _maybe_top_surface_grasp(self, best: dict, depth_img_m: np.ndarray) -> dict[str, float] | None:
        """Yaw-only bounding-box grasp point from the masked point cloud, or None.

        Returns ``None`` (caller then uses the centroid pixel) when the feature is
        off or there are too few valid points to trust the box — a graceful
        degrade, never worse than the default path.
        """
        if not self._grasp_top_surface_enabled:
            return None
        us, vs, depths_m = mask_pixels_and_depths(best, depth_img_m, erode_px=self._grasp_mask_erode_px)
        if us.size < self._grasp_min_points:
            return None
        points_base = np.asarray(self._project_mask_pixels_to_base_raw(us, vs, depths_m), dtype=np.float64)
        return select_top_surface_grasp(
            points_base,
            band_mm=self._grasp_top_band_mm,
            top_percentile=self._grasp_top_percentile,
            min_points=self._grasp_min_points,
        )

    def pop_last_detection_overlay(self) -> Any:
        """Return and clear the last annotated detection image (RGB ndarray) or None.

        A non-action accessor for the GUI: the run page pops the overlay after a
        detection step to show it. Clearing prevents a stale overlay leaking to a
        later, non-detection step.
        """
        overlay = self._last_detection_overlay
        self._last_detection_overlay = None
        return overlay

    def _stash_detection_overlay(
        self, rgb: Any, best: dict, centroid_uv: tuple[float, float], result: dict, top_surface: dict | None
    ) -> None:
        """Build + stash the annotated detection image (best-effort; never raises)."""
        try:
            grasp_uv = centroid_uv
            if top_surface is not None:
                reproj = self._project_base_to_pixel(result["position"])
                if reproj is not None:
                    grasp_uv = reproj
            self._last_detection_overlay = annotate_detection_overlay(
                rgb, best, centroid_uv=centroid_uv, grasp_uv=grasp_uv
            )
        except Exception as exc:  # overlay is a diagnostic; never break a grasp
            logger.debug("[detection-overlay] build failed: %s", exc)
            self._last_detection_overlay = None

    @implements(GET_GRASP_INFO_SIMPLE)
    def get_grasp_info_simple(self, object_name: str) -> GraspResult | GraspFailure:
        """Detect an object and return its 3D grasp/place geometry (eye-to-hand).

        Pipeline: grab -> detect_and_centroid -> eye-to-hand back-project ->
        xy-correct -> grasp/place geometry. ``tf_base_cam`` is a constant, so
        projection does NOT read the flange pose (the camera is desk-fixed).
        """
        result, _tracking = self._compute_grasp_info(object_name, include_tracking=False, build_overlay=True)
        return cast(GraspResult | GraspFailure, result)

    def get_grasp_tracking_sample(self, object_name: str) -> dict[str, Any]:
        """Private SO101 fast-path sample including mask/depth-quality metadata.

        This method intentionally carries no ``ActionSpec``: it is an
        adapter-to-runner hook, not an LLM/API action.  The public
        :meth:`get_grasp_info_simple` response remains JSON-compatible and
        unchanged.
        """
        result, tracking = self._compute_grasp_info(object_name, include_tracking=True)
        if tracking is not None:
            result.update(tracking)
        return result

    def _compute_grasp_info(
        self,
        object_name: str,
        *,
        include_tracking: bool,
        build_overlay: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Compute public grasp geometry plus optional private tracking data.

        ``build_overlay`` annotates the detection for the GUI; the tracking hot
        loop leaves it off.
        """
        from types import SimpleNamespace

        from jiuwensymbiosis.utils.geometry import apply_transform, pixel_and_depth_to_camera_xyz

        ll = self._ll()
        captured_monotonic_s = time.monotonic()
        frames = ll.grab_frames()
        if frames is None:
            return {"ok": False, "reason": "no_camera", "object": object_name}, None
        rgb, depth_img_m = frames

        self._ensure_detector()
        det = detect_and_centroid(
            rgb=rgb,
            depth_img_m=depth_img_m,
            seg_fn=self._seg_fn,
            captured_monotonic_s=captured_monotonic_s,
            object_name=object_name,
            tcp_at_grab=SimpleNamespace(x=0.0, y=0.0, z=0.0, r=0.0),
        )
        if not det.get("ok"):
            return dict(det), None

        u, v, depth_m = det["u"], det["v"], det["depth_m"]
        best = det["best"]
        img_w, img_h = det["img_shape"]
        mask_h, mask_w = det["mask_shape"]

        if ll.tf_base_cam is None:
            raise RuntimeError("get_grasp_info_simple needs a loaded eye-to-hand calibration (set calib_path in YAML).")
        intrinsics, intrinsics_src = self._resolve_intrinsics(ll)

        # Eye-to-hand: constant T_base_cam, no flange read.
        top_surface = self._maybe_top_surface_grasp(best, depth_img_m)
        if top_surface is not None:
            xyz_raw = np.asarray([top_surface["x"], top_surface["y"], top_surface["z_top"]], dtype=np.float64)
        else:
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
            grasp_z = max(grasp_z, float(z_floor) + float(self.env.cfg.minimum_floor_margin_mm))
        place_z = top_z + self._place_z_offset_mm
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
        result: dict[str, Any] = {
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
        if top_surface is not None:
            result["grasp_rz"] = top_surface["rz"]
            result["grasp_width_mm"] = top_surface["width_mm"]
        if build_overlay:
            self._stash_detection_overlay(rgb, best, (float(u), float(v)), result, top_surface)
        tracking = None
        if include_tracking:
            tracking = self._tracking_metadata(
                best=best,
                depth_img_m=depth_img_m,
                u=float(u),
                v=float(v),
            )
        check_detection_completion(self._seg_fn, captured_monotonic_s, invalidate=self.invalidate_sensing_cache)
        return result, tracking

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

    @implements(ANALYZE_SCENE)
    def analyze_scene(self, object_name: str | None = None) -> dict:
        """Scene analysis grounded on ``object_name`` (detection counts + scores)."""
        target = object_name or "object"
        captured_monotonic_s = time.monotonic()
        rgb = self.get_image()
        if rgb is None:
            return {"ok": False, "reason": "no_camera"}
        self._ensure_detector()
        if self._seg_fn is None:
            return {"ok": False, "reason": "detector_unavailable"}
        try:
            results = segment_image(
                self._seg_fn,
                rgb,
                target,
                captured_monotonic_s=captured_monotonic_s,
            )
        except RunCancelled:
            raise
        except InferenceServiceError as exc:
            self.invalidate_sensing_cache()
            return {"ok": False, "reason": "detector_unavailable", "error_code": exc.code}
        except Exception as exc:  # noqa: BLE001 - surface detector failure as ok=False
            return {"ok": False, "reason": str(exc)}
        scores = sorted((float(r.get("score", 0.0)) for r in results), reverse=True)
        # Shared meaning is "every instance"; this body has no per-instance depth, so an
        # entry carries score + pixel only — enough for a planner to size a multi-target loop.
        objects = [
            {"object": target, "score": float(r.get("score", 0.0)), "pixel_uv": r.get("center") or r.get("pixel_uv")}
            for r in results
        ]
        check_detection_completion(self._seg_fn, captured_monotonic_s, invalidate=self.invalidate_sensing_cache)
        return {
            "ok": True,
            "object": target,
            "count": len(objects),
            "objects": objects,
            "n_detections": len(results),
            "top_scores": scores[:5],
        }
