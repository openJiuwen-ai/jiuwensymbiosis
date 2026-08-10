# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Capability mixins.

Each mixin declares one capability string and the methods that make up that
capability. A concrete API class inherits from the mixins it supports.

The methods are decorated with ``@robot_tool`` already, so subclasses need
NOT decorate the overrides — the metadata propagates with the function.

Default behaviour
-----------------
Motion / joint / grasp methods, plus ``get_image``, ship **working default
implementations that delegate to the Env contract verbs** (``self.env.home`` /
``move_to_flange`` / ``move_joint`` / ``set_end_effector`` / ``get_flange_pose``
/ ``home_pose`` / ``tool_offset_mm`` / ``grab_rgb``). A robot whose body matches
the common case (top-down tip, tip == flange, two-state end effector) therefore
needs to write *no* api code for these — composing the mixins is enough. Override
a method only when the body has special geometry (e.g. a tilted tool, a tool offset).

The *high-level vision* methods (``get_grasp_info_simple`` / ``pixel_to_base_xyz``
/ ``analyze_scene``) cannot have a generic default — they depend on the adapter's
detector client and hand-eye calibration — so they stay abstract and raise
``NotImplementedError`` until the adapter provides them.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from jiuwensymbiosis.api.decorators import robot_tool
from jiuwensymbiosis.perception.detector_client import init_detector
from jiuwensymbiosis.perception.vision import (
    GraspFailure,
    GraspResult,
    apply_xy_correction,
    build_grasp_result,
    detect_and_centroid,
    dump_grasp_debug,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # Mixins are composed into BaseRobotApi subclasses, which set `self.env`.
    # Declared here only for type checking; runtime attribute is provided by
    # the composing class (see BaseRobotApi.__init__).
    from jiuwensymbiosis.env.base import BaseRobotEnv


def _pose_to_dict(pose: Any) -> dict:
    """Best-effort vendor-pose object → dict.

    Tolerates both the SCARA convention (``.r``) and the 6-DoF convention
    (``.rx/.ry/.rz``); only the fields the pose actually exposes are emitted.
    """
    out: dict[str, float] = {}
    for key in ("x", "y", "z", "rx", "ry", "rz", "r"):
        val = getattr(pose, key, None)
        if val is not None:
            out[key] = float(val)
    return out


# Base-frame unit offsets per direction word (arm faces +x; +y is left; +z up).
_DIRECTION_OFFSETS: dict[str, tuple[float, float, float]] = {
    "forward": (1.0, 0.0, 0.0),
    "back": (-1.0, 0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
    "up": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
}


def _check_translation_bounds(env: Any, x: float, y: float, z: float) -> None:
    """Reject a target below ``env.z_min_safe`` or outside ``env.workspace_bounds``."""
    if not all(math.isfinite(value) for value in (x, y, z)):
        raise ValueError(f"move blocked: target coordinates must be finite, got x={x}, y={y}, z={z}")
    z_floor = getattr(env, "z_min_safe", None)
    if z_floor is not None and z < float(z_floor):
        raise ValueError(f"move blocked: z={z:.1f} below z_min_safe={float(z_floor):.1f}")
    bounds = getattr(env, "workspace_bounds", None)
    if bounds is not None:
        xmin, ymin, xmax, ymax = bounds
        if not xmin <= x <= xmax:
            raise ValueError(f"move blocked: x={x:.1f} out of [{xmin}, {xmax}]")
        if not ymin <= y <= ymax:
            raise ValueError(f"move blocked: y={y:.1f} out of [{ymin}, {ymax}]")


# =============================================================================
# Motion
# =============================================================================
class MotionMixin:
    """Cartesian motion capability mixin."""

    env: BaseRobotEnv  # provided by the composing BaseRobotApi subclass
    capability = "motion.cartesian"

    @robot_tool(desc="Return to the home pose. Always safe.", tags=["motion"])
    def home(self) -> None:
        """Return to the home pose (delegates to the Env verb)."""
        self.env.home()

    @robot_tool(desc="Get current end-effector pose in mm/deg, base frame.")
    def get_pose(self) -> dict:
        """Current pose. Default reports the flange pose (assumes tip == flange;
        override when a tool offset applies).
        """
        return _pose_to_dict(self.env.get_flange_pose())

    @robot_tool(desc="Get the home pose constants (read-only) for this robot.")
    def get_home_pose(self) -> dict:
        """Home pose constants, read from the env."""
        return _pose_to_dict(self.env.home_pose)

    @robot_tool(
        desc="Move the end-effector tip to absolute (x, y, z[, r]) in mm/deg, base frame. "
        "If r is omitted, current r is preserved.",
        tags=["motion"],
    )
    def goto_xyzr(self, x: float, y: float, z: float, r: float | None = None) -> None:
        """Move the tip to an absolute Cartesian pose. Default is a top-down move
        (tip == flange, rx=180, ry=0); override for a tool offset or a tilted tool.
        """
        if r is None:
            cur = self.env.get_flange_pose()
            r = getattr(cur, "rz", getattr(cur, "r", 0.0))
        self.env.move_to_flange(SimpleNamespace(x=float(x), y=float(y), z=float(z), rx=180.0, ry=0.0, rz=float(r)))

    @robot_tool(
        desc=(
            "Move the end-effector a relative distance in one cardinal direction. "
            "direction ∈ {forward, back, left, right, up, down} (forward=+x, left=+y, up=+z, base frame); "
            "distance_mm is a positive number of millimetres. E.g. '往左移两厘米' → move_direction('left', 20). "
            "Orientation is preserved."
        ),
        tags=["motion"],
    )
    def move_direction(self, direction: str, distance_mm: float) -> dict:
        """Relative Cartesian nudge with a self-contained bounds check.

        ``move_direction`` is NOT in SafetyRail's watch set (the rail only sees
        ``goto_xyzr`` at the tool layer), so it validates the target against
        ``env.z_min_safe`` / ``env.workspace_bounds`` here and raises
        ``ValueError`` on a violation — the agent sees that and self-corrects.
        A relative translation is frame-agnostic, so tool offset does not matter.
        """
        key = (direction or "").strip().lower()
        if key not in _DIRECTION_OFFSETS:
            raise ValueError(f"unknown direction {direction!r}; expected one of {sorted(_DIRECTION_OFFSETS)}")
        dist = float(distance_mm)
        if not math.isfinite(dist) or dist <= 0:
            raise ValueError("distance_mm must be finite and positive; the direction controls the sign")
        ux, uy, uz = _DIRECTION_OFFSETS[key]
        cur = self.env.get_flange_pose()
        tx, ty, tz = cur.x + ux * dist, cur.y + uy * dist, cur.z + uz * dist
        _check_translation_bounds(self.env, tx, ty, tz)
        target = SimpleNamespace(
            x=tx,
            y=ty,
            z=tz,
            rx=getattr(cur, "rx", 180.0),
            ry=getattr(cur, "ry", 0.0),
            rz=getattr(cur, "rz", getattr(cur, "r", 0.0)),
        )
        self.env.move_to_flange(target)
        return {"ok": True, "direction": key, "distance_mm": dist, "pose": {"x": tx, "y": ty, "z": tz}}


class JointMotionMixin:
    """Joint-space motion capability mixin."""

    env: BaseRobotEnv  # provided by the composing BaseRobotApi subclass
    capability = "motion.joint"

    @robot_tool(desc="Move to a joint configuration q (rad or deg per robot convention).", tags=["motion"])
    def move_joint(self, q: list[float]) -> None:
        """Move to a joint configuration (delegates to the Env verb)."""
        self.env.move_joint(q)


# =============================================================================
# Grasp
# =============================================================================
class SuctionMixin:
    """Suction grasp capability mixin."""

    env: BaseRobotEnv  # provided by the composing BaseRobotApi subclass
    capability = "grasp.suction"

    @robot_tool(desc="Turn suction ON. Should be called only after the tip is on/near the target.", tags=["grasp"])
    def activate_suction(self) -> dict:
        """Turn suction ON (delegates to the Env end-effector verb)."""
        self.env.set_end_effector(True)
        return {"ok": True, "state": "on"}

    @robot_tool(desc="Turn suction OFF — releases whatever is held.", tags=["grasp"])
    def deactivate_suction(self) -> dict:
        """Turn suction OFF (delegates to the Env end-effector verb)."""
        self.env.set_end_effector(False)
        return {"ok": True, "state": "off"}


class ParallelGripperMixin:
    """Parallel gripper capability mixin."""

    env: BaseRobotEnv  # provided by the composing BaseRobotApi subclass
    capability = "grasp.parallel"

    @robot_tool(desc="Open the parallel gripper to width_mm.", tags=["grasp"])
    def open_gripper(self, width_mm: float = 80.0) -> dict:
        """Open the gripper (delegates to the Env end-effector verb). ``width_mm``
        is accepted for API parity; bodies with width control override this.
        """
        self.env.set_end_effector(False)
        return {"ok": True, "state": "open"}

    @robot_tool(desc="Close the parallel gripper, optionally with a target force in N.", tags=["grasp"])
    def close_gripper(self, force_n: float | None = None) -> dict:
        """Close the gripper (delegates to the Env end-effector verb). ``force_n``
        is accepted for API parity; bodies with force control override this.
        """
        self.env.set_end_effector(True)
        return {"ok": True, "state": "closed"}


# =============================================================================
# Vision
# =============================================================================
class VisionMixin:
    """Vision and object detection capability mixin.

    ``get_image`` / ``get_grasp_info_simple`` / ``pixel_to_base_xyz`` ship working
    defaults: the full grab → detect → project → correct → grasp/place pipeline
    lives here and delegates only the vendor-specific "pixel + depth → RAW base
    XYZ" projection to :meth:`_project_pixel_to_base_raw` (eye-in-hand reads the
    live flange; eye-to-hand uses a constant ``T_base_cam``). ``analyze_scene``
    still needs the adapter's detector wiring and stays abstract.
    """

    env: BaseRobotEnv  # provided by the composing BaseRobotApi subclass

    capability = "vision.detection"

    # Detector wiring + grasp/place geometry constants. Adapters set these in
    # __init__; the class-level defaults let a common-case body compose the mixin
    # with no vision code beyond the projection seam.
    _detector_service_url: str = "http://127.0.0.1:8114"
    _seg_fn: Callable[..., list[dict]] | None = None
    _z_correction_mm: float = 0.0
    _grasp_z_offset_mm: float = -25.0
    _place_z_offset_mm: float = 75.0
    _floor_margin_mm: float = 0.0

    # ---- vendor-specific projection seam ------------------------------------
    def _project_pixel_to_base_raw(self, u: float, v: float, depth_m: float) -> np.ndarray:
        """Project (pixel + metric depth) → RAW base-frame XYZ (mm), NO correction.

        The one vendor-specific step: eye-in-hand composes
        ``tf_base_flange(live) @ tf_flange_cam``; eye-to-hand uses a constant
        ``tf_base_cam``. Must NOT apply xy/z correction — that is owned by the
        shared geometry so it runs exactly once. Adapters must implement this.
        """
        raise NotImplementedError

    def _vision_driver(self) -> Any:
        """The low-level driver for vision reads (``grab_frames`` / ``calibration``).

        A controlled penetration point (see ``env/base.py``): typed ``Any`` because
        the concrete driver satisfies CameraDriver / VisionDriver structurally, not
        the base ``RobotDriver``. Raises if the env is not connected.
        """
        ll = self.env.low_level
        if ll is None:
            raise RuntimeError(f"{type(self).__name__}: env not connected. Use `with session:` / session.connect().")
        return ll

    def _ensure_detector(self) -> None:
        """Lazy-init the detector segmentation function if not already bound."""
        if self._seg_fn is not None:
            return
        try:
            self._seg_fn = init_detector(self._detector_service_url)
            logger.info("[VisionMixin] detector client bound to %s", self._detector_service_url)
        except Exception as exc:  # detector init best-effort; tools degrade to ok=False
            logger.warning("[VisionMixin] detector init failed (%s); detection tools will return ok=False.", exc)

    def _grasp_debug_tcp(self) -> Any:
        """TCP snapshot used ONLY for diagnostic dumps (needs ``.x/.y/.z/.r``).

        Eye-in-hand bodies report the live flange; eye-to-hand bodies (where the
        flange is irrelevant to projection) override this to return zeros.
        """
        p = self.env.get_flange_pose()
        return SimpleNamespace(x=p.x, y=p.y, z=p.z, r=getattr(p, "rz", getattr(p, "r", 0.0)))

    def _grasp_debug_extra(self) -> dict:
        """Adapter-specific fields merged into the grasp-debug JSON.

        Base default records the concrete API class; adapters override to add
        frame-model / live-pose context.
        """
        return {"api_class": self.__class__.__name__}

    @robot_tool(
        desc="One-shot: detect `object_name` in the live frame, project "
        "to base XYZ via depth+calibration. Returns "
        '{"ok": bool, "object": str, "position": [x,y,z]_mm, "grasp_z": float, '
        '"grasp_position": [x,y,z]_mm, "place_z": float, "place_position": [x,y,z]_mm, '
        '"score": float, "pixel_uv": [u,v], "depth_m": float}.',
    )
    def get_grasp_info_simple(self, object_name: str) -> GraspResult | GraspFailure:
        """Detect an object and return its 3D grasp/place geometry.

        The full pipeline lives here; adapters supply only the projection seam
        :meth:`_project_pixel_to_base_raw`.
        """
        return cast("GraspResult | GraspFailure", self._grasp_info_with_intermediates(object_name)[0])

    def _grasp_info_with_intermediates(self, object_name: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Shared grasp pipeline; returns ``(result, intermediates | None)``.

        ``intermediates`` carries the detection/projection snapshots so an adapter
        fast-path (e.g. SO-101 tracking) can reuse the same detection instead of
        re-running it; the public tool ignores it. ``None`` on any failure.
        """
        ll = self._vision_driver()
        frames = ll.grab_frames()
        if frames is None:
            return {"ok": False, "reason": "no_camera", "object": object_name}, None
        rgb, depth_img_m = frames

        tcp = self._grasp_debug_tcp()
        self._ensure_detector()
        det = detect_and_centroid(
            rgb=rgb,
            depth_img_m=depth_img_m,
            seg_fn=self._seg_fn,
            object_name=object_name,
            tcp_at_grab=tcp,
        )
        if not det.get("ok"):
            return det, None

        u, v, depth_m = det["u"], det["v"], det["depth_m"]
        best = det["best"]
        img_w, img_h = det["img_shape"]
        mask_h, mask_w = det["mask_shape"]

        xyz_raw = np.asarray(self._project_pixel_to_base_raw(u, v, depth_m), dtype=np.float64)
        calib = getattr(ll, "calibration", None)
        result, xyz_final = build_grasp_result(
            object_name=object_name,
            best=best,
            u=u,
            v=v,
            depth_m=depth_m,
            xyz_raw=xyz_raw,
            calib=calib,
            z_correction_mm=self._z_correction_mm,
            grasp_z_offset_mm=self._grasp_z_offset_mm,
            place_z_offset_mm=self._place_z_offset_mm,
            z_floor=self.env.z_min_safe,
            floor_margin_mm=self._floor_margin_mm,
        )
        logger.info(
            "[grasp-debug] %s: raw_xyz_mm=(%.2f, %.2f, %.2f) final_xyz_mm=(%.2f, %.2f, %.2f) "
            "grasp_z=%.1f place_z=%.1f score=%.2f",
            object_name,
            float(xyz_raw[0]),
            float(xyz_raw[1]),
            float(xyz_raw[2]),
            float(xyz_final[0]),
            float(xyz_final[1]),
            float(xyz_final[2]),
            result["grasp_z"],
            result["place_z"],
            result["score"],
        )
        try:
            dump_grasp_debug(
                rgb=rgb,
                object_name=object_name,
                best=best,
                u=u,
                v=v,
                depth_m=depth_m,
                tcp_grab=tcp,
                tcp_proj=tcp,
                xyz_raw=xyz_raw,
                xyz_final=xyz_final,
                xy_corr=calib.get("xy_correction_mm") if calib is not None else None,
                xy_transform=calib.get("xy_transform") if calib is not None else None,
                img_shape=(img_w, img_h),
                mask_shape=(mask_w, mask_h),
                extra_info=self._grasp_debug_extra(),
            )
        except Exception as exc:  # debug dump must never break a grasp
            logger.debug("[grasp-debug] dump failed: %s", exc)

        intermediates = {
            "rgb": rgb,
            "depth_img_m": depth_img_m,
            "best": best,
            "u": u,
            "v": v,
            "depth_m": depth_m,
            "xyz_raw": xyz_raw,
            "xyz_final": xyz_final,
        }
        return result, intermediates

    @robot_tool(
        desc="Pixel (u,v) at depth_m (meters) → base-frame XYZ in mm. Requires a loaded calibration.",
    )
    def pixel_to_base_xyz(self, u: float, v: float, depth_m: float) -> dict:
        """Reproject a pixel to base-frame XYZ (raw projection + xy correction).

        Mirrors the projection step of :meth:`get_grasp_info_simple` but WITHOUT
        the constant ``z_correction_mm`` (that is applied for grasp geometry only),
        matching the standalone tool's historical behavior.
        """
        xyz = self._project_pixel_to_base_raw(u, v, depth_m)
        calib = getattr(self.env.low_level, "calibration", None)
        if calib is not None:
            xyz, _desc = apply_xy_correction(
                np.asarray(xyz, dtype=np.float64),
                xy_transform=calib.get("xy_transform"),
                xy_correction_mm=calib.get("xy_correction_mm"),
            )
        return {"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])}

    @robot_tool(desc="Grab the latest RGB frame as numpy HxWx3 (rarely needed by the agent itself).")
    def get_image(self) -> Any:
        """Latest RGB frame, or None if no camera (delegates to the env)."""
        return self.env.grab_rgb()

    @robot_tool(desc="Higher-level scene analysis with prompt grounded on object_name.")
    def analyze_scene(self, object_name: str | None = None) -> dict:
        """Scene analysis grounded on ``object_name``.

        No generic default: requires the adapter's detector client.
        """
        raise NotImplementedError
