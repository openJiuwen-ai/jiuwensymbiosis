# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Agent-facing Cruzr API."""

from __future__ import annotations

import logging
from typing import Any, Optional

from jiuwensymbiosis.adapters.cruzr._calibration import load_cruzr_camera_calib
from jiuwensymbiosis.adapters.cruzr.env import CruzrEnv
from jiuwensymbiosis.api import defaults
from jiuwensymbiosis.api.actions import (
    ANALYZE_SCENE,
    APPROACH_FOR_GRASP,
    APPROACH_FOR_PLACE,
    LOCATE_FOR_GRASP,
    LOCATE_FOR_PLACE,
    SEARCH_TARGET,
    DRIVE_ARC,
    DUAL_ARM_GRASP,
    DUAL_ARM_PLACE,
    GET_IMAGE,
    GET_JOINT_POSITIONS,
    HOME,
    LIFT_TO_CLEARANCE,
    MOVE_NAMED_JOINT,
    NAVIGATE_RELATIVE,
    PIXEL_TO_BASE_XYZ,
    ROTATE_BASE,
    SET_LIFT_POSE,
    TURN_WAIST,
    implements,
)
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.api.decorators import robot_tool
from jiuwensymbiosis.api.components import Approach, Scene3D
from jiuwensymbiosis.kinematics.urdf_chain import parse_chain
from jiuwensymbiosis.motion import approach
from jiuwensymbiosis.perception import scene3d
from jiuwensymbiosis.perception.detector_client import init_detector
from jiuwensymbiosis.perception.frame import project_to_base
from jiuwensymbiosis.perception.vision import detect_and_centroid

logger = logging.getLogger(__name__)


# The approach geometry and the surface-footprint payload now live in the framework layer (every
# mobile body driving up to a target needs the same maths); kept under the old names because they
# are imported by name elsewhere.
_forward_step = approach.forward_step
_grasp_forward_step = approach.grasp_forward_step
_place_forward_step = approach.place_forward_step
_grasp_near_face_normal = approach.near_face_normal
_select_grasp_normal = approach.select_grasp_normal
_select_surface_square_normal = approach.select_surface_square_normal
_surface_footprint_fields = scene3d.surface_footprint_fields


class CruzrApi(BaseRobotApi):
    """Cruzr mobile dual-arm: base + lifter + waist + paddle grasp + waist RGBD vision.

    Every action this body offers is declared in this file, so it IS the capability
    list. The three remaining base classes are not action bundles: Scene3D and Approach
    are stateful components with body hooks, and Reachability answers a planning
    question rather than exposing an action.
    """

    # Scene 3-D sensing runs off the waist RGBD camera (the head camera is wide-FOV 2-D only).
    _scene_camera = "waist"

    # Marker capabilities: real things this body can do that no ACTION advertises, so the
    # class attr is the only way to declare them (see BaseRobotApi.capabilities).
    #   planning.reachability — check_reachable below weighs both arms plus an adaptive
    #                           lifter, so this body brings its own judge rather than the
    #                           generic single-arm one.
    capability = {"planning.reachability"}

    def __init__(
        self,
        env: CruzrEnv,
        *,
        detector_service_url: str = "http://127.0.0.1:8114",
        camera_calib_path: Optional[str] = None,
    ) -> None:
        """Bind to a Cruzr environment; configure detector + camera calibration."""
        super().__init__(env)
        # Held, not inherited: this body declares each sensing action below and forwards to
        # the component, so api.py stays the complete list of what Cruzr offers.
        self._scene = Scene3D(self)
        self._nav = Approach(self)
        self._detector_service_url = detector_service_url
        self._camera_calib_path = camera_calib_path
        self._seg_fn = None
        self._calib_cache = None
        self._calib_loaded = False
        # Last box geometry successfully grasped by dual_arm_grasp. dual_arm_place() falls
        # back to this when called without an explicit box, so an LLM agent need
        # not round-trip the (large, nested) box dict back as a tool argument.
        self._last_grasped_box: Optional[dict] = None
        # waist_yaw angle at the moment of the last successful grasp. dual_arm_place()
        # uses (current_waist - this) to rotate its paddle targets, so a box moved
        # by turn_waist between grasp and place is released at its NEW location.
        self._last_grasp_waist_yaw: Optional[float] = None
        # Last successful locate_for_grasp result. dual_arm_grasp() consumes this so
        # detection is an explicit, separate step (not embedded in the grasp) and
        # the LLM need not echo the geometry dict back as a tool argument.
        self._last_detection: Optional[dict] = None
        # Last successful locate_for_place result. dual_arm_place() consumes this when
        # called without a surface, so an LLM can call locate_for_place() then
        # dual_arm_place() without echoing the (nested) table-footprint dict back.
        self._last_surface: Optional[dict] = None
        # Lazily-created debug detection window (waist+head panes). None until the first
        # detection; stays a no-op unless cfg.viz_detections / $JIUWEN_CRUZR_VIZ=1.
        self._viz = None

    @robot_tool(
        desc="Raise Cruzr's left arm once and return it to the home position.",
        tags=["motion"],
        invalidates=["body.home"],
    )
    def raise_left_arm(self) -> dict:
        """Raise the left arm and lower it back home."""
        logger.info("[CruzrApi] raise_left_arm")
        return self._ll().raise_arm_blocking(arm="left", return_home=True)

    @robot_tool(
        desc="Raise Cruzr's right arm once and return it to the home position. Uses the configured mirrored right-arm target.",
        tags=["motion"],
        invalidates=["body.home"],
    )
    def raise_right_arm(self) -> dict:
        """Raise the right arm and lower it back home."""
        logger.info("[CruzrApi] raise_right_arm")
        return self._ll().raise_arm_blocking(arm="right", return_home=True)

    @robot_tool(
        desc="Lower Cruzr's left arm to the configured home position.",
        tags=["motion"],
        invalidates=["body.home"],
    )
    def lower_left_arm(self) -> dict:
        """Lower the left arm to home."""
        logger.info("[CruzrApi] lower_left_arm")
        return self._ll().home(arm="left")

    @robot_tool(
        desc="Lower Cruzr's right arm to the configured home position.",
        tags=["motion"],
        invalidates=["body.home"],
    )
    def lower_right_arm(self) -> dict:
        """Lower the right arm to home."""
        logger.info("[CruzrApi] lower_right_arm")
        return self._ll().home(arm="right")

    @robot_tool(
        desc=(
            "Raise one Cruzr arm. arm can be 'left' or 'right'. "
            "If target_rad is omitted, uses the configured per-arm target: left=+1.0, right=-1.0 by default."
        ),
        tags=["motion"],
        invalidates=["body.home"],
    )
    def raise_arm(
        self,
        arm: str = "left",
        return_home: bool = True,
        target_rad: Optional[float] = None,
    ) -> dict:
        """Raise the selected arm."""
        logger.info("[CruzrApi] raise_arm arm=%s return_home=%s target=%s", arm, return_home, target_rad)
        return self._ll().raise_arm_blocking(arm=arm, return_home=return_home, target_rad=target_rad)

    @implements(MOVE_NAMED_JOINT)
    def move_named_joint(self, joint_name: str, position_rad: float) -> dict:
        """Move a named joint to a target position."""
        logger.info("[CruzrApi] move_named_joint %s -> %.3f", joint_name, position_rad)
        return self._ll().move_joint_blocking({joint_name: float(position_rad)})

    @implements(TURN_WAIST)
    def turn_waist(self, delta_rad: float) -> dict:
        """Rotate ``waist_yaw`` by ``delta_rad``, holding both arms fixed.

        ``waist_yaw`` is proximal to both arms in the URDF, so turning it swings
        the whole upper body (arms + any grasped box) rigidly about the vertical
        axis — the grip is invariant and no arm IK is re-solved. ``delta_rad`` is
        applied in joint space (``+`` follows the URDF waist_yaw axis). The target
        is clamped to the URDF ``waist_yaw`` limit.
        """
        from jiuwensymbiosis.adapters.cruzr.geometry import ARM_JOINTS
        from jiuwensymbiosis.kinematics.urdf_chain import parse_chain

        cfg = self.env.cfg
        waist = cfg.waist_yaw_joint
        q = self._ll().get_joint_positions() or {}
        if waist not in q:
            return {"ok": False, "reason": "no_joint_state"}
        lo, hi = parse_chain(cfg.urdf_path, "base_link", cfg.left_arm_leaf).limits()[waist]
        frm = float(q[waist])
        target = frm + float(delta_rad)
        clamped = target < lo or target > hi
        target = min(max(target, lo), hi)
        hold = {j: float(q[j]) for a in ("left", "right") for j in ARM_JOINTS[a] if j in q}
        logger.info("[CruzrApi] turn_waist from=%.3f delta=%.3f -> %.3f clamped=%s",
                    frm, delta_rad, target, clamped)
        res = self._ll().turn_waist_blocking(target, hold=hold, waist_joint=waist)
        return {"ok": True, "joint": waist, "from_rad": frm, "to_rad": target,
                "delta_rad": target - frm, "clamped": clamped, "readback": res.get("readback")}

    def set_head(self, yaw_rad: float, pitch_rad: float) -> dict:
        """Move the head yaw+pitch via the CONTINUOUS SDK path (move_joints ramp).

        The head only actuates under a sustained high-rate RobotCommand stream;
        a single ``move_named_joint`` pulse does NOT move it. Sign (verified live):
        ``+pitch = look up, -pitch = look down`` (look down to keep a floor box in
        the high-mounted head camera's view as the base nears).
        """
        cfg = self.env.cfg
        logger.info("[CruzrApi] set_head yaw=%.3f pitch=%.3f", yaw_rad, pitch_rad)
        return self._ll().move_joints_blocking(
            {
                cfg.head_yaw_joint: float(yaw_rad),
                cfg.head_pitch_joint: float(pitch_rad),
            },
            ramp_duration_s=float(getattr(cfg, "head_ramp_duration_s", cfg.ramp_duration_s)),
        )

    # No ``move_joint``: the shared action means the FULL joint vector, and this body's
    # driver maps a bare list onto the default arm's shoulder pitch only. Single-joint
    # moves go through ``move_named_joint``, which says what it does.

    @implements(GET_JOINT_POSITIONS)
    def get_joint_positions(self) -> dict:
        """Return latest joint positions keyed by joint name."""
        return self._ll().get_joint_positions()

    @implements(HOME)
    def home(self) -> dict:
        """Retreat to a safe home, doing the minimum motion the current state needs.

        This is also what the generic ``RecoveryRail`` calls after a tool exception, so
        Cruzr needs no bespoke recovery rail: a gripperless dual-arm body simply makes
        its own ``home()`` safe. The staged implementation lives in ``_home_safely``.
        """
        return self._home_safely()

    # ---- generic actions: the Env delegation is the whole implementation ----
    @implements(SET_LIFT_POSE)
    def set_lift_pose(self, q_lifter: dict) -> dict:
        return defaults.set_lift_pose(self, q_lifter)

    # ============================================================  Vision
    # Body-specific 2-D debug view of the detector; the plannable answer is
    # ``locate_for_grasp``. Capability stated explicitly now that the class no longer
    # carries a blanket ``capability`` attribute.
    @robot_tool(
        desc="Debug view of the detector: 2-D box, score, centre depth and base-frame point for object_name in the waist RGBD camera.",
        capability="vision.detection",
        tags=["vision"],
        produces_location=True,
        returns={"type": "object", "properties": {
            "ok": {"type": "boolean"}, "reason": {"type": "string"}, "object": {"type": "string"},
            "score": {"type": "number"}, "bbox": {"type": "array"},
            "depth_m": {"type": "number"}, "position": {"type": "array"},
        }},
    )
    def detect(self, object_name: str = "box", camera_name: str = "waist_rgbd") -> dict:
        """Detect an object in the waist camera and project its centroid to base XYZ."""
        frames = self._ll().grab_frames(camera="waist")
        if frames is None:
            return {"ok": False, "reason": "no_camera", "camera_name": camera_name}
        rgb, depth_m, k_live, tf_live = frames
        if depth_m is None:
            return {"ok": False, "reason": "no_depth", "camera_name": camera_name}

        self._ensure_detector()
        det = detect_and_centroid(
            rgb=rgb,
            depth_img_m=depth_m,
            seg_fn=self._seg_fn,
            object_name=object_name,
            tcp_at_grab=_NullPose(),
        )
        if not det.get("ok"):
            det.setdefault("camera_name", camera_name)
            return det

        u, v, depth = det["u"], det["v"], det["depth_m"]
        best = det["best"]
        box_2d = [float(b) for b in best["box"][:4]]
        score = float(best["score"])

        intrinsics = k_live if k_live is not None else self._calib_intrinsics()
        tf_base_cam = tf_live if tf_live is not None else self._calib_extrinsics()
        position = None
        position_reason = None
        if intrinsics is None:
            position_reason = "no_intrinsics"
        elif tf_base_cam is None:
            position_reason = "no_extrinsics"
        else:
            position = [float(c) for c in project_to_base((u, v), depth, intrinsics, tf_base_cam)]

        logger.info(
            "[CruzrApi] detect %s: box=%s score=%.2f depth=%.3f pos=%s",
            object_name, box_2d, score, depth, position,
        )
        return {
            "ok": True,
            "object": object_name,
            "camera_name": camera_name,
            "box_2d": box_2d,
            "score": score,
            "pixel_uv": [u, v],
            "depth_m": depth,
            "position": position,
            "position_reason": position_reason,
        }

    # ------------------------------------------------------ Scene3D body hooks
    def _grab_calibrated_frame(self, camera: Optional[str] = None) -> Any:
        """One waist RGBD frame as a ``CameraFrame``. Goes through the driver rather than
        ``CruzrEnv.grab_calibrated_frame`` because it must apply the static-calib intrinsics
        fallback (intrinsics are pose-invariant, so that fallback is safe; extrinsics are NOT
        — a missing live TF stays None and the mixin fails loudly)."""
        from jiuwensymbiosis.perception.frame import CameraFrame

        frames = self._ll().grab_frames(camera=camera or "waist")
        if frames is None:
            return None
        rgb, depth_m, k_live, tf_live = frames
        return CameraFrame(rgb=rgb, depth_m=depth_m,
                           intrinsics=k_live if k_live is not None else self._calib_intrinsics(),
                           tf_base_cam=tf_live)

    def _detector_seg_fn(self) -> Any:
        """Lazily bind the detection sidecar, then hand the mixin its segmentation callable."""
        self._ensure_detector()
        return self._seg_fn

    def _viz_update(self, camera: str, prompt: str, rgb: Any, best: Optional[dict]) -> None:
        """Push one detection frame to the (lazily-created) debug window. No-op unless enabled via
        cfg.viz_detections or $JIUWEN_CRUZR_VIZ=1. ``best`` is a _run_detect_pick_best-style dict
        ({mask, box, score, ok}) or None for a miss."""
        if self._viz is None:
            import os

            from jiuwensymbiosis.adapters.cruzr._viz import DetectionViz

            enabled = (bool(getattr(self.env.cfg, "viz_detections", False))
                       or os.environ.get("JIUWEN_CRUZR_VIZ") == "1")
            self._viz = DetectionViz(enabled=enabled)
        ok = bool(best and best.get("ok") is not False and best.get("mask") is not None)
        self._viz.update(camera, prompt, rgb,
                         mask=(best or {}).get("mask") if ok else None,
                         box=(best or {}).get("box") if ok else None,
                         score=(best or {}).get("score") if ok else None, ok=ok)

    # ---------------------------------------------------------------- 3-D sensing
    @implements(LOCATE_FOR_GRASP)
    def locate_for_grasp(self, object_name: str = "box", reference: Optional[str] = None,
                         relation: str = "on") -> dict:
        return self._scene.locate_for_grasp(object_name, reference, relation)

    @implements(LOCATE_FOR_PLACE)
    def locate_for_place(self, object_name: str = "table", reference: Optional[str] = None,
                         relation: str = "on") -> dict:
        return self._scene.locate_for_place(object_name, reference, relation)

    @implements(ANALYZE_SCENE)
    def analyze_scene(self, object_name: str = "box") -> dict:
        return self._scene.analyze_scene(object_name)

    # ------------------------------------------------------------ search + approach
    @implements(SEARCH_TARGET)
    def search_target(self, object_name: str = "box", reference: Optional[str] = None,
                      relation: str = "on") -> dict:
        return self._nav.search_target(object_name, reference, relation)

    @implements(APPROACH_FOR_GRASP)
    def approach_for_grasp(self, object_name: str = "box", reference: Optional[str] = None,
                           relation: str = "on") -> dict:
        return self._nav.approach_for_grasp(object_name, reference, relation)

    @implements(APPROACH_FOR_PLACE)
    def approach_for_place(self, object_name: str = "table", reference: Optional[str] = None,
                           relation: str = "on") -> dict:
        return self._nav.approach_for_place(object_name, reference, relation)

    # ------------------------------------------------------ Approach body hooks
    def _base_driver(self) -> Any:
        """Drive the servo worker through the driver directly (the SDK handle is its own)."""
        return self._ll()

    def _nav_relative(self, dx_m: float, dy_m: float, dyaw_rad: float, **gains: Any) -> dict:
        """Relative base move that forwards the gentle approach-only steering gains."""
        return self._ll().navigate_relative(float(dx_m), float(dy_m), float(dyaw_rad), **gains)

    def _search_frames(self, camera: Optional[str] = None) -> Any:
        """One raw frame tuple from ``camera``. The head is a stereo pair read as its RIGHT eye
        alone — no depth on that path, which costs nothing here: looking around wants a bearing."""
        return self._ll().grab_frames(camera=camera or "waist")

    def _sweep_for_bearing(self, object_name: str, on: Optional[str] = None) -> dict:
        """Pan the HEAD left + right over the current facing — cruzr can aim a camera without
        moving the base, so it looks around with the neck before anything else turns."""
        return _acquire_with_head(self, object_name, self.env.cfg, on=on)

    def _reset_search_sensor(self) -> None:
        """Re-centre the head to its forward pose."""
        _reset_head(self, self.env.cfg)

    @implements(NAVIGATE_RELATIVE)
    def navigate_relative(self, dx_m: float, dy_m: float = 0.0, dyaw_rad: float = 0.0) -> dict:
        """Move the base by a relative offset via SDK wheel-velocity + odom closed loop."""
        logger.info("[CruzrApi] navigate_relative dx=%.3f dy=%.3f dyaw=%.3f", dx_m, dy_m, dyaw_rad)
        return self._ll().navigate_relative(float(dx_m), float(dy_m), float(dyaw_rad))

    @implements(ROTATE_BASE)
    def rotate_base(self, dyaw_rad: float) -> dict:
        """Rotate the base in place by ``dyaw_rad`` (dx=dy=0). Does NOT touch the arms."""
        logger.info("[CruzrApi] rotate_base dyaw=%.3f", dyaw_rad)
        return self._ll().navigate_relative(0.0, 0.0, float(dyaw_rad))

    @implements(DRIVE_ARC)
    def drive_arc(self, radius_m: float, dyaw_rad: float) -> dict:
        """Drive ONE constant-curvature arc (``radius_m``, signed ``dyaw_rad``) via the low-level ``--arc``
        mode. Bring-up / calibration tool: no perception — use it in open space to measure the realized
        radius vs commanded and tune ``arc_curv_gain``/``arc_k_fwd`` before ever enabling
        ``grasp_arc_enabled``. Does NOT touch the arms."""
        logger.info("[CruzrApi] drive_arc radius=%.3f dyaw=%.3f", radius_m, dyaw_rad)
        return self._ll().navigate_arc(float(radius_m), float(dyaw_rad))


    def check_reachable(self, target: dict) -> bool:
        """Override of ``Reachability.check_reachable`` with cruzr's exact DUAL-ARM + lifter judge:
        当前底盘/关节姿态下，双臂能否抓到该 base 系目标(允许自适应 lifter，不含底盘移动)。纯离线只读，
        复用 dual_arm_grasp 的 IK 可达搜索，无运动/无硬件副作用。供 fast 规划器判断"要不要先编长距离移动"。
        保守快筛：不确定/无关节状态/出错一律判不可达(保留 approach_for_grasp 兜底，对误判鲁棒)。target 为
        analyze_scene 形状的目标 dict(含 center_mm)。"""
        from jiuwensymbiosis.adapters.cruzr.geometry import LIFTER_JOINTS, search_lifter_for_box
        from jiuwensymbiosis.kinematics.urdf_chain import parse_chain
        from jiuwensymbiosis.perception.object_geometry import ObjectGeometry3D

        try:
            c = target.get("center_mm")
            if not (isinstance(c, (list, tuple)) and len(c) == 3):
                return False
            fixed_names = ("lifter_pitch_1_joint", "lifter_pitch_2_joint",
                           "lifter_pitch_3_joint", "waist_yaw_joint")
            q = self._ll().get_joint_positions()
            if not q or any(n not in q for n in fixed_names):
                return False   # 无关节状态 → 保守判不可达
            cfg = self.env.cfg
            # scene 只有 center/宽/高/forward，缺 front_x/top_z/back_x → 近似构造(depth≈width)。
            # 保守快筛，不追求精确；真正的精解仍由执行期 dual_arm_grasp 负责。
            forward = float(target.get("forward_mm", c[0]))
            width = float(target.get("width_mm", 0.0))
            height = float(target.get("height_mm", 0.0))
            box = ObjectGeometry3D(
                ok=True, reason="", center_mm=(float(c[0]), float(c[1]), float(c[2])),
                width_mm=width, height_mm=height,
                front_x_mm=forward - width / 2.0, top_z_mm=float(c[2]) + height / 2.0,
                n_points=500, back_x_mm=forward + width / 2.0,
            )
            left = parse_chain(cfg.urdf_path, "base_link", cfg.left_arm_leaf)
            right = parse_chain(cfg.urdf_path, "base_link", cfg.right_arm_leaf)
            current_lifter = {j: q[j] for j in LIFTER_JOINTS}
            lp = search_lifter_for_box(box, left, right, current_lifter,
                                       q["waist_yaw_joint"], ik_max_iters=300)
            return bool(lp.found)
        except Exception as exc:  # noqa: BLE001 — best-effort precheck; any failure → not reachable
            logger.debug("[CruzrApi] reachability precheck skipped: %s", exc)
            return False

    @implements(DUAL_ARM_GRASP)
    def dual_arm_grasp(self, target: Optional[dict] = None, object_name: str = "box") -> dict:
        """Plan dual-arm grasp IK for an ALREADY-DETECTED box, clamp, and verify
        FT contact. The pick-up/clearance lift is delegated to
        ``lift_to_clearance`` — this method does not lift.

        Detection is NOT performed here — call ``locate_for_grasp`` first. ``box`` is
        the geometry dict it returns; when omitted, the most recent cached
        ``locate_for_grasp`` result is used (so the LLM need not echo the dict back).
        Returns ``{"ok": False, "reason": "no_detection"}`` if neither is available.

        Sequence: resolve box → parse URDF chains → read q_fixed → adaptive lifter
        → solve_grasp; abort if plan not ok. Move pre-grasp (open) then clamp
        (inward). Read FT on both hands — return
        ``{"ok": False, "reason": "no_contact"}`` if either hand reports
        fmag < contact_force_threshold_n.
        """
        from jiuwensymbiosis.adapters.cruzr.geometry import (
            LIFTER_JOINTS,
            search_lifter_for_box,
            solve_arm_ik,
            solve_grasp,
        )
        from jiuwensymbiosis.kinematics.urdf_chain import parse_chain
        from jiuwensymbiosis.perception.object_geometry import ObjectGeometry3D

        # Invalidate any prior grasp up front: only a successful grasp below
        # re-populates it, so dual_arm_place() (which falls back to this) no-ops after
        # a failed grasp instead of placing a box we are not holding.
        self._last_grasped_box = None
        self._last_grasp_waist_yaw = None
        self._last_surface = None   # a new grasp invalidates any previously sensed surface
        self.env.holding_payload = False

        cfg = self.env.cfg
        left = parse_chain(cfg.urdf_path, "base_link", cfg.left_arm_leaf)
        right = parse_chain(cfg.urdf_path, "base_link", cfg.right_arm_leaf)
        chains = {"left": left, "right": right}
        fixed_names = ("lifter_pitch_1_joint", "lifter_pitch_2_joint",
                       "lifter_pitch_3_joint", "waist_yaw_joint")

        def _box_of(dd: dict) -> ObjectGeometry3D:
            return ObjectGeometry3D(
                True, "", tuple(dd["center_mm"]), dd["width_mm"], dd["height_mm"],
                dd["front_x_mm"], dd["top_z_mm"], dd["n_points"],
                back_x_mm=dd.get("back_x_mm", 0.0),
            )

        def _read_fixed() -> dict | None:
            q = self._ll().get_joint_positions()
            if not q or any(n not in q for n in fixed_names):
                return None
            return {k: q[k] for k in fixed_names}

        det = target if isinstance(target, dict) else self._last_detection
        if not det or not det.get("ok"):
            return {"ok": False, "reason": "no_detection"}
        box = _box_of(det)
        q_fixed = _read_fixed()
        if q_fixed is None:
            return {"ok": False, "reason": "no_joint_state"}

        # Adaptive lifter: pick the body pose that best reaches this box. If the
        # current pose can't grasp it, search the level-torso manifold.
        current_lifter = {j: q_fixed[j] for j in LIFTER_JOINTS}
        lp = search_lifter_for_box(box, left, right, current_lifter, q_fixed["waist_yaw_joint"])
        if not lp.found:
            return {"ok": False, "reason": lp.reason or "unreachable_any_lifter"}

        # Raise to the "ready" posture (hands up in front) BEFORE moving the
        # lifter: the arms stay clear of the table while the body leans, and the
        # home->grasp transition does not sweep up into the table from below.
        # Non-contact transit moves (ready, descend-outside-faces) run on the fast transit ramp; the
        # clamp below runs on the dedicated (faster-than-global) contact ramp — kept separate so its
        # closing speed can be tuned against deform / contact-force accuracy on its own.
        transit_ramp = getattr(cfg, "arm_transit_ramp_duration_s", None)
        contact_ramp = getattr(cfg, "arm_contact_ramp_duration_s", None)
        ready = self._ready_arm_q(chains, q_fixed)
        self._move_arms_sync(ready, ramp_duration_s=transit_ramp)
        if lp.improves:
            self._ll().set_lifter(lp.q_lifter)
            # Do NOT re-detect: the box's base_link coordinate is invariant to the
            # lifter move (base_link is fixed below the lifter), and leaning the
            # body often moves the box out of the camera FOV — a re-detect would
            # then return wrong/partial geometry. Reuse the original box; only
            # refresh q_fixed to reflect the new lifter angles.
            q_fixed = _read_fixed()
            if q_fixed is None:
                return {"ok": False, "reason": "no_joint_state"}

        # The self-collision model was warmed in a background thread at connect
        # (CruzrEnv._warm_self_collision_async). Join it before solve_grasp's collision
        # check so we neither rebuild it here — doubling the multi-second build right in
        # the base-arrived→arms-close gap the user sees — nor race-disable the guard.
        warm = getattr(self.env, "_warm_thread", None)
        if warm is not None and warm.is_alive():
            warm.join()
        plan = solve_grasp(box, left, right, q_fixed, inset_mm=float(cfg.grasp_inset_mm),
                           check_collision=True, package_dir=cfg.urdf_package_dir)
        if not plan.ok:
            return {"ok": False, "reason": plan.reason,
                    "ik": {a: plan.ik[a].pos_err_m for a in plan.ik}}

        # TOP-DOWN grasp (avoid sweeping up into the table from below):
        # 1) descend straight down to clamp height (still outside the faces),
        # 2) clamp inward. Descend warm-starts from the ready pose.
        de_ik = {a: solve_arm_ik(chains[a], q_fixed, a, plan.descend[a], q_init=ready.get(a))
                 for a in ("left", "right")}
        self._move_arms_sync({a: r.q for a, r in de_ik.items() if r.converged},
                             ramp_duration_s=transit_ramp)
        # clamp: both arms inward onto the side faces (paddle TCP at face - inset) — contact ramp
        self._move_arms_sync({a: plan.ik[a].q for a in ("left", "right")},
                             ramp_duration_s=contact_ramp)

        # FT contact verification — CRITICAL SAFETY: abort lift if no contact
        thr = float(cfg.contact_force_threshold_n)
        ft = {arm: self._ll().read_hand_ft(arm) for arm in ("left", "right")}
        contacted = all(
            ft[a].get("ok") and ft[a].get("fmag", 0.0) >= thr
            for a in ("left", "right")
        )
        if not contacted:
            return {"ok": False, "reason": "no_contact", "ft": ft}

        # Remember what we just grasped so dual_arm_place() can be called with no
        # arguments (the LLM cannot reliably echo this dict back as a tool arg).
        self._last_grasped_box = det
        # FT-confirmed clamp: tell RecoveryRail a payload is held, so a later motion
        # failure retreats without opening the paddles (i.e. without dropping the box).
        self.env.holding_payload = True
        # Record the waist angle now: if turn_waist rotates the torso before
        # dual_arm_place, the latter rotates its paddle targets by (waist_then - now).
        self._last_grasp_waist_yaw = float(q_fixed["waist_yaw_joint"])
        return {"ok": True, "object": object_name, "box": det, "ft": ft}

    def _move_arms_sync(self, q_by_arm: dict, *, ramp_duration_s: Optional[float] = None) -> None:
        """Command BOTH arms in ONE message so they move simultaneously. ``ramp_duration_s``
        overrides the global ramp for this move only — non-contact transit moves in grasp/place
        (ready / descend-outside-faces / raise-clear) pass the shorter
        ``cfg.arm_transit_ramp_duration_s``; the clamp and place-lower keep the careful default."""
        combined: dict = {}
        for arm_q in q_by_arm.values():
            combined.update(arm_q)
        if combined:
            self._ll().move_joints_blocking(combined, ramp_duration_s=ramp_duration_s)

    def _ready_arm_q(self, chains: dict, q_fixed: dict) -> dict:
        """'Ready' (pre-grasp embrace) joint poses: both paddles in front of the
        chest, facing inward, spread WIDE — the same orientation as the clamp but
        held high and box-independent. Used as a transit waypoint so the
        home<->grasp motion clears the table. Returns {arm: {joint: angle}} for
        the arms whose IK converged.
        """
        from jiuwensymbiosis.adapters.cruzr.geometry import ready_targets, solve_arm_ik
        tgts = ready_targets()
        out = {}
        for arm in ("left", "right"):
            r = solve_arm_ik(chains[arm], q_fixed, arm, tgts[arm])
            if r.converged:
                out[arm] = r.q
        return out

    def _arm_home_path_collides(self, q_fixed: dict, arm_start: dict, arm_goal: dict, n: int = 8) -> bool:
        """Self-collision along a linear arm interpolation start->goal (fixed lifter/waist). False if unavailable."""
        from jiuwensymbiosis.adapters.cruzr.geometry import ARM_JOINTS
        from jiuwensymbiosis.kinematics import self_collision as sc

        cfg = self.env.cfg
        pkg = cfg.urdf_package_dir
        if not sc.available(cfg.urdf_path, pkg):
            logger.warning("[CruzrApi] home: self-collision unavailable; homing UNCHECKED")
            return False
        names = [j for a in ("left", "right") for j in ARM_JOINTS[a]]
        for i in range(n + 1):
            t = i / n
            jv = dict(q_fixed)
            for j in names:
                jv[j] = (1.0 - t) * float(arm_start.get(j, 0.0)) + t * float(arm_goal.get(j, 0.0))
            qf = sc.full_q(cfg.urdf_path, pkg, jv)
            if qf is not None and sc.in_self_collision(cfg.urdf_path, pkg, qf):
                return True
        return False

    def _safe_home_arms(self) -> dict:
        """Home both arms to zero along a self-collision-checked path, escalating through
        recovery strategies so the robot reliably reaches home instead of getting stuck:

          0. PRIMARY — abduct the arms OUT to the sides (cfg.home_clearance_arm_q), then
             descend to 0, so the forearms come down along the body's sides and never drag
             across the torso (the self-collision model does not always catch that contact);
          1. both arms straight to zero (deep fallback — only if clearance is off/blocked);
          2. ONE arm at a time — breaks the common failure where both forearms cross in
             front of the chest when swept together; tried right-first then left-first;
          3. raise both to the box-independent ready pose, then descend (together, else
             one arm at a time from ready).

        Every leg is self-collision-checked before it is commanded and the first fully
        clear plan wins. Only if EVERY strategy still self-collides do we refuse (no
        motion) — a genuinely stuck pose that needs manual recovery, not a blind sweep
        through the body.
        """
        from jiuwensymbiosis.adapters.cruzr.geometry import ARM_JOINTS
        cfg = self.env.cfg
        qq = self._ll().get_joint_positions() or {}
        cur = {j: float(qq.get(j, 0.0)) for a in ("left", "right") for j in ARM_JOINTS[a]}
        qf = {n: float(qq.get(n, 0.0)) for n in ("lifter_pitch_1_joint", "lifter_pitch_2_joint",
                                                 "lifter_pitch_3_joint", cfg.waist_yaw_joint)}
        zeros = {j: 0.0 for j in cur}

        def _one_arm_zeroed(base: dict, arm: str) -> dict:
            """``base`` with ``arm``'s joints set to 0 (the other arm held where base has it)."""
            m = dict(base)
            m.update({j: 0.0 for j in ARM_JOINTS[arm]})
            return m

        def _try(name: str, waypoints: list[dict]) -> dict | None:
            """Command ``waypoints`` (cur -> ... -> zeros) iff every leg is collision-free."""
            legs = list(zip(waypoints, waypoints[1:]))
            if any(self._arm_home_path_collides(qf, s, g) for s, g in legs):
                return None
            logger.info("[CruzrApi] home: arm home via %s", name)
            for wp in waypoints[1:]:
                self._move_arms_sync({a: {j: wp[j] for j in ARM_JOINTS[a]} for a in ("left", "right")})
            return {"ok": True}

        # 0) PRIMARY: abduct the arms OUT to the sides (cfg.home_clearance_arm_q), THEN descend
        #    to 0 — so the forearms come down along the body's sides instead of dragging across
        #    the torso, which the self-collision model does not always catch. Empty config
        #    disables it and we fall straight through to the direct/one-arm/ready escalation.
        clr = {j: float(v) for j, v in (getattr(cfg, "home_clearance_arm_q", None) or {}).items()
               if j in zeros}
        if clr:
            cl = {**zeros, **clr}
            for name, wps in (("clearance", [cur, cl, zeros]),
                              ("clearance+right-first", [cur, cl, _one_arm_zeroed(cl, "right"), zeros]),
                              ("clearance+left-first", [cur, cl, _one_arm_zeroed(cl, "left"), zeros])):
                out = _try(name, wps)
                if out is not None:
                    return out

        # 1) both direct  2) one arm at a time (both orders) — fallback if clearance is off or
        #    its path self-collides (direct is now only a deep fallback, not the normal home).
        for name, wps in (("direct", [cur, zeros]),
                          ("right-arm-first", [cur, _one_arm_zeroed(cur, "right"), zeros]),
                          ("left-arm-first", [cur, _one_arm_zeroed(cur, "left"), zeros])):
            out = _try(name, wps)
            if out is not None:
                return out

        # 3) raise both to the box-independent ready pose, then descend (together / one-at-a-time)
        from jiuwensymbiosis.kinematics.urdf_chain import parse_chain
        chains = {"left": parse_chain(cfg.urdf_path, "base_link", cfg.left_arm_leaf),
                  "right": parse_chain(cfg.urdf_path, "base_link", cfg.right_arm_leaf)}
        ready = self._ready_arm_q(chains, qf)
        if all(a in ready for a in ("left", "right")):
            rf = {j: float(ready[a][j]) for a in ("left", "right") for j in ARM_JOINTS[a]}
            for name, wps in (("ready", [cur, rf, zeros]),
                              ("ready+right-first", [cur, rf, _one_arm_zeroed(rf, "right"), zeros]),
                              ("ready+left-first", [cur, rf, _one_arm_zeroed(rf, "left"), zeros])):
                out = _try(name, wps)
                if out is not None:
                    return out

        logger.error("[CruzrApi] home: NO self-collision-free home path "
                     "(clearance / direct / one-arm-at-a-time / ready all blocked); refusing — needs manual recovery")
        return {"ok": False, "reason": "home_path_self_collision"}

    @robot_tool(
        desc="Home both arms along a self-collision-checked path, escalating through fallbacks (abduct clear of the torso first, then descend). For wrapping up a task prefer home.",
        tags=["motion"],
        provides=["body.home"],
    )
    def home_arms(self) -> dict:
        """Home both arms to zero via the self-collision-checked escalation (see _safe_home_arms)."""
        return self._safe_home_arms()

    @implements(LIFT_TO_CLEARANCE)
    def lift_to_clearance(self, box: Optional[dict] = None, upright_tol_rad: float = 0.05) -> dict:
        """Stand the torso up (lifter -> 0) and raise the held box to the PRESET absolute
        height ``transit_lift_z_m`` (base-frame z) in ONE coordinated move (lifter + arms
        together); the paddle gap is preserved at the endpoints."""
        from dataclasses import replace

        import numpy as np

        from jiuwensymbiosis.adapters.cruzr.geometry import (
            ARM_JOINTS,
            LIFTER_JOINTS,
            TOOL_APPROACH_LOCAL,
            TOOL_PADDLE_LOCAL,
            plan_clamp_targets,
            solve_arm_ik,
        )
        from jiuwensymbiosis.kinematics.fk import fk_chain
        from jiuwensymbiosis.perception.object_geometry import ObjectGeometry3D

        box = box if isinstance(box, dict) else self._last_grasped_box
        if not box:
            return {"ok": False, "reason": "no_box_to_lift"}
        cfg = self.env.cfg
        b = ObjectGeometry3D(
            True, "", tuple(box["center_mm"]), box["width_mm"], box["height_mm"],
            box["front_x_mm"], box["top_z_mm"], box["n_points"], back_x_mm=box.get("back_x_mm", 0.0),
        )
        chains = {
            "left": parse_chain(cfg.urdf_path, "base_link", cfg.left_arm_leaf),
            "right": parse_chain(cfg.urdf_path, "base_link", cfg.right_arm_leaf),
        }
        q = self._ll().get_joint_positions()
        fixed_names = ("lifter_pitch_1_joint", "lifter_pitch_2_joint",
                       "lifter_pitch_3_joint", "waist_yaw_joint")
        if not q or any(n not in q for n in fixed_names):
            return {"ok": False, "reason": "no_joint_state"}
        q_fixed = {k: q[k] for k in fixed_names}
        cur = {a: {j: q.get(j, 0.0) for j in ARM_JOINTS[a]} for a in ("left", "right")}

        # Raise the box to a PRESET ABSOLUTE height (not a relative +dz, whose final
        # height varied with the grasp pose and could leave the camera occluded). Plan
        # everything for the UPRIGHT torso so we never move on an unreachable target:
        # standing up holds the arm joints (rigid), so FK with the current arm joints
        # but the lifter at 0 gives where each paddle will be right AFTER standing up.
        # We keep that x/y + orientation and set ONLY z to the preset -> both paddles go
        # to the same absolute z, so the gap (and box level) is preserved exactly.
        target_z = float(cfg.transit_lift_z_m)
        lifter_from = {j: float(q_fixed[j]) for j in LIFTER_JOINTS}
        stand_needed = any(abs(v) > upright_tol_rad for v in lifter_from.values())
        zero_lifter = {j: 0.0 for j in LIFTER_JOINTS}
        q_upright = {**q, **zero_lifter}                 # FK: arms held, lifter at 0
        q_fixed_upright = {**q_fixed, **zero_lifter}     # IK: solve with the torso upright

        _, _, clamp = plan_clamp_targets(b)              # per-arm tcp_offset template only
        lifted = {}
        for a in ("left", "right"):
            tf = fk_chain(chains[a], q_upright)
            r = tf[:3, :3]
            tcp = tf[:3, 3] + r @ np.asarray(clamp[a].tcp_offset_local, dtype=float)
            lifted[a] = replace(
                clamp[a],
                pos_m=(float(tcp[0]), float(tcp[1]), target_z),
                approach=tuple(float(v) for v in r @ np.asarray(TOOL_APPROACH_LOCAL, dtype=float)),
                paddle=tuple(float(v) for v in r @ np.asarray(TOOL_PADDLE_LOCAL, dtype=float)),
            )
        ik = {a: solve_arm_ik(chains[a], q_fixed_upright, a, lifted[a], q_init=cur[a],
                             check_collision=True, package_dir=cfg.urdf_package_dir)
              for a in ("left", "right")}
        if not all(ik[a].converged for a in ("left", "right")):
            return {"ok": False, "reason": "lift_unreachable",
                    "pos_err_m": {a: ik[a].pos_err_m for a in ("left", "right")}}

        # ONE coordinated motion: command the arms to the raised (upright-torso) IK pose
        # and — if the torso is leaned — the lifter to 0 IN THE SAME move, so they ramp
        # together. Endpoints are exact (box centred at the preset height, torso upright,
        # paddle gap preserved); the gap / box-level are NOT held constant mid-ramp because
        # the arms and lifter change at once — accepted for a single, faster lift.
        cmd = {**ik["left"].q, **ik["right"].q}
        if stand_needed:
            cmd.update(zero_lifter)
        logger.info("[CruzrApi] lift_to_clearance -> z=%.3f%s (one move)",
                    target_z, f", lifter 0 from {lifter_from}" if stand_needed else "")
        self._ll().move_joints_blocking(cmd, ramp_duration_s=getattr(cfg, "lifter_ramp_duration_s", None))
        return {"ok": True, "target_z_m": target_z, "stood_up": stand_needed, "lifter_from": lifter_from}

    # Not an action of its own — ``home`` IS the safe retreat (see api/actions.py:HOME).
    # Kept as a named method because the grasp/place paths and the bring-up script call
    # it directly, and because ``tol_rad`` is a body knob no contract should carry.
    def _home_safely(self, tol_rad: float = 0.05) -> dict:
        """Retreat to a safe home, doing the minimum motion for the current state.

        - already upright AND arms home AND waist neutral → do nothing (``skipped``);
        - upright but arms not home / waist rotated → neutralize the waist then home the arms;
        - leaned forward (or unknown) → straighten the lifter, neutralize the waist, then home.
        """
        from jiuwensymbiosis.adapters.cruzr.geometry import ARM_JOINTS, LIFTER_JOINTS

        cfg = self.env.cfg
        q = self._ll().get_joint_positions() or {}
        upright = all(abs(q.get(j, 0.0)) <= tol_rad for j in LIFTER_JOINTS)
        arms_home = all(
            abs(q.get(j, 0.0)) <= tol_rad
            for a in ("left", "right") for j in ARM_JOINTS[a]
        )
        waist_home = abs(q.get(cfg.waist_yaw_joint, 0.0)) <= tol_rad
        # With no joint state we cannot tell where we are; fall through to the
        # full safe sequence (treat as possibly leaned).
        have_state = bool(q)

        if have_state and upright and arms_home and waist_home:
            return {"ok": True, "skipped": "already_home"}
        if have_state and upright:
            self._neutralize_waist(tol_rad)   # body straight: no table to clear
            return self._safe_home_arms()

        # Leaned forward (or unknown): straighten the lifter, neutralize the waist, then home.
        self._ll().set_lifter({j: 0.0 for j in LIFTER_JOINTS})
        self._neutralize_waist(tol_rad)
        return self._safe_home_arms()

    def _neutralize_waist(self, tol_rad: float = 0.05) -> None:
        """Turn waist_yaw back to 0 if rotated, holding the arms where they now are."""
        from jiuwensymbiosis.adapters.cruzr.geometry import ARM_JOINTS

        cfg = self.env.cfg
        qq = self._ll().get_joint_positions() or {}
        if abs(qq.get(cfg.waist_yaw_joint, 0.0)) <= tol_rad:
            return
        hold = {j: float(qq.get(j, 0.0)) for a in ("left", "right")
                for j in ARM_JOINTS[a] if j in qq}
        self._ll().turn_waist_blocking(0.0, hold=hold, waist_joint=cfg.waist_yaw_joint)

    def recovery_home(self, tol_rad: float = 0.05) -> dict:
        """Payload-aware retreat; ``RecoveryRail`` prefers this over ``home``.

        ``home`` homes the arms, which un-clamps the paddles — with a box held
        that is a drop, not a recovery. While a payload is held, retreat only the parts
        that cannot drop it: straighten the lifter and neutralize the waist, leaving the
        arms clamped for a human (or a retried ``dual_arm_place``) to resolve.
        """
        if not getattr(self.env, "holding_payload", False):
            return self._home_safely(tol_rad)

        from jiuwensymbiosis.adapters.cruzr.geometry import LIFTER_JOINTS

        logger.warning("[CruzrApi] recovery_home: payload held → keeping arms clamped")
        self._ll().set_lifter({j: 0.0 for j in LIFTER_JOINTS})
        self._neutralize_waist(tol_rad)
        return {"ok": True, "payload_preserved": True}

    @implements(DUAL_ARM_PLACE)
    def dual_arm_place(self, target: Optional[dict] = None, inset_mm: float = 6.5,
                  descend_dz_m: float = 0.02, surface_z_mm: Optional[float] = None,
                  surface: Optional[dict] = None) -> dict:
        """Reverse of grasp: move the box over a validated landing point ON the table,
        lower, drop the torso ~``descend_dz_m`` to clear the box rim, release, then raise.

        Pass ``surface`` = the full ``locate_for_place()`` result (table XY footprint +
        top z); when omitted it falls back to the last ``locate_for_place`` (cached), so an
        LLM calls ``locate_for_place()`` then ``dual_arm_place()`` with no args. The box is landed
        FULLY on the table clear of the edges — Y centred, X just inside the near edge — so
        it never sits on the rim or overhangs. Returns a
        ``box_wider_than_table`` / ``box_deeper_than_table`` error (no motion) if the box
        can't fit, or an unreachable error if the landing point is out of arm range (drive
        closer and retry). ``surface_z_mm`` alone is the legacy z-only path (places at the
        current carried XY — un-validated). Does NOT return to zero — the lifter-straighten
        and arm-home are delegated to ``home()`` (call it after placing). ``box`` is
        the geometry dict from ``locate_for_grasp``/``dual_arm_grasp``; when omitted it falls back
        to the most recent successful ``dual_arm_grasp``.
        """
        from dataclasses import replace

        import numpy as np

        from jiuwensymbiosis.adapters.cruzr.geometry import (
            ARM_JOINTS,
            LIFTER_JOINTS,
            plan_clamp_targets,
            search_lifter_for_place,
            solve_arm_ik,
        )
        from jiuwensymbiosis.kinematics.fk import fk_chain
        from jiuwensymbiosis.kinematics.urdf_chain import parse_chain
        from jiuwensymbiosis.perception.object_geometry import ObjectGeometry3D

        box = target if isinstance(target, dict) else self._last_grasped_box
        if not box:
            return {"ok": False, "reason": "no_box_to_place"}
        # Fall back to the last sensed table so an LLM can call locate_for_place() then
        # dual_arm_place() with no args (the footprint dict is too large to echo back).
        if surface is None:
            surface = self._last_surface

        cfg = self.env.cfg
        b = ObjectGeometry3D(
            True, "", tuple(box["center_mm"]),
            box["width_mm"], box["height_mm"], box["front_x_mm"], box["top_z_mm"],
            box["n_points"], back_x_mm=box.get("back_x_mm", 0.0),
        )
        chains = {
            "left": parse_chain(cfg.urdf_path, "base_link", cfg.left_arm_leaf),
            "right": parse_chain(cfg.urdf_path, "base_link", cfg.right_arm_leaf),
        }
        q = self._ll().get_joint_positions()
        fixed_names = ("lifter_pitch_1_joint", "lifter_pitch_2_joint",
                       "lifter_pitch_3_joint", "waist_yaw_joint")
        if not q or any(name not in q for name in fixed_names):
            # no state to warm-start safely; at least open+home blind is unsafe, so bail
            return {"ok": False, "reason": "no_joint_state"}
        q_fixed = {k: q[k] for k in fixed_names}
        cur = {a: {j: q.get(j, 0.0) for j in ARM_JOINTS[a]} for a in ("left", "right")}

        # Where the box is being carried right now, via FK of the measured joints (the
        # mean of the two paddle TCPs is the box centre in x,y). Used for the legacy
        # z-only path and to log how far the box moves to reach the table.
        clamp_tmpl = plan_clamp_targets(b, inset_mm=inset_mm)[2]
        held_tcp: dict[str, tuple[float, float, float]] = {}
        for a in ("left", "right"):
            tf = fk_chain(chains[a], q)
            p = tf[:3, 3] + tf[:3, :3] @ np.asarray(clamp_tmpl[a].tcp_offset_local, dtype=float)
            held_tcp[a] = (float(p[0]), float(p[1]), float(p[2]))
        cx = [held_tcp["left"][0], held_tcp["right"][0]]
        cy = [held_tcp["left"][1], held_tcp["right"][1]]
        cx_now_mm = 500.0 * (cx[0] + cx[1])   # (x0 + x1)/2 * 1000
        cy_now_mm = 500.0 * (cy[0] + cy[1])   # (y0 + y1)/2 * 1000

        # Pick the landing XY. With a sensed table footprint, land the box FULLY on the
        # table clear of the edges (Y centred on the table; X just inside the near edge so
        # the whole box sits on it and the arms reach the least far) — NOT at the carried
        # XY, which is never validated against the table and drops the box on the rim or in
        # mid-air. Without `surface`, keep the legacy z-only behaviour (place at the carried
        # XY). Fit checks bail (no motion) if the box cannot sit inside the edges + margin.
        if surface is not None:
            near_x, far_x = float(surface["front_x_mm"]), float(surface["back_x_mm"])
            table_cy, table_w = float(surface["center_mm"][1]), float(surface["width_mm"])
            surf_z: Optional[float] = float(surface["surface_z_mm"])
            # box depth along x; fall back to width if the far face was not detected
            box_depth = (b.back_x_mm - b.front_x_mm) if b.back_x_mm > b.front_x_mm else b.width_mm
            m = float(cfg.place_edge_margin_mm)
            if b.width_mm / 2.0 + m > table_w / 2.0:
                return {"ok": False, "reason": "box_wider_than_table"}
            if box_depth + 2.0 * m > (far_x - near_x):
                return {"ok": False, "reason": "box_deeper_than_table"}
            landing_x = near_x + box_depth / 2.0 + m
            landing_y = table_cy
            logger.info("[CruzrApi] dual_arm_place: table x=[%.1f,%.1f] cy=%.1f w=%.1f -> landing (%.1f, %.1f) "
                        "(carried %.1f, %.1f)", near_x, far_x, table_cy, table_w,
                        landing_x, landing_y, cx_now_mm, cy_now_mm)
        else:
            landing_x, landing_y, surf_z = cx_now_mm, cy_now_mm, surface_z_mm

        # Rebuild LEVEL place targets at the landing x,y (front == back == depth mid so the
        # clamp sits at landing_x). No waist-delta rotation is needed — FK already reflects
        # the current waist. _apply_surface_z then lands the box bottom on the surface
        # (no-op when surf_z is None).
        b_now = replace(b, center_mm=(landing_x, landing_y, b.center_mm[2]),
                        front_x_mm=landing_x, back_x_mm=landing_x)
        approach, descend, clamp = plan_clamp_targets(b_now, inset_mm=inset_mm)
        # Keep the two-hand spacing EXACTLY as currently held. plan_clamp_targets set the CLAMP paddle y
        # from box.width - 2·inset, but the box is really held at whatever gap the GRASP achieved
        # (grasp_inset_mm vs place inset differ by ~46mm/side, plus arm compliance) — moving the paddles
        # to that re-derived spacing squeezes the box (crush) or spreads it (early drop) during the
        # lower. So override the CLAMP targets' x,y with the LIVE FK paddle positions (cx/cy above)
        # rigidly translated to the landing point: the paddle gap (cy[left]-cy[right]) is preserved
        # bit-for-bit. Only approach/descend keep the box-derived WIDE spread (they are the open/release
        # waypoints, meant to spread past the box). z stays the plan's grasp height (surface-shifted
        # below); a lifter pitch is a rotation in x-z, so it never disturbs the y spacing.
        dx_m = landing_x / 1000.0 - 0.5 * (cx[0] + cx[1])
        dy_m = landing_y / 1000.0 - 0.5 * (cy[0] + cy[1])
        # Reproduce dual_arm_grasp's inward SQUEEZE. Commanding the paddles to EXACTLY the held gap
        # is ~zero steady-state position error, so the stiff position controller applies almost no
        # inward force; at the table (arms extended/leaned → weak inward stiffness) the box then slips
        # out during the lower. Grasp instead overshoots the faces (grasp_inset_mm) to press in. So
        # pull each paddle place_squeeze_mm/2 PAST the held position toward the box centre: the arms
        # actively squeeze during the lower. Centre + x are untouched, so the box still lands on target.
        squeeze_half_m = 0.5 * float(getattr(cfg, "place_squeeze_mm", 0.0)) / 1000.0
        inward = {"left": -1.0, "right": 1.0}   # left paddle is +Y → inward is -Y; right mirrors
        clamp = {a: replace(clamp[a], pos_m=(cx[i] + dx_m,
                                             cy[i] + dy_m + inward[a] * squeeze_half_m,
                                             clamp[a].pos_m[2]))
                 for i, a in enumerate(("left", "right"))}
        approach, descend, clamp = self._apply_surface_z(b, (approach, descend, clamp), surf_z)

        # Nudge the lifter as LITTLE as possible so the ARMS can reach the place targets:
        # search the SMALLEST level-manifold lean (capped at place_max_lift_lean_rad) from
        # which both arms reach the (base-frame, fixed) place clamp targets, then lean there
        # holding the arm joints. A big forward lean would drop the torso onto the table, so
        # the arms — not the lifter — position the box; if the arms can't reach within the
        # cap, place returns unreachable (drive closer) rather than leaning into the table.
        cur_lifter = {j: q_fixed[j] for j in LIFTER_JOINTS}
        lp = search_lifter_for_place(clamp, chains["left"], chains["right"],
                                     cur_lifter, q_fixed["waist_yaw_joint"],
                                     max_lean_rad=float(cfg.place_max_lift_lean_rad))
        if not lp.found:
            return {"ok": False, "reason": lp.reason or "place_unreachable_any_lifter"}
        if lp.improves:
            held = {j: float(q.get(j, 0.0)) for a in ("left", "right") for j in ARM_JOINTS[a]}
            logger.info("[CruzrApi] dual_arm_place: leaning lifter to %s to reach place targets", lp.q_lifter)
            self._ll().move_joints_blocking(
                {**held, **lp.q_lifter}, ramp_duration_s=getattr(cfg, "lifter_ramp_duration_s", None))
            q = self._ll().get_joint_positions()
            if not q or any(name not in q for name in fixed_names):
                return {"ok": False, "reason": "no_joint_state"}
            q_fixed = {k: q[k] for k in fixed_names}
            cur = {a: {j: q.get(j, 0.0) for j in ARM_JOINTS[a]} for a in ("left", "right")}
        # Ramp split (same as grasp): the box-CONTACT lower (below) runs on the dedicated PLACE
        # contact ramp — separate from the grasp clamp's arm_contact_ramp_duration_s so the box can
        # be set down gently without slowing the clamp; falls back to the shared contact ramp if the
        # place-specific key is absent. The later non-contact open/raise moves run on the transit ramp.
        place_contact_ramp = getattr(cfg, "place_contact_ramp_duration_s", None)
        if place_contact_ramp is None:
            place_contact_ramp = getattr(cfg, "arm_contact_ramp_duration_s", None)
        place_transit_ramp = getattr(cfg, "arm_transit_ramp_duration_s", None)
        # 1. lower: descend to the clamp (place) height. A single joint-space ramp only
        #    preserves the paddle GAP at its two endpoints — mid-ramp the gap bows OPEN and
        #    the box slips out during this long, arms-extended descent (weak inward stiffness).
        #    Instead interpolate each paddle TCP linearly IN CARTESIAN from the live held pose
        #    down to the clamp target, solve IK per waypoint, and STREAM the waypoints as ONE
        #    continuous trajectory (same wall-clock as one move, no per-knot stutter). Cartesian
        #    interpolation makes the commanded gap only ever TIGHTEN toward the squeeze
        #    (gap(f) = gap0 - f·place_squeeze), so it never widens mid-descent. If any waypoint
        #    IK fails, fall back to the single-move (endpoints-only) path.
        down = {a: solve_arm_ik(chains[a], q_fixed, a, clamp[a], q_init=cur[a],
                                check_collision=True, package_dir=cfg.urdf_package_dir)
                for a in ("left", "right")}
        n_wp = max(1, int(getattr(cfg, "place_lower_waypoints", 4)))
        stream_ok = n_wp > 1 and all(down[a].converged for a in ("left", "right"))
        # Interpolate the descent from the ACTUAL current (post-lean) paddle TCP — NOT the pre-lean
        # carried held_tcp. The lifter lean above (if any) moved the shoulders while holding the arms,
        # so the box is no longer at held_tcp; starting the Cartesian interp from the stale pre-lean TCP
        # made the first stream segment swing the box UP/BACK toward the carry height — into the chest.
        # FK the post-lean pose so the stream is a clean straight descent from where the box IS now.
        start_tcp: dict[str, tuple[float, float, float]] = {}
        for a in ("left", "right"):
            tf = fk_chain(chains[a], q)
            p = tf[:3, 3] + tf[:3, :3] @ np.asarray(clamp[a].tcp_offset_local, dtype=float)
            start_tcp[a] = (float(p[0]), float(p[1]), float(p[2]))
        knots: list[dict] = [{**cur["left"], **cur["right"]}]   # start = measured, no jump
        prev = {a: cur[a] for a in ("left", "right")}
        for k in range(1, n_wp):                                # interior knots 1..n_wp-1
            if not stream_ok:
                break
            knot: dict = {}
            f = k / n_wp
            for a in ("left", "right"):
                sp, ep = start_tcp[a], clamp[a].pos_m
                pos = tuple(sp[j] + f * (ep[j] - sp[j]) for j in range(3))
                r = solve_arm_ik(chains[a], q_fixed, a, replace(clamp[a], pos_m=pos),
                                 q_init=prev[a], check_collision=True, package_dir=cfg.urdf_package_dir)
                if not r.converged:
                    stream_ok = False
                    break
                knot.update(r.q)
                prev[a] = r.q
            if stream_ok:
                knots.append(knot)
        if stream_ok:
            knots.append({**down["left"].q, **down["right"].q})   # final = clamp target
            self._ll().stream_joint_trajectory(knots, total_duration_s=place_contact_ramp)
        else:
            self._move_arms_sync({a: r.q for a, r in down.items() if r.converged},
                                 ramp_duration_s=place_contact_ramp)
        # 1b. drop the torso ~descend_dz_m vertically via the lifter so the held
        #     paddles slide STRAIGHT DOWN off the box rim before opening (else
        #     they hook the edge on the way out). Arms are held at the 'down'
        #     pose, so the paddles fall with the torso; the open below then
        #     happens at the lowered height.
        from jiuwensymbiosis.adapters.cruzr.geometry import lower_torso_lifter
        lifter_now = {j: q_fixed[j] for j in LIFTER_JOINTS}
        low = lower_torso_lifter(chains["left"], "left", lifter_now, q_fixed["waist_yaw_joint"], descend_dz_m)
        if low is not None:
            self._ll().set_lifter(low)
            q2 = self._ll().get_joint_positions()
            if q2 and all(n in q2 for n in fixed_names):
                q_fixed = {k: q2[k] for k in fixed_names}
        # 2. release: open the paddles outward at the LOWERED height (descend
        #    target dropped by descend_dz_m so opening doesn't climb back up into
        #    the rim we just cleared).
        open_tgt = {a: replace(descend[a], pos_m=(descend[a].pos_m[0], descend[a].pos_m[1],
                                                  descend[a].pos_m[2] - descend_dz_m))
                    for a in ("left", "right")}
        rel = {a: solve_arm_ik(chains[a], q_fixed, a, open_tgt[a],
                               q_init=down[a].q if down[a].converged else cur[a],
                               check_collision=True, package_dir=cfg.urdf_package_dir)
               for a in ("left", "right")}
        # open (non-squeeze, box already resting on the surface) → fast transit ramp
        self._move_arms_sync({a: r.q for a, r in rel.items() if r.converged},
                             ramp_duration_s=place_transit_ramp)
        # Cleared here, not at the return: the paddles are already open, so a failure
        # during the raise below must not make RecoveryRail preserve a phantom payload.
        self.env.holding_payload = False
        # 3. raise clear: up to the approach pose (above the box) before homing,
        #    so the hands lift off top-down instead of dragging across the table
        up = {a: solve_arm_ik(chains[a], q_fixed, a, approach[a],
                              q_init=rel[a].q if rel[a].converged else cur[a],
                              check_collision=True, package_dir=cfg.urdf_package_dir)
              for a in ("left", "right")}
        # raise clear (empty arms, box already placed) → fast transit ramp
        self._move_arms_sync({a: r.q for a, r in up.items() if r.converged},
                             ramp_duration_s=place_transit_ramp)
        # 4. back to the raised "ready" posture (arms up in front, clear). The
        #    return-to-zero (straighten lifter -> 0, then arms -> 0) is delegated
        #    to home() — call it after placing to retreat to a safe home.
        return {"ok": True, "leaned": bool(lp.improves), "lifter": lp.q_lifter,
                "surface_z_mm": surf_z, "landing_mm": [landing_x, landing_y],
                "carried_mm": [cx_now_mm, cy_now_mm]}

    def _rotate_targets_for_waist_delta(self, chain, q_fixed: dict, target_dicts: tuple) -> tuple:
        """Rotate paddle-target dicts by the torso transform from the recorded grasp
        waist to the current waist (the rigid motion the box underwent during
        turn_waist). Exact via FK of the sub-chain up to ``waist_yaw_joint`` (so a
        leaned torso is handled, not just base-Z). Returns ``target_dicts`` unchanged
        when no grasp waist was recorded (plain place, no prior turn) or the delta is
        negligible, so grasp->place with no turn is byte-for-byte the old behavior.
        """
        grasp_waist = self._last_grasp_waist_yaw
        if grasp_waist is None:  # plain place, no prior turn: nothing to rotate
            return target_dicts

        import numpy as np

        from jiuwensymbiosis.kinematics.fk import fk_chain
        from jiuwensymbiosis.kinematics.urdf_chain import Chain

        waist = self.env.cfg.waist_yaw_joint
        waist_now = float(q_fixed[waist])
        if abs(waist_now - float(grasp_waist)) <= 1e-6:
            return target_dicts

        js = chain.joints
        idx = next((i for i, j in enumerate(js) if j.name == waist), None)
        if idx is None:  # waist not on this chain: cannot transform, leave targets as-is
            return target_dicts
        sub = Chain(js[: idx + 1])
        m = fk_chain(sub, {**q_fixed, waist: waist_now}) @ np.linalg.inv(
            fk_chain(sub, {**q_fixed, waist: float(grasp_waist)}))
        r, t = m[:3, :3], m[:3, 3]
        logger.info("[CruzrApi] dual_arm_place: rotating targets by waist delta %.3f rad",
                    waist_now - float(grasp_waist))
        return tuple(
            {a: _rotate_arm_target(tg[a], r, t) for a in ("left", "right")}
            for tg in target_dicts
        )

    def _apply_surface_z(self, b, target_dicts: tuple, surface_z_mm) -> tuple:
        """Shift paddle-target dicts vertically so the box BOTTOM lands on ``surface_z_mm``
        (base frame, mm). ``dz = (surface_z_mm - (b.top_z_mm - b.height_mm)) / 1000``. Returns
        ``target_dicts`` unchanged when ``surface_z_mm`` is None (place at the grasp height).
        """
        if surface_z_mm is None:
            return target_dicts
        grasp_bottom_z = b.top_z_mm - b.height_mm            # box bottom base z at grasp (mm)
        dz = (float(surface_z_mm) - grasp_bottom_z) / 1000.0  # metres
        logger.info("[CruzrApi] dual_arm_place: surface z-shift dz=%.3f m (surface=%.1f, box_bottom=%.1f)",
                    dz, float(surface_z_mm), grasp_bottom_z)
        return tuple(
            {a: _shift_target_z(tg[a], dz) for a in ("left", "right")}
            for tg in target_dicts
        )

    @implements(GET_IMAGE)
    def get_image(self, camera_name: str = "waist_rgbd"):
        """Grab the latest RGB frame from the waist camera."""
        frames = self._ll().grab_frames(camera="waist")
        return None if frames is None else frames[0]

    @implements(PIXEL_TO_BASE_XYZ)
    def pixel_to_base_xyz(self, u: float, v: float, depth_m: float, camera_name: str = "waist_rgbd") -> dict:
        """Project a pixel + depth to base-frame XYZ in mm."""
        frames = self._ll().grab_frames(camera="waist")
        if frames is not None:
            _, _, k_live, tf_live = frames
        else:
            k_live, tf_live = None, None
        intrinsics = k_live if k_live is not None else self._calib_intrinsics()
        tf_base_cam = tf_live if tf_live is not None else self._calib_extrinsics()
        if intrinsics is None:
            return {"ok": False, "reason": "no_intrinsics"}
        if tf_base_cam is None:
            return {"ok": False, "reason": "no_extrinsics"}
        xyz = project_to_base((float(u), float(v)), float(depth_m), intrinsics, tf_base_cam)
        return {"ok": True, "x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])}

    # ---------------------------------------------------------------- helpers
    def _ensure_detector(self) -> None:
        """Lazy-bind the detector segmentation function."""
        if self._seg_fn is not None:
            return
        try:
            self._seg_fn = init_detector(self._detector_service_url)
            logger.info("[CruzrApi] detector client bound to %s", self._detector_service_url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CruzrApi] detector init failed (%s); detect will return ok=False.", exc)
            self._seg_fn = None

    def _calib(self) -> dict:
        if not self._calib_loaded:
            self._calib_loaded = True
            if self._camera_calib_path:
                try:
                    self._calib_cache = load_cruzr_camera_calib(self._camera_calib_path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[CruzrApi] calib load failed (%s)", exc)
                    self._calib_cache = None
        return self._calib_cache or {}

    def _calib_intrinsics(self):
        return self._calib().get("intrinsics")

    def _calib_extrinsics(self):
        return self._calib().get("tf_base_cam")

    def _ll(self) -> Any:
        return self.env.low_level


def _rotate_arm_target(tgt, r, t):
    """Rigid-transform an ArmTarget's base-frame pose by rotation ``r`` (3x3) and
    translation ``t`` (3,): the ``pos_m`` point moves by ``r @ pos + t`` and the
    ``approach``/``paddle`` direction vectors rotate by ``r``. ``tcp_offset_local``
    is expressed in the tool frame, so it is invariant and left unchanged.
    """
    from dataclasses import replace

    import numpy as np

    pos = r @ np.asarray(tgt.pos_m, dtype=float) + t
    apr = r @ np.asarray(tgt.approach, dtype=float)
    pad = r @ np.asarray(tgt.paddle, dtype=float)
    return replace(
        tgt,
        pos_m=(float(pos[0]), float(pos[1]), float(pos[2])),
        approach=(float(apr[0]), float(apr[1]), float(apr[2])),
        paddle=(float(pad[0]), float(pad[1]), float(pad[2])),
    )


def _shift_target_z(tgt, dz):
    """Translate an ArmTarget's base-frame ``pos_m`` up by ``dz`` metres (z only);
    the approach/paddle direction vectors and tool-frame offset are unchanged."""
    from dataclasses import replace

    x, y, z = tgt.pos_m
    return replace(tgt, pos_m=(x, y, z + float(dz)))


class _NullPose:
    """detect_and_centroid 仅读取 .x/.y/.z/.r 打日志；Cruzr 无 TCP，全置 0。"""

    x = y = z = r = 0.0


# ---------------------------------------------------------------------------
# Search→approach orchestration (head sweep → base wheel-velocity centering →
# waist handoff). Pure orchestration over this api's own tools (search_target /
# navigate_relative / set_head / locate_for_grasp) — no direct ROS, fully mockable.
# Head is mounted high: when the base nears a low box the target drops out of the
# head FOV → (A) pitch-track it down; (B) probe the (near-range) waist camera before
# declaring it lost. run_approach is the search-grasp demo entry (not a @robot_tool).
# ---------------------------------------------------------------------------

def _reset_head(api: Any, cfg: Any) -> None:
    """Return the head to the configured forward pose (continuous set_head)."""
    api.set_head(float(cfg.head_forward_yaw_rad), float(cfg.head_forward_pitch_rad))


def _acquire_with_head(api: Any, object_name: str, cfg: Any, *, on: Optional[str] = None) -> dict:
    """Pan ONLY the head (base stays put) to find the target.

    Returns ``{"found": bool, "total_bearing": float}``. When found at head yaw
    theta with in-image bearing beta, total_bearing = theta + beta (the image
    bearing is relative to the head, so it must be added to the head yaw). ``on``
    (when set) threads through to the grounded ``search_target`` so a head hit is
    2-D-verified (right-eye bbox overlap) on the reference surface before it counts
    (degrades fail-open). Always the ``on`` relation — bbox containment is the only one a
    single wide-angle image can decide; the caller passes None for any other relation.
    """
    for theta in list(cfg.head_search_yaw_positions_rad):
        api.set_head(float(theta), float(cfg.head_search_pitch_rad))
        s = api._nav._look_for(object_name, on, camera="head")
        if s.get("found"):
            return {"found": True, "total_bearing": float(theta) + float(s["bearing_rad"])}
    return {"found": False, "total_bearing": 0.0}


def _track_head_pitch(api: Any, cfg: Any, pitch_state: list, v_center: float, image_h: int) -> None:
    """Nudge head_pitch so the target stays vertically centered as the base nears.

    Box low in the frame (v_err>0) → look further down. Sign lives in
    ``head_pitch_track_gain`` (negative: +up/-down convention, verified live).
    ``pitch_state`` is a 1-element mutable holding the current commanded pitch.
    """
    v_err = (float(v_center) - image_h / 2.0) / float(image_h)
    if abs(v_err) < float(getattr(cfg, "head_pitch_track_tol", 0.10)):
        return
    gain = float(getattr(cfg, "head_pitch_track_gain", -0.5))
    lo = float(getattr(cfg, "head_pitch_min_rad", -0.78))
    hi = float(getattr(cfg, "head_pitch_max_rad", 0.52))
    pitch_state[0] = min(max(pitch_state[0] + gain * v_err, lo), hi)
    api.set_head(float(cfg.head_forward_yaw_rad), pitch_state[0])


def _nudge_head_down(api: Any, cfg: Any, pitch_state: list) -> None:
    """On a head miss, look a bit further down (-pitch) to re-see a near/low box."""
    lo = float(getattr(cfg, "head_pitch_min_rad", -0.78))
    step = abs(float(getattr(cfg, "head_pitch_track_gain", -0.5))) * 0.2
    pitch_state[0] = max(pitch_state[0] - step, lo)  # down = decrease pitch
    api.set_head(float(cfg.head_forward_yaw_rad), pitch_state[0])


def _waist_probe(api: Any, cfg: Any, object_name: str, i: int) -> dict | None:
    """Probe the waist camera. Returns a terminal result dict (handoff / too_close),
    ``{"beyond": center_x}`` when the box is detected but beyond the grasp band
    (caller clamps its forward step), or ``None`` when the waist sees nothing."""
    det = api.locate_for_grasp(object_name)
    if not det.get("ok"):
        return None
    cx = det["center_mm"][0] / 1000.0
    if float(cfg.grasp_forward_min_m) <= cx <= float(cfg.grasp_forward_max_m):
        logger.info("[approach] handoff at iter %d center_x=%.3f m", i, cx)
        return {"ok": True, "handoff": True, "detection": det, "iterations": i}
    if cx < float(cfg.grasp_forward_min_m):
        logger.info("[approach] too close at iter %d center_x=%.3f m", i, cx)
        return {"ok": False, "reason": "too_close", "center_x_m": cx, "iterations": i}
    return {"beyond": cx}


def run_approach(api: Any, object_name: str = "box") -> dict:
    """Acquire (head sweep) → turn base toward target → creep + re-detect → handoff.

    Success (``ok=True, handoff=True``) means a waist ``locate_for_grasp`` sees the
    box within the graspable forward band; ``detection`` is that result (already
    cached inside the api, so ``dual_arm_grasp()`` can consume it). Failure ``reason``
    ∈ {target_not_found, lost_target, nav_failed, too_close, max_iterations}.
    """
    cfg = api.env.cfg

    # 1) Acquire by panning the head only (base does not rotate during search).
    acq = _acquire_with_head(api, object_name, cfg)
    if not acq["found"]:
        _reset_head(api, cfg)
        logger.info("[approach] target not found after head sweep")
        return {"ok": False, "reason": "target_not_found", "iterations": 0}

    # Reset head to forward, then turn the BASE by the total bearing (theta+beta).
    _reset_head(api, cfg)
    total_bearing = acq["total_bearing"]
    if abs(total_bearing) > 1e-3:
        nav = api.navigate_relative(0.0, 0.0, total_bearing)
        if not nav.get("ok"):
            return {"ok": False, "reason": "nav_failed", "nav": nav, "iterations": 0}

    # 2) Approach loop.
    lost = 0
    pitch_state = [float(cfg.head_forward_pitch_rad)]  # current commanded head pitch
    max_iter = int(cfg.approach_max_iterations)
    for i in range(1, max_iter + 1):
        s = api._nav._look_for(object_name, camera="head")

        if not s.get("found"):
            # Head lost it (likely too close / below its high FOV): try the waist handoff.
            ho = _waist_probe(api, cfg, object_name, i)
            if ho is not None and "beyond" not in ho:
                return ho
            lost += 1
            if lost >= int(cfg.lost_target_max):
                _reset_head(api, cfg)
                return {"ok": False, "reason": "lost_target", "iterations": i}
            _nudge_head_down(api, cfg, pitch_state)  # look further down to re-acquire
            continue
        lost = 0

        # (A) keep the box vertically centered as the base nears it.
        _track_head_pitch(api, cfg, pitch_state, s.get("v_center", s["image_h"] / 2.0), s["image_h"])

        # (B) probe the waist once the head sees the box reasonably large.
        bbox = s["bbox"]
        bbox_h_frac = (bbox[3] - bbox[1]) / float(s["image_h"])
        forward_step = float(cfg.approach_step_m)
        if bbox_h_frac >= float(cfg.probe_bbox_frac):
            ho = _waist_probe(api, cfg, object_name, i)
            if ho is not None:
                if "beyond" in ho:
                    forward_step = min(forward_step, ho["beyond"] - float(cfg.grasp_forward_max_m))
                else:
                    return ho

        if abs(s["u_error_frac"]) > float(cfg.center_tol_frac):
            nav = api.navigate_relative(0.0, 0.0, s["bearing_rad"])
        else:
            nav = api.navigate_relative(forward_step, 0.0, 0.0)
        if not nav.get("ok"):
            return {"ok": False, "reason": "nav_failed", "nav": nav, "iterations": i}

    return {"ok": False, "reason": "max_iterations", "iterations": max_iter}
