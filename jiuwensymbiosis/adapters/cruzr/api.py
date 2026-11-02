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
    DRIVE_ARC,
    DUAL_ARM_GRASP,
    DUAL_ARM_PLACE,
    GET_IMAGE,
    GET_JOINT_POSITIONS,
    HOME,
    LIFT_TO_CLEARANCE,
    LOCATE_FOR_GRASP,
    LOCATE_FOR_PLACE,
    MOVE_JOINT,
    MOVE_NAMED_JOINT,
    NAVIGATE_RELATIVE,
    PIXEL_TO_BASE_XYZ,
    ROTATE_BASE,
    SEARCH_TARGET,
    SET_LIFT_POSE,
    TURN_WAIST,
    implements,
)
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.motion import approach, dual_arm, lift
from jiuwensymbiosis.perception import scene3d
from jiuwensymbiosis.perception.config import DetectorConfig
from jiuwensymbiosis.perception.detector_client import create_detector_client, init_detector
from jiuwensymbiosis.perception.frame import project_to_base
from jiuwensymbiosis.perception.object_geometry import ObjectGeometry3D
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

# A payload can arrive straight from an LLM tool call, so a missing field is bad INPUT, not a
# bug: the callers turn it into a structured failure instead of a bare KeyError traceback.
_BOX_FIELDS = ("center_mm", "width_mm", "height_mm", "front_x_mm", "top_z_mm", "n_points")
_SURFACE_FIELDS = ("front_x_mm", "back_x_mm", "center_mm", "width_mm", "surface_z_mm")


def _geometry_from_payload(box: dict) -> Optional[ObjectGeometry3D]:
    """Rebuild ``ObjectGeometry3D`` from a detection payload; ``None`` if a field is missing."""
    if any(f not in box for f in _BOX_FIELDS):
        return None
    return ObjectGeometry3D(
        True, "", tuple(box.get("center_mm")), box.get("width_mm"), box.get("height_mm"),
        box.get("front_x_mm"), box.get("top_z_mm"), box.get("n_points"),
        back_x_mm=box.get("back_x_mm", 0.0),
    )


class CruzrApi(BaseRobotApi):
    """Cruzr mobile dual-arm: base + lifter + waist + paddle grasp + waist RGBD vision.

    Every action this body offers is declared in this file, so it IS the capability
    list. The remaining held object is not an action bundle: Scene3D
    are stateful components with body hooks, and Reachability answers a planning
    question rather than exposing an action.
    """

    # Scene 3-D sensing runs off the waist RGBD camera (the head camera is wide-FOV 2-D only).
    scene_camera = "waist"

    # Marker capability: what the end effector IS, which no ACTION advertises — the dual-arm
    # actions gate on the TOPOLOGY (motion.dual_arm), so nothing would derive this. Two plates
    # that clamp a face each side; a dual-arm body carrying grippers would say grasp.parallel
    # here instead and call the same actions.
    #
    # planning.reachability is NOT here: it is DERIVED — this body holds its own judge
    # (``check_reachable`` below weighs both arms plus an adaptive lifter, rather than the
    # generic single-arm one) and its Env ships the URDF that judge reads, so the api ∩ env
    # intersection says so without anyone writing it down.
    capability = {"grasp.paddle"}

    def __init__(
        self,
        env: CruzrEnv,
        *,
        detector_service_url: str | None = "http://127.0.0.1:8114",
        detector_client: Any = None,
        camera_calib_path: Optional[str] = None,
    ) -> None:
        """Bind to a Cruzr environment; configure detector + camera calibration."""
        super().__init__(env)
        # Held, not inherited: this body declares each sensing action below and forwards to
        # the component, so api.py stays the complete list of what Cruzr offers.
        self._detector_service_url = detector_service_url
        self._camera_calib_path = camera_calib_path
        self._seg_fn = detector_client
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
        # _last_detection / _last_surface are NOT initialised here: BaseRobotApi owns them
        # (super().__init__ above) together with the invalidation that keeps them in step
        # with memory.locations. dual_arm_grasp / dual_arm_place consume them so detection
        # stays an explicit separate step and the LLM need not echo the nested geometry
        # dict back as a tool argument. Re-declaring them here would read as a second cache.
        # Lazily-created debug detection window (waist+head panes). None until the first
        # detection; stays a no-op unless cfg.viz_detections / $JIUWEN_CRUZR_VIZ=1.
        self._viz = None

    # No raise_/lower_arm tools: "raise the left arm" is one named joint driven to a
    # configured angle, which ``move_named_joint`` already says — and says on any body
    # with ``motion.joint``, where a cruzr-only tool said it nowhere else.

    @implements(MOVE_JOINT)
    def move_joint(self, targets: dict[str, float]) -> dict:
        """Move the named joints to absolute radians, holding the rest.

        A bare joint VECTOR has no meaning on this body — two arms plus a waist and a lifter
        have no single chain order — which is why the shared action speaks names. Names come
        from ``get_joint_positions`` / the URDF.
        """
        logger.info("[CruzrApi] move_joint %s", {k: round(float(v), 3) for k, v in targets.items()})
        return self.env.move_named_joints({k: float(v) for k, v in targets.items()})

    @implements(MOVE_NAMED_JOINT)
    def move_named_joint(self, joint_name: str, position_rad: float) -> dict:
        """Move a named joint to a target position."""
        logger.info("[CruzrApi] move_named_joint %s -> %.3f", joint_name, position_rad)
        return self.env.move_named_joints({joint_name: float(position_rad)})

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
        hold = {}
        for arm_joints in (ARM_JOINTS["left"], ARM_JOINTS["right"]):
            for j in arm_joints:
                if j in q:
                    hold[j] = float(q[j])
        logger.info("[CruzrApi] turn_waist from=%.3f delta=%.3f -> %.3f clamped=%s",
                    frm, delta_rad, target, clamped)
        res = self._ll().turn_waist_blocking(target, hold=hold, waist_joint=waist)
        return {"ok": True, "joint": waist, "from_rad": frm, "to_rad": target,
                "delta_rad": target - frm, "clamped": clamped, "readback": res.get("readback")}

    def look_for(self, object_name: str, on: str | None = None, *, camera: str | None = None) -> dict:
        """One look through ``camera`` → a bearing dict.

        NOT an action: the head-search helpers in this module need it. Camera-pinned on
        purpose — a bearing is measured relative to wherever that camera was pointing, so a
        caller that aimed one (panning the head to yaw θ, then adding θ) must be answered by
        that same camera, not by whichever one happens to see the thing.
        """
        return approach.look_once(self, object_name, on, camera=camera)

    def set_head(self, yaw_rad: float, pitch_rad: float) -> dict:
        """Move the head yaw+pitch via the CONTINUOUS SDK path (move_joints ramp).

        The head only actuates under a sustained high-rate RobotCommand stream;
        a single ``move_named_joint`` pulse does NOT move it. Sign (verified live):
        ``+pitch = look up, -pitch = look down`` (look down to keep a floor box in
        the high-mounted head camera's view as the base nears).
        """
        cfg = self.env.cfg
        logger.info("[CruzrApi] set_head yaw=%.3f pitch=%.3f", yaw_rad, pitch_rad)
        return self.env.move_named_joints(
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
    # NOT an action: a 2-D debug view of the detector, for eyeballing what it saw.
    # The plannable answer is ``locate_for_grasp``. Called from
    # ``scripts/cruzr/debug_detect.py``; never emitted as a tool.
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
        — a missing live TF stays None and the mixin fails loudly).
        """
        from jiuwensymbiosis.perception.frame import CameraFrame

        frames = self._ll().grab_frames(camera=camera or "waist")
        if frames is None:
            return None
        rgb, depth_m, k_live, tf_live = frames
        return CameraFrame(rgb=rgb, depth_m=depth_m,
                           intrinsics=k_live if k_live is not None else self._calib_intrinsics(),
                           tf_base_cam=tf_live)

    def detector_seg_fn(self) -> Any:
        """Lazily bind the detection sidecar, then hand the mixin its segmentation callable."""
        self._ensure_detector()
        return self._seg_fn

    def viz_update(self, camera: str, prompt: str, rgb: Any, best: Optional[dict]) -> None:
        """Push one detection frame to the (lazily-created) debug window. No-op unless enabled via
        cfg.viz_detections or $JIUWEN_CRUZR_VIZ=1. ``best`` is a _run_detect_pick_best-style dict
        ({mask, box, score, ok}) or None for a miss.
        """
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
        return defaults.locate_for_grasp(self, object_name, reference, relation)

    @implements(LOCATE_FOR_PLACE)
    def locate_for_place(self, object_name: str = "table", reference: Optional[str] = None,
                         relation: str = "on") -> dict:
        return defaults.locate_for_place(self, object_name, reference, relation)

    @implements(ANALYZE_SCENE)
    def analyze_scene(self, object_name: str = "box") -> dict:
        return defaults.analyze_scene(self, object_name)

    # ------------------------------------------------------------ search + approach
    @implements(SEARCH_TARGET)
    def search_target(self, object_name: str = "box", reference: Optional[str] = None,
                      relation: str = "on") -> dict:
        return defaults.search_target(self, object_name, reference, relation)

    @implements(APPROACH_FOR_GRASP)
    def approach_for_grasp(self, object_name: str = "box", reference: Optional[str] = None,
                           relation: str = "on") -> dict:
        return defaults.approach_for_grasp(self, object_name, reference, relation)

    @implements(APPROACH_FOR_PLACE)
    def approach_for_place(self, object_name: str = "table", reference: Optional[str] = None,
                           relation: str = "on") -> dict:
        return defaults.approach_for_place(self, object_name, reference, relation)

    # ------------------------------------------------- end-effector hook (grasp.paddle)
    def grasp_plan(self, box: Any, *, inset_mm: float | None = None,
                   pre_clear_mm: float = 60.0) -> tuple[dict, dict, dict]:
        """Where the two contact points go — THIS BODY'S end effector decides.

        Paddle geometry: two flat plates that clamp a face each side, TCP 9 cm along the
        tool-x axis, overshooting the faces by ``grasp_inset_mm`` so they press in. Returns
        ``(approach, descend, clamp)`` waypoint sets, one ArmTarget per arm.

        This is the seam between the two axes. A dual-arm body carrying grippers or a hand
        overrides this with its own contact planning; everything that consumes it — two-arm
        IK, the approach→descend→contact sequence, the force confirmation — is the same job
        whatever is on the end, and is shared.
        """
        from jiuwensymbiosis.adapters.cruzr.geometry import plan_clamp_targets

        cfg = self.env.cfg
        return plan_clamp_targets(
            box,
            inset_mm=float(cfg.grasp_inset_mm) if inset_mm is None else float(inset_mm),
            pre_clear_mm=pre_clear_mm,
        )

    def ready_plan(self) -> dict:
        """Transit pose: both paddles in front of the chest, facing inward, spread WIDE and
        held high — clear of a table in front. An END-EFFECTOR fact (where two plates wait is
        not where two grippers wait), so it sits beside ``grasp_plan``.
        """
        from jiuwensymbiosis.adapters.cruzr.geometry import ready_targets

        return ready_targets()

    def lifter_for_object(self, box: Any, q_fixed: dict, waist_yaw: float) -> Any:
        """Which torso pose best reaches ``box`` — this body leans, so it searches."""
        from jiuwensymbiosis.adapters.cruzr.geometry import LIFTER_JOINTS, search_lifter_for_box
        from jiuwensymbiosis.kinematics.urdf_chain import parse_chain

        cfg = self.env.cfg
        left = parse_chain(cfg.urdf_path, "base_link", cfg.left_arm_leaf)
        right = parse_chain(cfg.urdf_path, "base_link", cfg.right_arm_leaf)
        current = {j: q_fixed[j] for j in LIFTER_JOINTS}
        return search_lifter_for_box(box, left, right, current, waist_yaw)

    def contact_confirmed(self) -> tuple[bool, Any]:
        """Both hands' force/torque above the configured threshold. SAFETY: no lift without it."""
        thr = float(self.env.cfg.contact_force_threshold_n)
        ft = {arm: self._ll().read_hand_ft(arm) for arm in ("left", "right")}
        held = all(ft[a].get("ok") and ft[a].get("fmag", 0.0) >= thr for a in ("left", "right"))
        return held, ft

    def place_plan(self, box: Any, landing: tuple, held: dict, *, inset_mm: float = 6.5,
                   place_squeeze_mm: float | None = None, **_: Any) -> tuple[dict, dict, dict]:
        """Paddle waypoints for setting the box down at ``landing``.

        Two things here are paddle facts and nothing else's: the CLAMP targets keep the gap
        the grasp actually achieved (re-deriving it from the box width squeezes or drops the
        box during the long, arms-extended descent), and each paddle is pulled a little PAST
        that held position toward the centre so the arms actively press in — commanding the
        exact held gap leaves near-zero position error, so a stiff controller applies almost
        no inward force and the box slides out.
        """
        from dataclasses import replace

        from jiuwensymbiosis.adapters.cruzr.geometry import plan_clamp_targets
        from jiuwensymbiosis.motion.dual_arm import both

        cfg = self.env.cfg
        b = box
        landing_x, landing_y, surf_z = landing
        # The two contact points as measured right now, in the (x-list, y-list) shape the
        # spacing arithmetic below reads.
        cx = [held["left"][0], held["right"][0]]
        cy = [held["left"][1], held["right"][1]]
        squeeze_mm = float(getattr(cfg, "place_squeeze_mm", 0.0)
                           if place_squeeze_mm is None else place_squeeze_mm)
        # Rebuild LEVEL place targets at the landing x,y (front == back == depth mid so the
        # clamp sits at landing_x). No waist-delta rotation is needed — FK already reflects
        # the current waist. _apply_surface_z then lands the box bottom on the surface
        # (no-op when surf_z is None).
        b_now = replace(b, center_mm=(landing_x, landing_y, b.center_mm[2]),
                        front_x_mm=landing_x, back_x_mm=landing_x)
        approach_t, descend, clamp = plan_clamp_targets(b_now, inset_mm=inset_mm)
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
        squeeze_half_m = 0.5 * squeeze_mm / 1000.0
        inward = {"left": -1.0, "right": 1.0}   # left paddle is +Y → inward is -Y; right mirrors
        clamp = {a: replace(tgt, pos_m=(cx[i] + dx_m,
                                        cy[i] + dy_m + sign * squeeze_half_m,
                                        tgt.pos_m[2]))
                 for i, (a, tgt, sign) in enumerate(both(clamp, inward))}
        approach_t, descend, clamp = self._apply_surface_z(b, (approach_t, descend, clamp), surf_z)
        return approach_t, descend, clamp

    def lower_and_release(self, chains: dict, arm_joints: dict, q_fixed: dict,
                          plan: tuple, held_q: dict, *, descend_dz_m: float = 0.02,
                          **_: Any) -> dict:
        """Descend keeping the paddle gap, slide the paddles off the rim, then open outward.

        The descent is streamed in CARTESIAN, not as one joint ramp: a joint ramp only holds
        the gap at its two endpoints and bows it OPEN in between, which is exactly when the
        arms are most extended and least stiff — the box slips out. Interpolating the TCPs
        makes the commanded gap only ever tighten toward the squeeze.
        """
        from dataclasses import replace

        import numpy as np

        from jiuwensymbiosis.adapters.cruzr.geometry import (
            LIFTER_JOINTS,
            solve_arm_ik,
        )
        from jiuwensymbiosis.kinematics.fk import fk_chain
        from jiuwensymbiosis.motion.dual_arm import both

        cfg = self.env.cfg
        approach_t, descend, clamp = plan
        cur = held_q
        q = self._ll().get_joint_positions() or {}
        fixed_names = tuple(self.env.torso_joints)
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
        down = {a: solve_arm_ik(chain, q_fixed, a, tgt, q_init=c,
                                check_collision=True, package_dir=cfg.urdf_package_dir)
                for a, chain, tgt, c in both(chains, clamp, cur)}
        n_wp = max(1, int(getattr(cfg, "place_lower_waypoints", 4)))
        stream_ok = n_wp > 1 and all(down[a].converged for a in ("left", "right"))
        # Interpolate the descent from the ACTUAL current (post-lean) paddle TCP — NOT the pre-lean
        # carried held_tcp. The lifter lean above (if any) moved the shoulders while holding the arms,
        # so the box is no longer at held_tcp; starting the Cartesian interp from the stale pre-lean TCP
        # made the first stream segment swing the box UP/BACK toward the carry height — into the chest.
        # FK the post-lean pose so the stream is a clean straight descent from where the box IS now.
        start_tcp: dict[str, tuple[float, float, float]] = {}
        for a, chain, tgt in both(chains, clamp):
            tf = fk_chain(chain, q)
            p = tf[:3, 3] + tf[:3, :3] @ np.asarray(tgt.tcp_offset_local, dtype=float)
            start_tcp[a] = (float(p[0]), float(p[1]), float(p[2]))
        knots: list[dict] = [{**cur["left"], **cur["right"]}]   # start = measured, no jump
        prev = dict(cur)
        for k in range(1, n_wp):                                # interior knots 1..n_wp-1
            if not stream_ok:
                break
            knot: dict = {}
            f = k / n_wp
            for a, chain, tgt, sp, qp in both(chains, clamp, start_tcp, prev):
                ep = tgt.pos_m
                pos = tuple(sp[j] + f * (ep[j] - sp[j]) for j in range(3))
                r = solve_arm_ik(chain, q_fixed, a, replace(tgt, pos_m=pos),
                                 q_init=qp, check_collision=True, package_dir=cfg.urdf_package_dir)
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
        rel = {a: solve_arm_ik(chain, q_fixed, a, tgt,
                               q_init=d.q if d.converged else c,
                               check_collision=True, package_dir=cfg.urdf_package_dir)
               for a, chain, tgt, d, c in both(chains, open_tgt, down, cur)}
        # open (non-squeeze, box already resting on the surface) → fast transit ramp
        self._move_arms_sync({a: r.q for a, r in rel.items() if r.converged},
                             ramp_duration_s=place_transit_ramp)
        # Cleared here, not at the return: the paddles are already open, so a failure
        # during the raise below must not make RecoveryRail preserve a phantom payload.
        self.env.holding_payload = False
        # 3. raise clear: up to the approach pose (above the box) before homing,
        #    so the hands lift off top-down instead of dragging across the table
        up = {a: solve_arm_ik(chain, q_fixed, a, tgt,
                              q_init=r.q if r.converged else c,
                              check_collision=True, package_dir=cfg.urdf_package_dir)
              for a, chain, tgt, r, c in both(chains, approach_t, rel, cur)}
        # raise clear (empty arms, box already placed) → fast transit ramp
        self._move_arms_sync({a: r.q for a, r in up.items() if r.converged},
                             ramp_duration_s=place_transit_ramp)
        # 4. back to the raised "ready" posture (arms up in front, clear). The
        #    return-to-zero (straighten lifter -> 0, then arms -> 0) is delegated
        #    to home() — call it after placing to retreat to a safe home.
        return {"ok": True}

    @property
    def last_grasped(self) -> dict | None:
        """What dual_arm_grasp last took hold of — what a bare dual_arm_place acts on."""
        return self._last_grasped_box

    def lift_plan(self, box: Any, chains: dict, q_upright: dict, target_z_m: float) -> dict:
        """Paddle TCPs at the transit height: keep the live x/y + orientation, set only z.

        Both paddles go to the SAME absolute z, so the gap — and with it the box level — is
        preserved exactly.
        """
        from dataclasses import replace

        import numpy as np

        from jiuwensymbiosis.adapters.cruzr.geometry import (
            TOOL_APPROACH_LOCAL,
            TOOL_PADDLE_LOCAL,
            plan_clamp_targets,
        )
        from jiuwensymbiosis.kinematics.fk import fk_chain
        from jiuwensymbiosis.motion.dual_arm import both

        target_z = target_z_m
        b = box
        _, _, clamp = plan_clamp_targets(b)              # per-arm tcp_offset template only
        lifted = {}
        for a, chain, tgt in both(chains, clamp):
            tf = fk_chain(chain, q_upright)
            r = tf[:3, :3]
            tcp = tf[:3, 3] + r @ np.asarray(tgt.tcp_offset_local, dtype=float)
            lifted[a] = replace(
                clamp[a],
                pos_m=(float(tcp[0]), float(tcp[1]), target_z),
                approach=tuple(float(v) for v in r @ np.asarray(TOOL_APPROACH_LOCAL, dtype=float)),
                paddle=tuple(float(v) for v in r @ np.asarray(TOOL_PADDLE_LOCAL, dtype=float)),
            )
        return lifted

    # ------------------------------------------------- place hooks (grasp.paddle)
    def carried_contacts(self, chains: dict, q_all: dict, box: Any, *,
                         inset_mm: float = 6.5, **_: Any) -> dict:
        """Where each paddle TCP is right now, by FK of the measured joints."""
        import numpy as np

        from jiuwensymbiosis.adapters.cruzr.geometry import plan_clamp_targets
        from jiuwensymbiosis.kinematics.fk import fk_chain
        from jiuwensymbiosis.motion.dual_arm import both

        tmpl = plan_clamp_targets(box, inset_mm=float(inset_mm))[2]
        out: dict = {}
        for a, chain, t in both(chains, tmpl):
            tf = fk_chain(chain, q_all)
            p = tf[:3, 3] + tf[:3, :3] @ np.asarray(t.tcp_offset_local, dtype=float)
            out[a] = (float(p[0]), float(p[1]), float(p[2]))
        return out

    def lifter_for_place(self, clamp: dict, q_fixed: dict) -> Any:
        """Smallest level-manifold lean from which both arms reach the place targets."""
        from jiuwensymbiosis.adapters.cruzr.geometry import LIFTER_JOINTS, search_lifter_for_place
        from jiuwensymbiosis.kinematics.urdf_chain import parse_chain

        cfg = self.env.cfg
        chains = {"left": parse_chain(cfg.urdf_path, "base_link", cfg.left_arm_leaf),
                  "right": parse_chain(cfg.urdf_path, "base_link", cfg.right_arm_leaf)}
        cur_lifter = {j: q_fixed[j] for j in LIFTER_JOINTS}
        return search_lifter_for_place(clamp, chains["left"], chains["right"], cur_lifter,
                                       q_fixed[self.env.waist_joint],
                                       max_lean_rad=float(cfg.place_max_lift_lean_rad))

    # ---------------------------------- approach body hooks (motion/approach.py reads these)
    def base_driver(self) -> Any:
        """Drive the servo worker through the driver directly (the SDK handle is its own)."""
        return self._ll()

    def nav_relative(self, dx_m: float, dy_m: float, dyaw_rad: float, **gains: Any) -> dict:
        """Relative base move that forwards the gentle approach-only steering gains."""
        return self._ll().navigate_relative(float(dx_m), float(dy_m), float(dyaw_rad), **gains)

    def search_frames(self, camera: Optional[str] = None) -> Any:
        """One raw frame tuple from ``camera``. The head is a stereo pair read as its RIGHT eye
        alone — no depth on that path, which costs nothing here: looking around wants a bearing.
        """
        return self._ll().grab_frames(camera=camera or "waist")

    def sweep_for_bearing(self, object_name: str, on: Optional[str] = None) -> dict:
        """Pan the HEAD left + right over the current facing — cruzr can aim a camera without
        moving the base, so it looks around with the neck before anything else turns.
        """
        return _acquire_with_head(self, object_name, self.env.cfg, on=on)

    def reset_search_sensor(self) -> None:
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
        ``grasp_arc_enabled``. Does NOT touch the arms.
        """
        logger.info("[CruzrApi] drive_arc radius=%.3f dyaw=%.3f", radius_m, dyaw_rad)
        return self._ll().navigate_arc(float(radius_m), float(dyaw_rad))


    def check_reachable(self, target: dict) -> bool:
        """Override of ``Reachability.check_reachable`` with cruzr's exact DUAL-ARM + lifter judge:
        当前底盘/关节姿态下，双臂能否抓到该 base 系目标(允许自适应 lifter，不含底盘移动)。纯离线只读，
        复用 dual_arm_grasp 的 IK 可达搜索，无运动/无硬件副作用。供 fast 规划器判断"要不要先编长距离移动"。
        保守快筛：不确定/无关节状态/出错一律判不可达(保留 approach_for_grasp 兜底，对误判鲁棒)。target 为
        analyze_scene 形状的目标 dict(含 center_mm)。
        """
        from jiuwensymbiosis.adapters.cruzr.geometry import LIFTER_JOINTS, search_lifter_for_box
        from jiuwensymbiosis.kinematics.urdf_chain import parse_chain

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
        """Both paddles take hold of an already-detected box and confirm by force.

        The sequence is shared (``motion/dual_arm.py``); this body supplies the four seams it
        needs — ``grasp_plan`` and ``ready_plan`` (paddle geometry), ``lifter_for_object`` (it
        leans) and ``contact_confirmed`` (hand FT).
        """
        out = dual_arm.dual_arm_grasp(self, target, object_name)
        if out.get("ok"):
            # Bookkeeping this body needs for a later bare dual_arm_place: WHAT was grasped,
            # and at WHICH waist angle — if turn_waist rotates the torso in between, place
            # rotates its paddle targets by the difference.
            self._last_grasped_box = out["box"]
            q = self._ll().get_joint_positions() or {}
            self._last_grasp_waist_yaw = float(q.get(self.env.cfg.waist_yaw_joint, 0.0))
        else:
            self._last_grasped_box = None
            self._last_grasp_waist_yaw = None
        return out


    def _move_arms_sync(self, q_by_arm: dict, *, ramp_duration_s: Optional[float] = None) -> None:
        """Command BOTH arms in ONE message so they move simultaneously. ``ramp_duration_s``
        overrides the global ramp for this move only — non-contact transit moves in grasp/place
        (ready / descend-outside-faces / raise-clear) pass the shorter
        ``cfg.arm_transit_ramp_duration_s``; the clamp and place-lower keep the careful default.
        """
        combined: dict = {}
        for arm_q in q_by_arm.values():
            combined.update(arm_q)
        if combined:
            self.env.move_named_joints(combined, ramp_duration_s=ramp_duration_s)

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
        _fixed_names = ("lifter_pitch_1_joint", "lifter_pitch_2_joint",
                        "lifter_pitch_3_joint", cfg.waist_yaw_joint)
        qf = {n: float(qq.get(n, 0.0)) for n in _fixed_names}
        zeros = dict.fromkeys(cur, 0.0)

        def _one_arm_zeroed(base: dict, arm: str) -> dict:
            """``base`` with ``arm``'s joints set to 0 (the other arm held where base has it)."""
            m = dict(base)
            m.update(dict.fromkeys(ARM_JOINTS[arm], 0.0))
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

    # No ``home_arms`` tool: it was one line of ``self._safe_home_arms()``, and ``home``
    # — the action every body owes — already runs it as part of the safe posture.

    @implements(LIFT_TO_CLEARANCE)
    def lift_to_clearance(self, box: Optional[dict] = None, upright_tol_rad: float = 0.05) -> dict:
        """Stand the torso up and raise the held box to the transit height.

        Shared sequence (``motion/lift.py``); this body supplies ``lift_plan`` — where the
        paddle TCPs end up once the box is up.
        """
        return lift.lift_to_clearance(self, box, upright_tol_rad)

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

        at_home_posture = arms_home and waist_home
        if have_state and upright and at_home_posture:
            return {"ok": True, "skipped": "already_home"}
        if have_state and upright:
            self._neutralize_waist(tol_rad)   # body straight: no table to clear
            return self._safe_home_arms()

        # Leaned forward (or unknown): straighten the lifter, neutralize the waist, then home.
        self._ll().set_lifter(dict.fromkeys(LIFTER_JOINTS, 0.0))
        self._neutralize_waist(tol_rad)
        return self._safe_home_arms()

    def _neutralize_waist(self, tol_rad: float = 0.05) -> None:
        """Turn waist_yaw back to 0 if rotated, holding the arms where they now are."""
        from jiuwensymbiosis.adapters.cruzr.geometry import ARM_JOINTS

        cfg = self.env.cfg
        qq = self._ll().get_joint_positions() or {}
        if abs(qq.get(cfg.waist_yaw_joint, 0.0)) <= tol_rad:
            return
        hold = {}
        for arm_joints in (ARM_JOINTS["left"], ARM_JOINTS["right"]):
            for j in arm_joints:
                if j in qq:
                    hold[j] = float(qq.get(j, 0.0))
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
        self._ll().set_lifter(dict.fromkeys(LIFTER_JOINTS, 0.0))
        self._neutralize_waist(tol_rad)
        return {"ok": True, "payload_preserved": True}

    @implements(DUAL_ARM_PLACE)
    def dual_arm_place(self, target: dict | None = None, inset_mm: float = 6.5,
                       descend_dz_m: float = 0.02, surface_z_mm: float | None = None,
                       surface: dict | None = None) -> dict:
        """Set the held box down on a sensed surface and withdraw.

        The sequence is shared (``motion/dual_arm.py``); this body supplies the paddle seams:
        ``place_plan`` (keep the held gap, add the squeeze), ``lower_and_release`` (Cartesian
        descent, slide off the rim, open outward), ``carried_contacts`` and ``lifter_for_place``.
        """
        return dual_arm.dual_arm_place(
            self, target, surface, inset_mm=inset_mm, descend_dz_m=descend_dz_m,
            surface_z_mm=surface_z_mm)


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
        """Bind the legacy URL only for directly constructed, unmanaged APIs."""
        if self._seg_fn is None:
            self._seg_fn = (
                create_detector_client(DetectorConfig())
                if self._detector_service_url is None
                else init_detector(self._detector_service_url, request_scoped=True)
            )

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
    the approach/paddle direction vectors and tool-frame offset are unchanged.
    """
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
# declaring it lost. run_approach is the search-grasp demo entry, not an action.
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
        s = api.look_for(object_name, on, camera="head")
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
    (caller clamps its forward step), or ``None`` when the waist sees nothing.
    """
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
        s = api.look_for(object_name, camera="head")

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
