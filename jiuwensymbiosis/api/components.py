# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The two stateful perception/approach components, plus the reachability prior.

**What an action is** lives in :mod:`jiuwensymbiosis.api.actions`; the *simple*
implementations — the ones that are a line of delegation to an Env verb — are plain
functions in :mod:`jiuwensymbiosis.api.defaults`, which an adapter calls explicitly.
Neither needs a base class.

What is left here genuinely does: ``_Scene3DBody`` and ``_ApproachBody`` are
components with their own state (the last detection / the last sensed surface) and a
set of body hooks (how to grab a calibrated frame, which detector to run, how to
drive the base, whether a wide-angle sensor exists). They carry ~350 lines of shared
geometry between them, so a body inherits the algorithm and supplies only the hooks.

``Reachability`` is not an action provider at all: it answers a planning
question (can this arm reach that point) that the planner reads directly.

.. note::
   These are still base classes only because their hook surface has not been turned
   into a collaborator object yet. That is the intended shape — an adapter should
   *hold* a Scene3D, not *be* one — and it is a separate change from dismantling the
   thin mixins, because it touches the live perception and base-drive paths.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from jiuwensymbiosis.api.actions import (
    ANALYZE_SCENE,
    APPROACH_FOR_GRASP,
    APPROACH_FOR_PLACE,
    LOCATE_FOR_GRASP,
    LOCATE_FOR_PLACE,
    SEARCH_TARGET,
    implements,
)
from jiuwensymbiosis.motion import approach
from jiuwensymbiosis.perception import scene3d

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # Mixins are composed into BaseRobotApi subclasses, which set `self.env`.
    # Declared here only for type checking; runtime attribute is provided by
    # the composing class (see BaseRobotApi.__init__).
    from jiuwensymbiosis.env.base import BaseRobotEnv


def _screen_ref_2d(reference: str | None, relation: str) -> str | None:
    """The reference a 2-D look may screen against, or None to skip screening.

    That screen is bbox containment (``motion.approach.coarse_detect_on_reference_2d``),
    which can only decide ``on``: whether one thing is *beside* another is not recoverable
    from overlap in a flat picture, and guessing it there would steer the base at the wrong
    candidate. For every other relation the look-around falls back to a plain bearing search
    and the metric 3-D grounding up close does the deciding — later, but right.
    """
    return reference if reference and relation == "on" else None


def _unknown_relation(relation: str, object_name: str) -> dict | None:
    """Failure dict when ``relation`` is outside the closed set, else None.

    The schema advertises the enum, but nothing enforces a schema at dispatch, so a
    grounded sensing action checks before measuring anything — an unknown relation must
    fail by name rather than quietly behave like ``on``.
    """
    if relation in scene3d.SPATIAL_RELATIONS:
        return None
    return {
        "ok": False,
        "reason": f"unknown_relation:{relation}",
        "object": object_name,
        "known_relations": list(scene3d.SPATIAL_RELATIONS),
    }


class _Scene3DBody:
    """Base-frame 3-D geometry of objects and support surfaces.

    ``get_grasp_info_simple`` answers "give me a grasp pose for this object" — a
    single-gripper convenience. A body that must **drive** to a target first needs the more primitive
    answer: the target's centre, extents, footprint yaw and face normal, and a support
    surface's footprint and near-edge line. Those numbers are the same on every body, so
    the pipeline lives here (``perception.scene3d``) and an adapter supplies only the two
    genuinely body-specific pieces — how to grab a calibrated frame, and which detector
    to run.

    The grounded forms (``reference=`` + ``relation=``) resolve "the white box **on** the
    brown table", "the box **beside** the hat" and their mirrors from ONE frame; a second
    grab would be taken at a different instant and the relation could disagree with
    itself. Which relations exist is not this class's to decide — the closed set lives in
    ``perception.scene3d.SPATIAL_RELATIONS`` and the geometry in ``relation_holds``.
    """

    env: BaseRobotEnv

    # Pane/sensor name for bodies with several cameras; also labels debug overlays.
    _scene_camera: str = "scene"
    # Most recent successful sensing, so the acting step (grasp / place) need not have a
    # large nested geometry dict echoed back to it as a tool argument. A failed re-sense
    # clears it — acting on a stale geometry is worse than failing to act.
    _last_detection: dict | None = None
    _last_surface: dict | None = None

    # ---------------------------------------------------------------- body hooks
    def _grab_calibrated_frame(self, camera: str | None = None) -> Any:
        """One rgb + depth + intrinsics + base←camera frame (default: the Env verb)."""
        return self.env.grab_calibrated_frame(camera)

    def _detector_seg_fn(self) -> Any:
        """The open-vocabulary segmentation callable, or None when no detector is up."""
        return getattr(self, "_seg_fn", None)

    def _viz_update(self, camera: str, prompt: str, rgb: Any, best: dict | None) -> None:
        """Debug-overlay hook; no-op unless the body provides a viewer."""

    # ------------------------------------------------------------------- tools
    @implements(LOCATE_FOR_GRASP)
    def locate_for_grasp(self, object_name: str = "box", reference: str | None = None,
                         relation: str = "on") -> dict:
        """Detect a target and return its 3-D geometry in the robot base frame (mm).

        With ``reference`` set, accept only the candidate standing in ``relation`` to it —
        a coarse-to-fine grounding that disambiguates same-class targets (see
        ``_detect_related``).
        """
        # Invalidate any prior detection up front: a failed detect must not leave a stale
        # geometry that the next grasp would then act on. Only a fresh success re-fills it.
        self._last_detection = None
        bad = _unknown_relation(relation, object_name)
        if bad is not None:
            return bad
        frame, fail = self._calibrated_frame_or_reason(object_name)
        if fail is not None:
            return fail
        if reference:  # fine-grained: keep only the candidate related to the reference
            return self._detect_related(
                object_name, reference, relation, frame.rgb, frame.depth_m, frame.intrinsics, frame.tf_base_cam
            )
        result = scene3d.detect_object_geometry(
            frame.rgb,
            frame.depth_m,
            frame.intrinsics,
            frame.tf_base_cam,
            seg_fn=self._detector_seg_fn(),
            object_name=object_name,
            log_prefix="[scene3d-object]",
            on_pick=lambda best: self._viz_update(self._scene_camera, object_name, frame.rgb, best),
        )
        if result.get("ok"):
            self._last_detection = result
        return result

    @implements(LOCATE_FOR_PLACE)
    def locate_for_place(self, object_name: str = "table", reference: str | None = None,
                        relation: str = "on") -> dict:
        """Detect a support surface and return its base-frame top height in mm.

        With ``reference`` set, keep only the surface standing in ``relation`` to it. The
        phrase reads the same way round as everywhere else — the table that *has* a cup on
        it is the table ``relation="under"`` the cup (see ``_sense_surface_related``).
        """
        bad = _unknown_relation(relation, object_name)
        if bad is not None:
            return bad
        frame, fail = self._calibrated_frame_or_reason(object_name)
        if fail is not None:
            return fail
        if reference:  # fine-grained: keep only the surface related to the reference object
            return self._sense_surface_related(
                object_name, reference, relation, frame.rgb, frame.depth_m, frame.intrinsics, frame.tf_base_cam
            )
        result = self._sense_surface_plain(object_name, frame.rgb, frame.depth_m, frame.intrinsics, frame.tf_base_cam)
        if result.get("ok"):
            self._last_surface = result
        return result

    @implements(ANALYZE_SCENE)
    def analyze_scene(self, object_name: str = "box") -> dict:
        """Every instance of ``object_name`` in view, nearest-first."""
        import numpy as np

        from jiuwensymbiosis.perception.vision import detect_all_object_geometry

        frame, fail = self._calibrated_frame_or_reason(object_name)
        if fail is not None:
            return fail
        objs = detect_all_object_geometry(
            frame.rgb,
            frame.depth_m,
            np.asarray(frame.intrinsics),
            np.asarray(frame.tf_base_cam),
            seg_fn=self._detector_seg_fn(),
            object_name=object_name,
        )
        return {"ok": True, "object": object_name, "count": len(objs), "objects": objs}

    # ------------------------------------------------------ shared internals
    def _calibrated_frame_or_reason(self, object_name: str) -> tuple[Any, dict | None]:
        """The first frame carrying everything 3-D needs, or ``(None, failure_dict)``.

        Every camera is asked, and the one that answers is whichever FRAME turns out to
        carry depth + intrinsics + a live extrinsic — not whichever camera was written down
        as "the 3-D one". That is the same fact the caller acts on: a frame with depth
        yields a face normal to square up to, a frame without one yields at best a bearing.
        A body with a single RGBD camera and a body with a plain camera plus an RGBD one
        therefore take the same path; the second simply has one candidate that cannot answer.

        The reported reason is the last camera's, so a one-camera body still says exactly
        what was missing instead of a blanket "no camera".
        """
        fail = {"ok": False, "reason": "no_camera", "object": object_name}
        for camera in getattr(self.env, "cameras", None) or (self._scene_camera,):
            frame = self._grab_calibrated_frame(camera)
            if frame is None:
                fail = {"ok": False, "reason": "no_camera", "object": object_name}
            elif frame.depth_m is None:
                fail = {"ok": False, "reason": "no_depth", "object": object_name}
            elif frame.intrinsics is None:
                fail = {"ok": False, "reason": "no_intrinsics", "object": object_name}
            # Extrinsics are POSE-DEPENDENT: a static calib is only valid for the body pose it
            # was captured at, so a missing live TF must fail loudly rather than fall back —
            # the fallback returns coordinates for where the body used to be, and IK then aims
            # the arms there.
            elif frame.tf_base_cam is None:
                fail = {"ok": False, "reason": "no_live_tf", "object": object_name}
            else:
                return frame, None
        return None, fail

    def _candidate_geometries(
        self, object_name: str, rgb: Any, depth_m: Any, intr: Any, tf_base_cam: Any, *, min_z_mm: float | None = None
    ) -> list:
        """All ``object_name`` detections in the frame → colour-verified base-frame geometries."""
        return scene3d.candidate_geometries(
            rgb,
            depth_m,
            intr,
            tf_base_cam,
            seg_fn=self._detector_seg_fn(),
            object_name=object_name,
            min_z_mm=min_z_mm,
        )

    def _sense_surface_plain(self, object_name: str, rgb: Any, depth_m: Any, intr: Any, tf: Any) -> dict:
        """Detect a plain support surface from an ALREADY-GRABBED frame. Factored out of
        ``locate_for_place`` so a caller already holding a frame (``_detect_on_surface``) reuses
        it instead of grabbing again. Does NOT cache — that is the caller's decision."""
        return scene3d.sense_surface_geometry(
            rgb,
            depth_m,
            intr,
            tf,
            seg_fn=self._detector_seg_fn(),
            object_name=object_name,
            log_prefix="[scene3d-surface]",
            on_pick=lambda best: self._viz_update(self._scene_camera, object_name, rgb, best),
        )

    def _relation_thresholds(self) -> dict[str, float]:
        """Tolerances the relation predicate reads off the body config (missing → defaults)."""
        cfg = getattr(self.env, "cfg", None)
        return {
            "margin_mm": float(getattr(cfg, "on_surface_margin_mm", 80.0)),
            "beside_max_gap_mm": float(getattr(cfg, "beside_max_gap_mm", 400.0)),
            "near_max_dist_mm": float(getattr(cfg, "near_max_dist_mm", 1000.0)),
        }

    def _measure_reference(
        self, reference: str, relation: str, rgb: Any, depth_m: Any, intr: Any, tf_base_cam: Any
    ) -> Any:
        """The reference's geometry, measured the way this relation needs it.

        ``on`` / ``under`` are relations to a *support surface*, so the reference is sensed
        as one — that also yields the top height the target's face-normal fit must stay
        above. Every other relation is to an ordinary object, which must be measured as
        one: a surface record states no thickness, and ``beside`` compares z-extents.
        Returns None when the reference is not in view.
        """
        if relation in ("on", "under"):
            ref = self._sense_surface_plain(reference, rgb, depth_m, intr, tf_base_cam)
            return ref if ref.get("ok") else None
        refs = [g for (g, _d) in self._candidate_geometries(reference, rgb, depth_m, intr, tf_base_cam)]
        return min(refs, key=lambda g: g.center_mm[0]) if refs else None  # nearest

    def _detect_related(
        self, object_name: str, reference: str, relation: str,
        rgb: Any, depth_m: Any, intr: Any, tf_base_cam: Any,
    ) -> dict:
        """Two-stage 'X <relation> Y' grounding. Measure the reference, detect **all**
        ``object_name`` candidates from the given frame, and return the one that stands in
        ``relation`` to it. Nearest-to-robot wins. Caches the pick for the grasping step.
        """
        cfg = getattr(self.env, "cfg", None)
        # Reuse the frame already grabbed by locate_for_grasp rather than re-grabbing a second
        # one (halves grounded detection's camera cost). This is the target's reference, not a
        # place surface, so it must NOT touch self._last_surface.
        ref = self._measure_reference(reference, relation, rgb, depth_m, intr, tf_base_cam)
        if ref is None:
            return {"ok": False, "reason": f"reference_not_found:{reference}", "object": object_name}
        # Face normal from the target's own wall: for an ON relation drop points at/below the
        # reference surface top (+ a small margin) so the edge fit is the target's, not the
        # surface's dominant edge. No support surface is implied by the other relations.
        min_z = None
        if relation == "on":
            min_z = float(ref["surface_z_mm"]) + float(getattr(cfg, "grasp_face_above_surface_mm", 20.0))
        cands = self._candidate_geometries(object_name, rgb, depth_m, intr, tf_base_cam, min_z_mm=min_z)
        thresholds = self._relation_thresholds()
        picks = [(g, d) for (g, d) in cands if scene3d.relation_holds(g, ref, relation, **thresholds)]
        prompt = f"{object_name} {relation} {reference}"
        if not picks:
            self._viz_update(self._scene_camera, prompt, rgb, None)
            logger.info("[scene3d] locate_for_grasp %r %s %r: %d cand → 0 matched",
                        object_name, relation, reference, len(cands))
            return {"ok": False, "reason": "no_target_matching_reference", "object": object_name,
                    "reference": reference, "relation": relation}
        geo, det = min(picks, key=lambda gd: gd[0].center_mm[0])  # nearest (smallest forward X)
        self._viz_update(self._scene_camera, prompt, rgb, {"ok": True, **det})
        result = {
            "ok": True,
            "reason": "",
            "object": object_name,
            "reference": reference,
            "relation": relation,
            **scene3d.object_geometry_fields(geo),
        }
        self._last_detection = result
        scene3d.log_grounded_pick(
            "[scene3d] locate_for_grasp",
            object_name,
            f"{relation} {reference}",
            len(cands),
            len(picks),
            geo,
            face_flatness_max=float(getattr(cfg, "grasp_face_flatness_max", 0.15)),
            square_min_aspect=float(getattr(cfg, "grasp_square_min_aspect", 1.2)),
        )
        return result

    def _sense_surface_related(
        self, object_name: str, reference: str, relation: str,
        rgb: Any, depth_m: Any, intr: Any, tf_base_cam: Any,
    ) -> dict:
        """The place-side mirror of ``_detect_related``: among all ``object_name`` surface
        candidates keep the one standing in ``relation`` to the reference. So "the table
        that has the cup on it" arrives here as ``relation='under'``, reference='cup'.
        Nearest matching surface wins; cached for the placing step.
        """
        refs = [g for (g, _d) in self._candidate_geometries(reference, rgb, depth_m, intr, tf_base_cam)]
        if not refs:
            return {"ok": False, "reason": f"reference_not_found:{reference}", "object": object_name}
        ref = min(refs, key=lambda g: g.center_mm[0])  # nearest reference
        thresholds = self._relation_thresholds()
        cands = [g for (g, _d) in self._candidate_geometries(object_name, rgb, depth_m, intr, tf_base_cam)]
        picks = [s for s in cands if scene3d.relation_holds(s, ref, relation, **thresholds)]
        if not picks:
            logger.info("[scene3d] locate_for_place %r %s %r: %d surf → 0 matched",
                        object_name, relation, reference, len(cands))
            return {"ok": False, "reason": "no_surface_matching_reference", "object": object_name,
                    "reference": reference, "relation": relation}
        surf = min(picks, key=lambda s: s.center_mm[0])  # nearest matching surface
        # Same footprint payload as the plain path (via surface_footprint_fields) — the grounded path
        # MUST carry yaw_rad/edge_normal too, else the place-side squaring reads no normal.
        result = {"ok": True, "object": object_name, "reference": reference, "relation": relation,
                  **scene3d.surface_footprint_fields(surf)}
        self._last_surface = result
        logger.info(
            "[scene3d] locate_for_place %r %s %r: %d surf → %d matched, picked cy=%.1f z=%.1f%s",
            object_name,
            relation,
            reference,
            len(cands),
            len(picks),
            surf.center_mm[1],
            surf.top_z_mm,
            scene3d.edge_log_str(surf),
        )
        return result


class Scene3D(_Scene3DBody):
    """The 3-D sensing component an adapter **holds**, instead of inheriting.

    Same algorithm as ``_Scene3DBody`` — this subclass only supplies the plumbing that
    inheritance used to supply for free: where ``env`` comes from, where the cached sensing
    lives, and which body hooks are overridden. Adapters declare each action explicitly and
    forward to it::

        class CruzrApi(BaseRobotApi):
            def __init__(self, env, ...):
                super().__init__(env)
                self._scene = Scene3D(self)

            @implements(LOCATE_FOR_GRASP)
            def locate_for_grasp(self, object_name="box", reference=None, relation="on"):
                return self._scene.locate_for_grasp(object_name, reference, relation)

    Holding rather than inheriting is what stops "I want locate_for_grasp" from also meaning
    "…and analyze_scene, and locate_for_place" — and, because capabilities are derived from
    the actions a body implements, what stops a body from advertising a detector it has not
    got. The cached sensing deliberately stays ON THE API: ``motion/approach.py`` and the
    dual-arm grasp/place read it there, and moving it would drag the live motion path into
    this change.
    """

    def __init__(self, api: Any) -> None:
        self.api = api

    @property
    def env(self) -> Any:  # type: ignore[override]
        return self.api.env

    # --- cached sensing: one copy, owned by the api, shared with the approach loops ---
    @property
    def _last_detection(self) -> dict | None:  # type: ignore[override]
        return getattr(self.api, "_last_detection", None)

    @_last_detection.setter
    def _last_detection(self, value: dict | None) -> None:
        self.api._last_detection = value

    @property
    def _last_surface(self) -> dict | None:  # type: ignore[override]
        return getattr(self.api, "_last_surface", None)

    @_last_surface.setter
    def _last_surface(self, value: dict | None) -> None:
        self.api._last_surface = value

    # --- body hooks: the adapter's override wins, else the generic default ---
    @property
    def _scene_camera(self) -> Any:  # type: ignore[override]
        return getattr(self.api, "_scene_camera", _Scene3DBody._scene_camera)

    def _grab_calibrated_frame(self, camera: str | None = None) -> Any:
        override = getattr(self.api, "_grab_calibrated_frame", None)
        return override(camera) if override else super()._grab_calibrated_frame(camera)

    def _detector_seg_fn(self) -> Any:
        override = getattr(self.api, "_detector_seg_fn", None)
        return override() if override else getattr(self.api, "_seg_fn", None)

    def _viz_update(self, camera: str, prompt: str, rgb: Any, best: dict | None) -> None:
        override = getattr(self.api, "_viz_update", None)
        if override:
            override(camera, prompt, rgb, best)


# =============================================================================
# Mobility (mobile base / lift / waist / goal approach / dual-arm grasp)
# =============================================================================
# Same pattern as above: one ``capability`` class attr + ``@implements(SPEC)`` methods
# delegating to the Env verbs. Where no cross-body default exists (dual-arm clamping,
# lifter clearance) the method is abstract but still declares its schema and
# pre-conditions/effects here — that contract, not the implementation, is what makes a
# plan written for one body runnable on the next.


class _ApproachBody:
    """Search for a target, face it, and drive the base to a workable pose.

    ``_Scene3DBody`` answers "where is it"; this answers "get me to where I can act on
    it". A mobile body cannot act on anything it has not first driven up to, so this is
    the bridge between perceiving and acting — and none of it is body-specific: the
    search sweep, the face-normal squaring and the three approach strategies (discrete
    re-detect loop, single-shot L route, continuous servo creep) are differential-base
    geometry, tuned by :class:`~jiuwensymbiosis.motion.approach.ApproachTuning` rather
    than reimplemented. The algorithms live in :mod:`jiuwensymbiosis.motion.approach`;
    this class owns the tool schemas and the state contracts.

    Two sensors, named by role. The **precise** sensor is the short-range metric RGBD that
    ``_Scene3DBody`` reads (base-frame 3-D). The **coarse** sensor is an optional wide-FOV
    camera that yields a bearing only, letting the body find a target too far for the
    precise camera. A body without one keeps the default hooks and simply searches its
    current view plus a 180° turn.
    """

    env: BaseRobotEnv

    # ---------------------------------------------------------------- body hooks
    def _approach_tuning(self) -> approach.ApproachTuning:
        """Approach knobs read off the body config by field name (missing → defaults)."""
        return approach.ApproachTuning.from_cfg(getattr(self.env, "cfg", None))

    def _base_driver(self) -> Any:
        """Object exposing ``start/steer/hold/stop_base_drive`` + ``base_drive_running``."""
        return self.env

    def _nav_relative(self, dx_m: float, dy_m: float, dyaw_rad: float, **gains: Any) -> dict:
        """Relative base move. ``gains`` are the optional gentle approach-only steering gains;
        the default drops them — a body whose driver accepts them overrides this."""
        return self.env.navigate_relative(float(dx_m), float(dy_m), float(dyaw_rad))

    def _detector_seg_fn(self) -> Any:
        """The open-vocabulary segmentation callable, or None when no detector is up.

        The detector is a body resource that BOTH the sensing and the look-around read, so
        each resolves it from the body rather than one borrowing it off the other.
        """
        return getattr(self, "_seg_fn", None)

    def _viz_update(self, camera: str, prompt: str, rgb: Any, best: dict | None) -> None:
        """Debug-overlay hook; no-op unless the body provides a viewer."""

    def _search_frames(self, camera: str | None = None) -> Any:
        """One raw frame tuple (rgb first) from ``camera``, or None if it cannot be read.

        Used for the LOOK-AROUND pass, which only ever reports a bearing — so any camera
        will do, RGBD included. The default reads the body's single camera through the Env
        verb; a body whose extra camera needs a different path (a head on its own ROS topic,
        say) overrides this.
        """
        frame = self.env.grab_calibrated_frame(camera)
        return None if frame is None else (frame.rgb, frame.depth_m)

    def _look_for(self, object_name: str, on: str | None = None, camera: str | None = None) -> dict:
        """One look through ONE camera → a bearing dict. The single place "take a look" lives.

        Camera-pinned on purpose: a bearing is measured relative to wherever that camera was
        pointing, so a caller that aimed one (cruzr panning its head to yaw θ, then adding θ)
        must be answered by that same camera, not by whichever one happens to see the thing.
        """
        return approach.search_target(self, object_name, on, camera=camera)

    def _sweep_for_bearing(self, object_name: str, on: str | None = None) -> dict:
        """Look around from where the body stands for ``object_name``, without driving off.

        The default **turns the whole body** a step at a time and looks through every camera
        at each stop, until it finds the target or has come back round. Turning the body
        rather than aiming a camera is the general answer for one reason: every downstream
        step — grasp, place, approach — is measured in the BASE frame, so a rotation that
        carries the base frame leaves the body already facing what it found, while aiming a
        neck or a waist has to be undone by a base turn afterwards anyway. It also needs no
        extra hardware: a quadruped turning on its legs is the same capability as a
        differential base spinning in place.

        Returns ``{"found", "total_bearing", "turned_rad", "exhaustive"}``. ``total_bearing``
        is relative to where the body is standing WHEN IT RETURNS, and ``turned_rad`` says how
        far it turned to get there (0 for a body that aimed a camera instead). ``exhaustive``
        means the whole circle was covered, so the caller need not turn round and re-ask.

        A body that can aim a camera on its own overrides this — worth it only where turning
        the base is expensive or unsafe (tight spaces, carrying a payload), since it buys a
        look for less motion but still owes the base turn once the target is found.
        """
        cameras = getattr(self.env, "cameras", None) or (None,)
        # Step by a little less than one field of view, so consecutive looks overlap and no
        # sector can fall between two stops. Anything wider trades coverage for speed.
        tuning = self._approach_tuning()
        step = 0.8 * float(getattr(tuning, "head_hfov_rad", 1.2))
        can_turn = "motion.base" in getattr(self.env, "capabilities", frozenset())

        turned = 0.0
        while True:
            for camera in cameras:
                hit = self._look_for(object_name, on, camera)
                if hit.get("found"):
                    return {"found": True, "total_bearing": float(hit.get("bearing_rad", 0.0)),
                            "turned_rad": turned, "exhaustive": True}
            if not can_turn or turned + step >= 2.0 * math.pi:
                break
            nav = self.rotate_base(step)
            if not nav.get("ok"):  # blocked mid-sweep: report what was covered, do not pretend
                return {"found": False, "total_bearing": 0.0, "turned_rad": turned, "exhaustive": False}
            turned += step
        if can_turn and turned:  # came back round empty — leave the heading as it was found
            self.rotate_base(-turned)
            turned = 0.0
        return {"found": False, "total_bearing": 0.0, "turned_rad": turned, "exhaustive": can_turn}

    def _reset_search_sensor(self) -> None:
        """Re-centre an aimable camera after a sweep; no-op for a fixed one."""

    # --------------------------------------------------- search + face (internal)
    # NOT actions of their own: facing is never useful without then driving up to the
    # target, and as separate actions the mandatory order could only be stated in
    # SKILL.md prose. Folded into approach_for_grasp / approach_for_place, which the
    # contract can express (they produce the location the grasp/place then consumes).
    def _face_object(self, object_name: str = "box", reference: str | None = None,
                     relation: str = "on") -> dict:
        """Face a grasp target by its perceived bearing; coarse-search if not in view.

        Grasp-side mirror of ``_face_surface``. ``reference`` + ``relation`` restrict the
        search to the target standing in that relation (fine-grained grounding). On success
        ``locate_for_grasp`` has cached the detection so the approach / grasp steps reuse
        it; on ``object_not_found`` the caller must NOT grasp.
        """
        if reference:
            def detect(name: str) -> dict:
                return self.locate_for_grasp(name, reference=reference, relation=relation)
        else:
            detect = self.locate_for_grasp
        # Grounded grasp: the coarse sensor searches the REAL target and, for an ON relation,
        # 2-D-verifies it on the reference; the precise sensor does the final grounding up close.
        return approach.face_by_sweep(
            self,
            detect,
            object_name,
            result_key="detection",
            not_found_reason="object_not_found",
            head_name=object_name,
            head_on=_screen_ref_2d(reference, relation),
            ground_ref=reference,
        )

    def _face_surface(self, object_name: str = "table", reference: str | None = None,
                      relation: str = "on") -> dict:
        """Face a support surface by its perceived bearing; coarse-search if not in view.

        Sensor-guided replacement for a hard-coded ``rotate_base(π)``. ``reference`` +
        ``relation`` restrict the search to the surface standing in that relation. On
        success ``locate_for_place`` has cached the surface so the place step reuses it; on
        ``surface_not_found`` the caller must NOT place blindly.
        """
        if reference:
            def sense(name: str) -> dict:
                return self.locate_for_place(name, reference=reference, relation=relation)
        else:
            sense = self.locate_for_place
        # Grounded place under an UNDER relation: the coarse sensor searches the reference OBJECT
        # — a distinctive noun sitting ON the surface, so its bearing ≈ the surface's — and
        # 2-D-verifies it rests on the surface. Symmetric to _face_object.
        coarse_on_surface = bool(reference) and relation == "under"
        return approach.face_by_sweep(
            self,
            sense,
            object_name,
            result_key="surface",
            not_found_reason="surface_not_found",
            head_name=reference if coarse_on_surface else object_name,
            head_on=object_name if coarse_on_surface else None,
            ground_ref=reference,
        )

    @implements(SEARCH_TARGET)
    def search_target(self, object_name: str = "box", reference: str | None = None,
                      relation: str = "on") -> dict:
        """Look through EVERY camera at the current heading and report the first bearing found.

        All of them, because which camera happens to see a thing is not something a plan can
        know, and a camera that carries depth is not thereby disqualified from answering
        "which way" — it just ignores the depth it has (see
        ``motion.approach.search_target``). Reports the last miss when nothing is found,
        so the caller still gets the reason.
        """
        screen = _screen_ref_2d(reference, relation)
        miss: dict = {"ok": False, "found": False, "reason": "no_camera", "object": object_name}
        for camera in getattr(self.env, "cameras", (None,)):
            miss = self._look_for(object_name, screen, camera)
            if miss.get("found"):
                return miss
        return miss

    @implements(APPROACH_FOR_GRASP)
    def approach_for_grasp(self, object_name: str = "box", reference: str | None = None,
                        relation: str = "on") -> dict:
        """Search for the target, face it, then drive the base square to its face at the
        work distance.

        The search pass is skipped when a usable detection is already cached — the same
        cache ``dual_arm_grasp`` consumes — because sweeping for something we have just
        located wastes a full turn. A cache that has gone stale is not a silent hazard:
        the drive loop re-detects every iteration and fails with ``lost_after_move`` /
        ``no_detection`` rather than driving on it.
        """
        if not (self._last_detection or {}).get("ok"):
            faced = self._face_object(object_name, reference, relation)
            if not faced.get("ok"):
                return faced
        return approach.approach_for_grasp(self, None)

    @implements(APPROACH_FOR_PLACE)
    def approach_for_place(self, object_name: str = "table", reference: str | None = None,
                         relation: str = "on") -> dict:
        """Search for the surface, face it, then drive to its near edge at placing distance.

        Mirror of ``approach_for_grasp``: the search pass is skipped when a surface is already
        sensed, and already being in range means no motion at all (``status=in_range``).
        """
        if not (self._last_surface or {}).get("ok"):
            faced = self._face_surface(object_name, reference, relation)
            if not faced.get("ok"):
                return faced
        return approach.approach_for_place(self, object_name, reference, relation)

    # -------------------------------------------------------- shared internals
    # Methods, not module functions, so an adapter or a test can substitute them.
    def _drive_base(
        self,
        forward: float,
        turn: float,
        *,
        invalidate: Any,
        k_rot: float | None = None,
        k_rot_slow_rad: float | None = None,
        k_fwd: float | None = None,
    ) -> dict:
        """One approach step: advance + turn, or turn in place when already within tolerance."""
        return approach.drive_base(
            self, forward, turn, invalidate=invalidate, k_rot=k_rot, k_rot_slow_rad=k_rot_slow_rad, k_fwd=k_fwd
        )

    def _redetect(self, obj: str, reference: str | None, relation: str = "on") -> dict:
        """Post-move grounded re-detect that degrades to a plain one while still far."""
        return approach.redetect(self, obj, reference, relation)


class Approach(_ApproachBody):
    """The search-and-drive component an adapter **holds**, instead of inheriting.

    Same algorithm as ``_ApproachBody``; this subclass supplies what inheritance used to
    supply for free. It matters more here than for :class:`Scene3D`, because the loop
    functions in :mod:`jiuwensymbiosis.motion.approach` are written against an object
    with fifteen members — twelve private hooks plus ``locate_for_grasp`` /
    ``locate_for_place`` / ``rotate_base`` — an interface that had never been written down
    anywhere. This class IS that interface, stated once: what it defines it owns, what it
    forwards belongs to the body.

    The loops receive this object where they used to receive the api, so their control flow
    is untouched — the point of the change is where the actions come from, not how the base
    is driven.
    """

    def __init__(self, api: Any) -> None:
        self.api = api

    @property
    def env(self) -> Any:  # type: ignore[override]
        return self.api.env

    # --- sensing cache: owned by the api, shared with Scene3D and the grasp/place steps ---
    @property
    def _last_detection(self) -> dict | None:
        return self.api._last_detection

    @_last_detection.setter
    def _last_detection(self, value: dict | None) -> None:
        self.api._last_detection = value

    @property
    def _last_surface(self) -> dict | None:
        return self.api._last_surface

    @_last_surface.setter
    def _last_surface(self, value: dict | None) -> None:
        self.api._last_surface = value

    # --- the body's own actions the loops call back into ---
    def locate_for_grasp(self, object_name: str = "box", reference: str | None = None,
                         relation: str = "on") -> dict:
        # Ungrounded stays a one-argument call: that is the shape the loops have always used
        # when no reference was named, and forwarding the defaults explicitly would change it.
        if reference is None:
            return self.api.locate_for_grasp(object_name)
        return self.api.locate_for_grasp(object_name, reference=reference, relation=relation)

    def locate_for_place(self, object_name: str = "table", reference: str | None = None,
                         relation: str = "on") -> dict:
        if reference is None:
            return self.api.locate_for_place(object_name)
        return self.api.locate_for_place(object_name, reference=reference, relation=relation)

    def rotate_base(self, dyaw_rad: float) -> dict:
        return self.api.rotate_base(dyaw_rad)

    # --- body hooks: the adapter's override wins, else the generic default ---
    def _approach_tuning(self) -> approach.ApproachTuning:
        override = getattr(self.api, "_approach_tuning", None)
        return override() if override else super()._approach_tuning()

    def _base_driver(self) -> Any:
        override = getattr(self.api, "_base_driver", None)
        return override() if override else super()._base_driver()

    def _nav_relative(self, dx_m: float, dy_m: float, dyaw_rad: float, **gains: Any) -> dict:
        override = getattr(self.api, "_nav_relative", None)
        return override(dx_m, dy_m, dyaw_rad, **gains) if override else super()._nav_relative(
            dx_m, dy_m, dyaw_rad, **gains)

    def _search_frames(self, camera: str | None = None) -> Any:
        override = getattr(self.api, "_search_frames", None)
        return override(camera) if override else super()._search_frames(camera)

    def _sweep_for_bearing(self, object_name: str, on: str | None = None) -> dict:
        override = getattr(self.api, "_sweep_for_bearing", None)
        return override(object_name, on=on) if override else super()._sweep_for_bearing(object_name, on)

    def _reset_search_sensor(self) -> None:
        override = getattr(self.api, "_reset_search_sensor", None)
        if override:
            override()

    def _detector_seg_fn(self) -> Any:
        override = getattr(self.api, "_detector_seg_fn", None)
        return override() if override else getattr(self.api, "_seg_fn", None)

    def _viz_update(self, camera: str, prompt: str, rgb: Any, best: dict | None) -> None:
        override = getattr(self.api, "_viz_update", None)
        if override:
            override(camera, prompt, rgb, best)


class Reachability:
    """URDF-based reachability prior, HELD by a body that wants the generic judge.

    Reads ``env.urdf_path`` + ``env.arm_chains`` and runs the single-arm reach judge
    (``kinematics.reach``), degrading to ``None`` when the body exposes no URDF. NOT an
    action — the planner reads ``check_reachable`` directly, the LLM never calls it.

    A body whose geometry needs a different judge (cruzr weighs both arms plus an adaptive
    lifter) simply writes its own ``check_reachable`` and holds nothing here. Declaring
    ``planning.reachability`` is then the body's own marker, which is honest: having a URDF
    prior is a fact about the body, not about which class it inherited.
    """

    def __init__(self, api: Any) -> None:
        self.api = api

    @property
    def env(self) -> Any:
        return self.api.env

    def check_reachable(self, target: Any) -> bool | None:
        """Whether the end effector can reach ``target`` (a scene object dict with ``center_mm``, or an
        xyz-mm sequence) from the current body pose. ``None`` when the body has no URDF (caller skips).
        Any arm chain reaching the point counts as reachable."""
        urdf = getattr(self.env, "urdf_path", None)
        chains = getattr(self.env, "arm_chains", None)
        if not urdf or not chains:
            return None
        xyz = target.get("center_mm") if isinstance(target, dict) else target
        if not (isinstance(xyz, (list, tuple)) and len(xyz) == 3):
            return None
        from jiuwensymbiosis.kinematics.reach import reachable_point
        from jiuwensymbiosis.kinematics.urdf_chain import parse_chain

        q = self._reach_joint_positions()
        for root, leaf in chains.values():
            try:
                if reachable_point(parse_chain(urdf, root, leaf), xyz, q):
                    return True
            except Exception:  # noqa: BLE001 - best-effort precheck; failure → treat as not reachable
                pass
        return False

    def describe_reach(self) -> dict | None:
        """Coarse reachable-workspace envelope (forward/lateral/height ranges, m) of one arm at the
        current pose — a no-target planning prior. ``None`` when the body has no URDF."""
        urdf = getattr(self.env, "urdf_path", None)
        chains = getattr(self.env, "arm_chains", None)
        if not urdf or not chains:
            return None
        from jiuwensymbiosis.kinematics.reach import reach_envelope
        from jiuwensymbiosis.kinematics.urdf_chain import parse_chain

        root, leaf = next(iter(chains.values()))  # one representative arm
        try:
            return reach_envelope(parse_chain(urdf, root, leaf), self._reach_joint_positions())
        except Exception:  # noqa: BLE001 - best-effort; no prior on failure
            return None

    def _reach_joint_positions(self) -> dict[str, float]:
        """Current joint angles for the reachability IK, read from the env observation. Adapters with a
        richer/faster joint source may override."""
        try:
            obs = self.env.get_observation()
            jp = (obs.extra or {}).get("joint_positions") if obs is not None else None
            return dict(jp) if jp else {}
        except Exception:  # noqa: BLE001
            return {}
