# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Body-agnostic base approach: coarse search → face the target → converge to a work pose.

Three layers, each reusable by any differential-drive mobile manipulator:

* **pure geometry** — the advance-step policies and the face/edge normal selectors
  (:func:`forward_step`, :func:`near_face_normal`, …). No robot, no state, no I/O.
* **tuning** — :class:`ApproachTuning` gathers the knobs the loops read off a body
  config **by field name**, so a second body only has to name its YAML fields the
  same way; a field it omits falls back to the documented default here.
* **loops** — ``api``-first functions (:func:`face_by_sweep`, :func:`approach_for_grasp`,
  …) driven through a handful of hooks on the api object, so the hardware-specific
  half (which camera, which drive handle) stays with the adapter.

``_ApproachBody`` in :mod:`jiuwensymbiosis.api.components` is the thin ``@robot_tool``
facade over this module: it owns the tool schemas and the state contracts, this
module owns the algorithm — the same split as ``_Scene3DBody`` ↔
:mod:`jiuwensymbiosis.perception.scene3d`.

Two sensors, by role rather than by mounting point. The **precise** sensor is the
short-range metric RGBD that yields base-frame 3-D (Cruzr's waist camera); the
**coarse** sensor is an optional wide-FOV camera that yields a bearing only
(Cruzr's head). A body with no coarse sensor degrades gracefully: the search then
only covers the precise sensor's current view plus the 180° turn.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, fields
from typing import Any, Literal, NamedTuple, TypedDict

from jiuwensymbiosis.motion.base_goal import plan_base_goal_for_grasp, plan_grasp_right_angle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result shapes, declared next to the functions that build them (same rule as
# perception/scene3d.py). Fed to ``ActionSpec.result`` so ``parse_sequence`` can
# check every ``<bind>.field`` a plan reads off an approach step.
# ---------------------------------------------------------------------------
class ApproachFailure(TypedDict, total=False):
    """Shape an approach returns when it did not reach a workable pose.

    A caller that sees this must NOT act on the target: either nothing was found
    (``object_not_found`` / ``no_detection``) or the base ended up somewhere it
    cannot work from (``too_close`` / ``lidar_blocked`` / ``lost_after_move``).
    """

    ok: Literal[False]
    reason: str
    turn_rad: float
    forward_m: float
    iters: int


class ApproachResult(TypedDict, total=False):
    """Success shape of an approach — the base is at a pose the arms can work from.

    ``status`` is ``in_band`` / ``in_range`` when the convergence check passed, and
    ``max_iters`` when the loop ran out while still improving (still a usable pose,
    but the caller may want to re-sense).
    """

    ok: Literal[True]
    status: str
    turn_rad: float
    forward_m: float
    iters: int


class SearchResult(TypedDict, total=False):
    """Coarse bearing-only search result: direction to the target, never a distance."""

    ok: bool
    found: bool
    verified: bool
    object: str
    reference: str
    camera: str
    bearing_rad: float  # + = target left of the body's heading
    score: float
    overlap: float
    reason: str


@dataclass(frozen=True)
class ApproachTuning:
    """The knobs the approach loops read, gathered off a body config by field name.

    ``from_cfg`` is deliberately name-based rather than an explicit mapping: an adapter
    that names its YAML fields like these gets the whole approach stack for free, and one
    that omits a field silently keeps the default documented here. Defaults are the ones
    the algorithms themselves used before this module existed, so behaviour is unchanged
    for a config that already sets a field.
    """

    # Convergence geometry shared by both sides.
    base_pos_tol_m: float = 0.05
    base_yaw_tol_rad: float = 0.05
    approach_converge_iters: int = 4
    approach_forward_step_m: float = 0.25
    # Gentle approach-only base gains; ``None`` leaves the driver's global gains in place.
    approach_k_rot: float | None = None
    approach_k_rot_slow_rad: float | None = None
    approach_k_fwd: float | None = None

    # Grasp side.
    grasp_target_forward_m: float = 0.40
    grasp_forward_min_m: float = 0.30
    grasp_forward_max_m: float = 0.50
    grasp_square_tol_rad: float = 0.26
    grasp_face_flatness_max: float = 0.15
    grasp_approach_forward_reserve_m: float = 0.15
    grasp_arc_enabled: bool = False
    grasp_arc_standoff_m: float = 0.25
    grasp_lat_gain: float = 1.0
    grasp_servo_enabled: bool = False
    grasp_servo_creep_k_fwd: float = 0.6
    grasp_servo_fwd_max_m: float = 0.40
    grasp_servo_commit_dist_m: float = 0.30
    grasp_servo_max_polls: int = 40
    grasp_servo_lost_hold: bool = True

    # Place side.
    place_approach_edge_m: float = 0.35
    place_approach_forward_reserve_m: float = 0.15
    place_square_tol_rad: float = 0.15
    place_max_turn_step_rad: float = 0.7
    place_edge_quality_max: float = 0.25
    place_edge_min_len_mm: float = 150.0
    place_servo_enabled: bool = False

    # Coarse (wide-FOV, bearing-only) sensor. ``head_*`` for continuity with the deployed
    # configs; they describe the coarse sensor's role, not a particular mounting point.
    head_hfov_rad: float = 1.2
    head_on_overlap_min: float = 0.15
    head_ground_verify: bool = True
    head_grounded_strict: bool = False

    @classmethod
    def from_cfg(cls, cfg: Any) -> ApproachTuning:
        """Read every field off ``cfg`` by name, falling back to this class's defaults."""
        if cfg is None:
            return cls()
        return cls(**{f.name: getattr(cfg, f.name, f.default) for f in fields(cls)})


# --------------------------------------------------------------------------------------
# Pure geometry — no robot, no state. ``tuning`` may be an ApproachTuning or any config
# object exposing the same field names.
# --------------------------------------------------------------------------------------


def forward_step(forward: float, tuning: Any) -> float:
    """Cap each fine-approach forward advance to ``approach_forward_step_m`` (~0.25 m) so ONE detection
    frame can never drive a large lunge: a distant false positive (``front_x`` wildly overshot) then
    advances only one small step, and the next iteration's re-detect — now closer — corrects it, instead
    of ramming a nearer real object. A non-positive ``forward`` (already in-band / bearing-only step)
    passes straight through.
    """
    step = float(getattr(tuning, "approach_forward_step_m", 0.25))
    return step if forward > step else forward


def grasp_forward_step(forward: float, tuning: Any) -> float:
    """Grasp-side STRAIGHT-IN advance (after the base has SQUARED to the target's face) using the reserve
    strategy — the symmetric twin of :func:`place_forward_step`: once facing the face-normal, drive
    all-but-a-reserve to the grasp standoff in ONE continuous move (so odom overshoot still stops
    ``reserve`` short), leaving a final ≤reserve gentle landing inside the decel ramp → ≤2 forward moves,
    not a per-``approach_forward_step_m`` chop (each cap step ends in a full base STOP + a re-detect).
    Safe here because it runs only AFTER squaring, driving straight at a repeatedly-detected target.
    """
    reserve = float(getattr(tuning, "grasp_approach_forward_reserve_m", 0.15))
    return forward if forward <= 2.0 * reserve else forward - reserve


def place_forward_step(forward: float, tuning: Any) -> float:
    """Place-side STRAIGHT-IN advance using the same reserve strategy: once squared to the surface edge,
    drive all-but-a-reserve to the near edge in ONE move (so odom overshoot still stops ``reserve`` short
    of the surface — no ram), leaving a final ≤reserve gentle landing → ≤2 forward moves.
    """
    reserve = float(getattr(tuning, "place_approach_forward_reserve_m", 0.15))
    return forward if forward <= 2.0 * reserve else forward - reserve


class Footprint(NamedTuple):
    """Ground-plane footprint: centre plus the oriented long/short extents.

    Bundled because the four values are always read off ONE detection dict together;
    ``from_detection`` is the single place that knows the field names and defaults.
    """

    center_mm: list[float]
    yaw_rad: float
    long_mm: float
    short_mm: float

    @classmethod
    def from_detection(cls, det: Any) -> "Footprint":
        """Read the footprint out of a detection/surface dict (callers gate on ``yaw_rad``)."""
        return cls(det["center_mm"], float(det.get("yaw_rad", 0.0)),
                   float(det.get("long_mm", 0.0)), float(det.get("short_mm", 0.0)))


def near_face_normal(
    fp: Footprint, tuning: Any, prev_normal: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    """Robot-facing outward normal of ONE of the object's vertical faces from its ground **footprint**
    yaw, for squaring the base to that face before a grasp or a place.

    The four candidate normals are the ±long/±short footprint axes. Which one we pick uses frame-to-frame
    HYSTERESIS so a jittery footprint yaw can't flip the face we square to:

    - **First pick** (``prev_normal is None``): the candidate whose outward normal is most aligned with the
      object→robot direction — the face already nearest to facing the robot (smallest square turn).
    - **Locked** (``prev_normal`` given): the candidate closest to ``prev_normal`` (largest dot) — STAY on
      the same physical face. Footprint yaw jitter (±~15°) is far below the 90° face spacing, so the locked
      face never 90°-flips even for a near-square footprint or a ~corner-on view. That is why this does not
      bail to a radial fallback on low aspect / a corner-on tie: for a near-cube payload any face is an
      acceptable grasp face, so once we commit to a face and stop re-choosing, the ill-conditioned-yaw
      oscillation is gone. NOTE the caller must rotate ``prev_normal`` into the current base frame after
      each base turn (base yaws +turn ⇒ world-fixed directions rotate −turn in the base frame) — see
      :func:`approach_for_grasp`.

    ``tuning`` is accepted for call-site symmetry with the selectors below but is unused. Returns the unit
    outward normal ``(nx, ny)`` pointing toward the robot, or ``None`` only when the footprint is genuinely
    degenerate (zero range or a zero span).
    """
    center_mm, yaw_rad, long_mm, short_mm = fp
    cx = float(center_mm[0]) / 1000.0
    cy = float(center_mm[1]) / 1000.0
    rng = math.hypot(cx, cy)
    if rng < 1e-6 or min(long_mm, short_mm) <= 1e-6:
        return None
    u = (math.cos(yaw_rad), math.sin(yaw_rad))          # long axis
    v = (-math.sin(yaw_rad), math.cos(yaw_rad))         # short axis
    cands = ((u[0], u[1]), (-u[0], -u[1]), (v[0], v[1]), (-v[0], -v[1]))
    if prev_normal is None:
        gx, gy = -cx / rng, -cy / rng                   # unit centre → robot; face nearest to facing us
        return max(cands, key=lambda n: n[0] * gx + n[1] * gy)
    px, py = float(prev_normal[0]), float(prev_normal[1])
    return max(cands, key=lambda n: n[0] * px + n[1] * py)   # hysteresis: stay on the locked face


def select_grasp_normal(
    det: dict, tuning: Any, prev_normal: tuple[float, float] | None = None
) -> tuple[float, float] | None:
    """Pick the target face normal for route planning: prefer the point-cloud ``face_normal`` when it is a
    trusted flat front face (``|n|>0.5`` and ``face_flatness ≤ grasp_face_flatness_max`` — the same
    predicate the detection uses), else the hysteresis-locked footprint normal (:func:`near_face_normal`);
    ``None`` → caller uses the radial fallback. The result is a UNIT normal, sign-fixed to point toward the
    robot (the point-cloud eigenvector sign is arbitrary; the footprint normal is already robot-facing so
    the flip is a no-op there).
    """
    n: tuple[float, float] | None = None
    fn = det.get("face_normal")
    if fn and math.hypot(float(fn[0]), float(fn[1])) > 0.5 \
            and float(det.get("face_flatness", 1.0)) <= float(getattr(tuning, "grasp_face_flatness_max", 0.15)):
        n = (float(fn[0]), float(fn[1]))
    elif "yaw_rad" in det:
        n = near_face_normal(Footprint.from_detection(det), tuning, prev_normal=prev_normal)
    if n is None:
        return None
    cx, cy = float(det["center_mm"][0]), float(det["center_mm"][1])   # mm; only the sign of the dot matters
    if n[0] * (-cx) + n[1] * (-cy) < 0.0:                             # flip toward the robot (viewer)
        n = (-n[0], -n[1])
    return n


def select_surface_square_normal(
    s: dict, tuning: Any, prev_normal: tuple[float, float] | None = None
) -> tuple[float, float] | None:
    """Squaring normal (unit, toward the robot) for :func:`approach_for_place`'s phase 2. Prefer the near-edge
    line fit (``near_edge_line`` — directly the surface's front-lip normal, robust on a partially-seen large
    surface) when it is trusted (``|n|>0.5``, ``edge_quality ≤ place_edge_quality_max``,
    ``edge_len_mm ≥ place_edge_min_len_mm``); else fall back to the whole-footprint principal-axis normal
    (:func:`near_face_normal`, hysteresis-locked). ``None`` only when neither is available (degenerate
    footprint) → nothing to square to. Mirrors :func:`select_grasp_normal` on the grasp side.
    """
    en = s.get("edge_normal")
    n: tuple[float, float] | None = None
    edge_trustworthy = (
        float(s.get("edge_quality", 1.0)) <= float(getattr(tuning, "place_edge_quality_max", 0.25))
        and float(s.get("edge_len_mm", 0.0)) >= float(getattr(tuning, "place_edge_min_len_mm", 150.0))
    )
    if en and math.hypot(float(en[0]), float(en[1])) > 0.5 and edge_trustworthy:
        n = (float(en[0]), float(en[1]))
    elif "yaw_rad" in s:
        n = near_face_normal(Footprint.from_detection(s), tuning, prev_normal=prev_normal)
    if n is None:
        return None
    cx, cy = float(s["center_mm"][0]), float(s["center_mm"][1])   # sign-fix toward the robot (defensive)
    if n[0] * (-cx) + n[1] * (-cy) < 0.0:
        n = (-n[0], -n[1])
    return n


# --------------------------------------------------------------------------------------
# Base motion primitives — methods on the mixin so a test/adapter can substitute them.
# --------------------------------------------------------------------------------------


def drive_base(api: Any, forward: float, turn: float, *, invalidate: Any,
               k_rot: float | None = None,
               k_rot_slow_rad: float | None = None,
               k_fwd: float | None = None) -> dict:
    """One-shot base approach shared by :func:`approach_for_grasp` / :func:`approach_for_place`: advance
    ``forward`` m (+``turn`` rad) when past the position tolerance, else just rotate to centre.
    ``k_rot``/``k_rot_slow_rad``/``k_fwd`` (when given) override the global base steering gains with
    gentler approach-only values so the fine-positioning steps don't overshoot and fling the target out of
    the precise camera's edge (search big-turns keep the fast global gains). On success call ``invalidate``
    to clear the now-stale cached detection/surface (base moved → re-sense before grasp/place).
    """
    if forward > float(api.approach_tuning().base_pos_tol_m):
        res = api.nav_relative(forward, 0.0, turn,
                                k_rot=k_rot, k_rot_slow_rad=k_rot_slow_rad, k_fwd=k_fwd)
    else:
        # In-place turn (rotate_base equivalent, inlined to thread the gentle gains without widening the
        # rotate_base tool schema); k_fwd is irrelevant with no forward component.
        logger.info("[approach] rotate_base dyaw=%.3f", turn)
        res = api.nav_relative(0.0, 0.0, turn, k_rot=k_rot, k_rot_slow_rad=k_rot_slow_rad)
    if res.get("ok"):
        invalidate()
        return {"ok": True, "turn_rad": turn, "forward_m": forward, "move": res}
    return {"ok": False, "reason": res.get("reason", "nav_failed"),
            "turn_rad": turn, "forward_m": forward, "move": res}


def redetect(api: Any, obj: str, reference: str | None, relation: str = "on") -> dict:
    """Post-move grounded re-detect that degrades to a plain detection while still far.

    A grounded ``locate_for_grasp(obj, reference=…, relation=…)`` only resolves the relation up close;
    while still far it returns not-ok. So when a grounded re-detect misses, retry a PLAIN detection and, if
    it resolves, carry the reference forward (the next, closer iteration re-attempts the grounding).
    Returns the detection dict — possibly not-ok if the target is truly lost (caller aborts). When
    ``reference`` is falsy this is a plain re-detect with no fallback.
    """
    det = api.locate_for_grasp(obj, reference=reference, relation=relation)
    if not det.get("ok") and reference:
        plain = api.locate_for_grasp(obj)
        if plain.get("ok"):
            plain["reference"], plain["relation"] = reference, relation
            det = plain
    return det


# --------------------------------------------------------------------------------------
# Coarse (bearing-only) search over the wide-FOV sensor.
# --------------------------------------------------------------------------------------


def coarse_bearing(api: Any, rgb: Any, seg_fn: Any, object_name: str, tuning: Any,
                   *, label: str = "camera", note: str | None = None) -> dict:
    """Top-1 coarse-sensor detection → in-image bearing only (no depth). Shared by the plain coarse search
    and the grounded-search DEGRADE fallback. ``rgb`` must be 3-channel; ``note`` (when set) is attached so
    the caller can see a degrade happened.
    """
    from jiuwensymbiosis.perception.vision import _run_detect_pick_best

    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    best = _run_detect_pick_best(rgb, seg_fn, object_name, 0.05, "[approach-search]")
    api.viz_update(label, object_name, rgb, best)
    if best.get("ok") is False:
        out = {"ok": True, "found": False, "reason": best.get("reason", "no_detection"),
               "camera": label, "object": object_name, "image_w": w, "image_h": h}
        if note:
            out["note"] = note
        return out

    box = [float(b) for b in best["box"][:4]]
    u = 0.5 * (box[0] + box[2])
    v = 0.5 * (box[1] + box[3])
    u_error_frac = (u - w / 2.0) / float(w)
    hfov = float(getattr(tuning, "head_hfov_rad", 1.2))
    bearing_rad = -u_error_frac * hfov
    logger.info("[approach] search_target %s: box=%s u_err=%.3f bearing=%.3f",
                object_name, box, u_error_frac, bearing_rad)
    out = {"ok": True, "found": True, "object": object_name, "camera": label,
           "bbox": box, "score": float(best["score"]),
           "u_center": u, "v_center": v, "image_w": w, "image_h": h,
           "u_error_frac": u_error_frac, "bearing_rad": bearing_rad}
    if note:
        out["note"] = note
    return out


def coarse_detect_on_reference_2d(api: Any, rgb: Any, seg_fn: Any, object_name: str, on: str,
                                  *, tuning: Any, label: str = "camera") -> dict | None:
    """Coarse-sensor 2-D grounding (NO depth/cloud/TF): detect the ``on`` reference and the ``object_name``
    target in the same image, then judge "target ON reference" purely by 2-D BBOX OVERLAP — the fraction of
    the target's bounding box that lies inside the reference's bounding box (``|t∩r| / |t|``, boxes not
    masks) must be ≥ ``head_on_overlap_min``. Bbox is used instead of pixel-mask overlap because a small
    object resting on top of a tall reference seen from the front barely touches the reference's front-face
    mask (mask overlap ≈0) even though its box sits fully within the reference's box; strict IoU
    (``|t∩r|/|t∪r|``) also under-fires here because the union is dominated by the large reference. This is a
    COARSE gate: it only steers the base toward the right candidate; the precise RGBD does the final
    on-surface grounding, so the threshold is deliberately lenient.

    Returns the winning target's bearing dict (``found=True, verified=True``); ``found=False,
    verified=True`` when the reference resolves but no target overlaps it enough (a genuine reject); or
    ``None`` — the DEGRADE signal — when the reference can't be detected at all, so the caller falls back
    to bearing-only (fail-open) / refuses under strict.
    """
    import numpy as np

    from jiuwensymbiosis.perception.vision import extract_color_word, region_color_matches

    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    hfov = float(getattr(tuning, "head_hfov_rad", 1.2))
    thr = float(getattr(tuning, "head_on_overlap_min", 0.15))

    def _bbox_overlap_over_target(tb: list, rb: list) -> float:
        """Fraction of the TARGET bbox area covered by the reference bbox (``|t∩r| / |t|`` on boxes).
        Not full IoU: I/U under-counts a small target sitting on a large surface.
        """
        ix1, iy1 = max(tb[0], rb[0]), max(tb[1], rb[1])
        ix2, iy2 = min(tb[2], rb[2]), min(tb[3], rb[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        ta = max(1e-6, (tb[2] - tb[0]) * (tb[3] - tb[1]))
        return inter / ta

    # Reference footprint = the best-scoring 2-D detection BOX of the reference noun.
    ref_box = None
    ref_score = -1.0
    for r in seg_fn(rgb, text_prompt=on):
        s = float(r.get("score", 0.0))
        if s < 0.05:
            continue
        if s > ref_score:
            ref_score = s
            ref_box = [float(b) for b in r["box"][:4]]
    if ref_box is None or (ref_box[2] - ref_box[0]) <= 0 or (ref_box[3] - ref_box[1]) <= 0:
        return None   # reference not resolvable in 2-D → degrade (fail-open)

    # Target candidates (colour-verified); keep those whose bbox overlaps the reference enough.
    cw = extract_color_word(object_name)
    picks = []   # (overlap, bearing, box, score, mask)
    seen_bearing = None   # best colour-verified target bearing, even if it fails the overlap gate
    seen_score = -1.0
    for r in seg_fn(rgb, text_prompt=object_name):
        if float(r.get("score", 0.0)) < 0.05:
            continue
        m = np.asarray(r["mask"]).astype(bool)
        if int(m.sum()) == 0:
            continue
        if cw and not region_color_matches(rgb, m, cw):   # wrong colour → skip
            continue
        box = [float(b) for b in r["box"][:4]]
        u = 0.5 * (box[0] + box[2])
        bearing = -((u - w / 2.0) / float(w)) * hfov
        sc = float(r.get("score", 0.0))
        if sc > seen_score:               # remember where the target IS, overlap or not, so the
            seen_score = sc               # caller can re-aim the sensor toward an off-reference target
            seen_bearing = bearing
        overlap = _bbox_overlap_over_target(box, ref_box)   # fraction of target box over reference
        if overlap < thr:
            continue
        picks.append((overlap, bearing, box, sc, m))
    if not picks:
        api.viz_update(label, f"{object_name} on {on}", rgb, None)
        logger.info("[approach] coarse 2-D %r on %r: reference found, no target overlaps it "
                    "(thr=%.2f seen_bearing=%s)", object_name, on, thr,
                    None if seen_bearing is None else round(seen_bearing, 3))
        out = {"ok": True, "found": False, "verified": True, "reason": "no_target_on_reference",
               "object": object_name, "reference": on, "camera": label,
               "image_w": w, "image_h": h}
        if seen_bearing is not None:
            out["seen_bearing_rad"] = seen_bearing   # sensor can re-aim here to recover overlap
        return out
    picks.sort(key=lambda p: p[0], reverse=True)   # most-overlapping target wins
    overlap, bearing, box, score, mask = picks[0]
    api.viz_update(label, f"{object_name} on {on}", rgb,
                    {"ok": True, "mask": mask, "box": box, "score": score})
    logger.info("[approach] coarse 2-D %r on %r: overlap=%.2f bearing=%.3f",
                object_name, on, overlap, bearing)
    return {"ok": True, "found": True, "verified": True, "object": object_name, "reference": on,
            "camera": label, "bbox": box, "score": score, "overlap": overlap,
            "bearing_rad": bearing, "image_w": w, "image_h": h}


def search_target(api: Any, object_name: str = "box", on: str | None = None,
                  camera: str | None = None) -> dict:
    """Detect ``object_name`` in one camera's raw image; return a bearing only.

    Deliberately 2-D: it reads the picture and reports WHERE TO TURN, so any camera serves —
    an RGBD one included, it simply ignores the depth it happens to have. Yields DIRECTION only:
    ``bearing_rad`` (target left of image centre → positive) for the approach loop to turn toward. With
    ``on`` set (and ``head_ground_verify``), the reference and target are both detected in the SAME 2-D
    image and the target is confirmed to rest on the reference purely by 2-D BBOX OVERLAP
    (``head_on_overlap_min``) before returning its bearing; if the reference can't be detected the frame
    degrades fail-open to the plain bearing (strict → not-found). Metric 3-D is still deferred to the
    precise ``locate_for_grasp``.
    """
    import numpy as np

    tuning = api.approach_tuning()
    label = camera if camera is not None else "camera"
    grounded = bool(on) and bool(getattr(tuning, "head_ground_verify", True))
    # Strict (fail-closed): when grounding is requested but the reference can't be resolved, refuse to
    # report a bearing "hit" so the caller does NOT advance on bearing alone.
    strict = grounded and bool(getattr(tuning, "head_grounded_strict", False))

    frames = api.search_frames(camera)
    if frames is None:
        return {"ok": False, "found": False, "reason": "no_camera", "camera": label, "object": object_name}
    rgb = frames[0]
    if rgb is None:
        return {"ok": False, "found": False, "reason": "no_image", "camera": label, "object": object_name}
    if rgb.ndim == 2:  # mono coarse frame → 3-channel for the detector
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    seg_fn = api.detector_seg_fn()
    if seg_fn is None:
        return {"ok": False, "found": False, "reason": "no_detector", "camera": label, "object": object_name}

    if grounded:
        res = coarse_detect_on_reference_2d(api, rgb, seg_fn, object_name, on,
                                            tuning=tuning, label=label)
        if res is not None:
            return res
        # Reference undetected this frame → the on-reference overlap could NOT be checked. Make it LOUD
        # (silently this looks like "found target → drove forward" with no on-check). strict → not-found
        # (no blind advance); else fail-open to the plain coarse bearing.
        logger.warning(
            "[approach] coarse 2-D grounded verify %r on %r DEGRADED%s: reference %r was not "
            "detected this frame; the on-reference overlap was NOT checked.",
            object_name, on, " (strict → not-found)" if strict else " to bearing-only", on)
        if strict:
            return {"ok": True, "found": False, "verified": False,
                    "reason": "head_reference_undetected_strict",
                    "note": "head_reference_undetected_degraded",
                    "camera": label, "object": object_name}
        return coarse_bearing(api, rgb, seg_fn, object_name, tuning, label=label,
                              note="head_reference_undetected_degraded")

    return coarse_bearing(api, rgb, seg_fn, object_name, tuning, label=label)


# --------------------------------------------------------------------------------------
# Search + face: coarse points, base approaches, precise sensor acquires.
# --------------------------------------------------------------------------------------


def face_by_sweep(api: Any, detect_fn: Any, object_name: str, *, result_key: str,
                  not_found_reason: str, head_name: str | None = None,
                  head_on: str | None = None, ground_ref: str | None = None) -> dict:
    """Face ``object_name`` by its PERCEIVED bearing, choosing what to do next by WHAT THE
    CAMERAS COULD ANSWER rather than by which camera was asked.

    Two answers are possible and they lead to different moves:

    * **A metric hit** — some camera's frame carried depth, so the target's position AND face
      normal are known. Square up to it directly (``_align``); no driving needed to learn more.
    * **Only a bearing** — every camera that saw it gave a direction and nothing else. Then the
      only useful move is to CLOSE IN along that direction (:func:`coarse_approach`) until a
      frame with depth acquires the target, and square up then.

    Which is a property of the FRAMES, not of the hardware: a body with one RGBD camera takes
    the first path, cruzr's head (no depth) takes the second and hands over to its waist RGBD,
    and a plain-RGB-only body takes the second until it is close enough for… nothing, and fails
    safe. ``detect_fn`` looks through every camera for a metric answer;
    ``api.sweep_for_bearing`` looks through every camera for a direction, additionally aiming
    any camera the body can aim (cruzr pans its head left + right once each).

    Search shape: try for a metric hit in the current view first (a near/front target aligns
    with no motion); on a miss sweep for a bearing over the CURRENT facing and hand off to the
    close-in. If that finds nothing either, the base rotates 180° ONCE and both passes repeat on
    the half behind us. Fails safe (``not_found_reason``) if neither facing resolves the target —
    the caller must NOT grasp/place blindly. A body that can aim nothing keeps the default hook,
    so its sweep is just "look with what I have", and it still gets the 180° re-scan.

    ``head_name`` is the noun the coarse sensor searches and ``head_on`` (optional) the reference it
    2-D-verifies the hit rests on. For a *grounded* grasp (``locate_for_grasp(target, on=surface)``) the
    coarse sensor searches the REAL target and verifies it on the reference surface
    (``head_name=target, head_on=surface``); for a *grounded* place it searches the reference object
    (which sits ON the surface, so its bearing ≈ the surface's) and verifies it on the surface
    (``head_name=reference, head_on=object_name``). ``head_on=None`` → plain bearing search, which is also
    what every relation other than ``on`` gets: 2-D bbox containment can only decide ``on``.
    The precise ``detect_fn`` still does the final grounding once close.

    ``detect_fn(object_name)`` must return ``center_mm`` (base frame) on ``ok`` and cache its own result for
    downstream consumers (the grasp/place tools).
    """
    head_name = head_name or object_name

    def _align(det: dict, turned: float) -> dict:
        # Centroid facing is deliberately NOT done here: on a differential base that radial turn fights the
        # following approach_for_grasp/approach_for_place face-normal squaring, so the approach owns ALL base
        # alignment (it re-detects from here). Surface the cached detection UNTURNED; the bearing is
        # reported for info only.
        cx, cy = float(det["center_mm"][0]), float(det["center_mm"][1])
        bearing = math.atan2(cy, cx)
        return {"ok": True, "status": "acquired", "bearing_rad": bearing,
                "turned_rad": turned, result_key: det}

    def _handoff(bearing: float, turned: float) -> dict:
        """Coarse-approach toward a coarse bearing until the precise sensor acquires, then align.
        Returns {} if the target never enters the precise view (caller decides next step).
        """
        got = coarse_approach(api, detect_fn, object_name, float(bearing))
        if got is None:
            return {}
        fresh = detect_fn(object_name)   # detect_fn cached its own result for downstream use
        chosen = fresh if fresh.get("ok") else got
        # Grounded handoff: the post-stop grounded re-detect can still miss at hand-off range — re-cache
        # the chosen (grounded) detection so the downstream approach reads it and finishes the grounding up
        # close. Route to the side's own cache: grasp → _last_detection, place → _last_surface.
        if ground_ref is not None:
            if result_key == "detection":
                api.last_detection = chosen
            else:
                api.last_surface = chosen
        return _align(chosen, turned)

    # 1) Current precise view first — a near/front target aligns with no motion.
    d = detect_fn(object_name)
    if d.get("ok"):
        return _align(d, 0.0)

    # 2) Look around from here. How far that reaches is the body's business: turning the
    #    whole body covers the circle in one pass, aiming a neck covers only its own range.
    acq = api.sweep_for_bearing(head_name, on=head_on)
    swept = float(acq.get("turned_rad", 0.0))
    if acq["found"]:
        # Found it — stop searching and go. If the look-around turned the BODY, the target is
        # in front of us now, so ask for a metric measurement right here: when it answers, the
        # approach loop can take over from a real position and the blind creep below is not
        # needed at all. A body that only aimed a camera (swept == 0) is still pointing
        # wherever it was, so there is nothing new to measure and this costs it nothing.
        if swept:
            measured = detect_fn(object_name)
            if measured.get("ok"):
                return _align(measured, swept)
        res = _handoff(acq["total_bearing"], swept)
        if res:
            return res
        # Drove toward the bearing but no camera with depth ever acquired → don't blind-grasp.
        api.reset_search_sensor()
        logger.info("[approach] look-around for %r: seen, metric sensing never acquired", object_name)
        return {"ok": False, "reason": not_found_reason, "turned_rad": swept,
                "note": "head_seen_no_waist_acquire"}

    # 3) Nothing found. A sweep that already covered the circle has nothing left to show us,
    #    so only a body whose look-around is limited to its own facing gets the 180° re-scan.
    if acq.get("exhaustive"):
        api.reset_search_sensor()
        logger.info("[approach] look-around for %r: whole circle covered, not found", object_name)
        return {"ok": False, "reason": not_found_reason, "turned_rad": swept, "note": "sweep_exhausted"}
    nav = api.rotate_base(math.pi)
    if not nav.get("ok"):
        api.reset_search_sensor()
        return {"ok": False, "reason": not_found_reason, "turned_rad": 0.0,
                "note": "rotate_base_failed", "nav_reason": nav.get("reason")}
    acq2 = api.sweep_for_bearing(head_name, on=head_on)
    if acq2["found"]:
        res = _handoff(acq2["total_bearing"], math.pi)
        if res:
            return res
        api.reset_search_sensor()
        logger.info("[approach] coarse scan for %r: back seen, precise never acquired", object_name)
        return {"ok": False, "reason": not_found_reason, "turned_rad": math.pi,
                "note": "head_seen_no_waist_acquire"}

    # 4) Neither facing resolved it → fail safe; caller must NOT grasp/place blind.
    api.reset_search_sensor()
    logger.info("[approach] coarse scan for %r: not found front+back (turned≈π)", object_name)
    return {"ok": False, "reason": not_found_reason, "turned_rad": math.pi,
            "note": "panscan_exhausted"}


def coarse_approach(api: Any, detect_fn: Any, object_name: str,
                    initial_bearing: float) -> dict | None:
    """Precise-sensor-only coarse approach: turn to ``initial_bearing`` ONCE (the direction the coarse
    search already found), then DRIVE FORWARD continuously while polling ONLY the precise ``detect_fn`` for
    the handoff — no per-poll coarse detection.

    The coarse search has already located the target and given the bearing; from here the base commits to
    that direction and the ONLY signal polled during the creep is the GROUNDED precise ``detect_fn``
    (a grounded ``locate_for_grasp`` grasp / ``locate_for_place`` place). Dropping the
    coarse re-detect roughly doubles the precise poll rate — the handoff signal — because each poll no
    longer also pays the coarse sensor's two extra detector inferences. At range the grounded precise test
    misses (a short target isn't resolvable ON its reference until ~0.7 m); it hands off only once it
    confirms, so the base keeps closing until it does. The worker's own distance/timeout caps are a
    GENEROUS safety backstop; after it self-stops (lidar standoff / cap), ONE final precise look is taken
    (the base may have closed the last stretch after the final in-loop poll).

    TRADEOFF — no steering: the one-shot initial turn is open-loop (wheel/odom, no ramp) and can
    under/over-shoot, so a target left off to one side is NOT steered back into view; the base drives
    straight on the initial bearing. Returns the triggering precise detection on acquisition (already
    cached by ``detect_fn`` for downstream consumers), or ``None`` if it never acquires (the worker
    finished its bounded run, or the initial turn failed). The final grasp/place-distance convergence stays
    with :func:`approach_for_grasp` / :func:`approach_for_place` — this only gets the target into view.
    """
    # 1) Turn to the coarse bearing the search found — ONCE (open-loop wheel/odom).
    if abs(initial_bearing) > 1e-3:
        nav = api.rotate_base(initial_bearing)
        if not nav.get("ok"):
            api.reset_search_sensor()
            return None

    # Re-centre the coarse sensor BEFORE the creep: it panned to find the target and the base has now
    # turned to face it, so a sensor left at the search yaw points off to the side. The creep polls only
    # the precise sensor, so centre it and drive with the coarse one looking straight ahead.
    api.reset_search_sensor()

    # 2) Drive forward CONTINUOUSLY on that bearing, polling ONLY the precise sensor (the grounded handoff
    #    trigger) so its poll rate isn't throttled by the coarse sensor's extra inferences.
    drv = api.base_driver()
    handle = drv.start_base_drive()
    polls = 0                                            # precise detect attempts (each ~a subprocess grab)
    got: dict | None = None
    while drv.base_drive_running(handle):
        polls += 1
        det = detect_fn(object_name)                     # base-frame coords yet? (grounded on-surface)
        if det.get("ok"):
            got = det
            break
        # No coarse re-detect/steer — the forward worker keeps driving straight on the initial bearing;
        # just keep polling until it acquires or the worker self-stops.
    res = drv.stop_base_drive(handle) or {}
    # The worker self-stopped (lidar standoff / generous safety cap); the base may have closed the last
    # stretch after the final in-loop poll → give the now-closer precise sensor ONE last look.
    if got is None:
        det = detect_fn(object_name)
        if det.get("ok"):
            got = det
    api.reset_search_sensor()
    logger.info("[approach] coarse-approach %r: %s after %d polls; drive reason=%s dist=%.2fm",
                object_name, "acquired" if got is not None else "MISS",
                polls, res.get("reason", "?"), float(res.get("dist_traveled", 0.0) or 0.0))
    return got


# --------------------------------------------------------------------------------------
# Grasp-side approach: square to a face, then straight in.
# --------------------------------------------------------------------------------------


def approach_for_grasp(api: Any, box: dict | None = None) -> dict:
    """Iteratively drive the base until it is SQUARE to the target's front face at the work distance,
    re-detecting + correcting each step.

    An optional SINGLE-SHOT approach (``grasp_arc_enabled``, OFF by default) runs first
    (:func:`approach_single_shot`): from ONE detection it drives the whole route to the work pose — a
    RIGHT-ANGLE (L-shaped) route onto the target's face-normal line — then takes ONE final re-detect + a
    single alignment, with NO per-step re-detect polling. When disabled, or when the face normal is
    untrustworthy, it falls back to the discrete polling loop described here.

    An end-effector that closes along the base Y axis IGNORES the target's yaw, so the base must face the
    target's near face perpendicular. On a differential base (no strafe) "square to the face" and "centred
    on the centroid" only coincide when the base sits on the face-normal line; chasing centring (radial)
    *after* squaring rotates the target back off-square. So we SQUARE FIRST: rotate in place (forward=0) to
    face ONE of the target's vertical faces, then advance STRAIGHT in (turn=0), which preserves the square.
    Which face is HYSTERESIS-LOCKED on the first footprint frame and only re-picked near that lock
    afterwards (frame-compensated for each base turn), so a jittery footprint yaw can't 90°-flip the face.
    Only a frame with no footprint (no ``yaw_rad``) falls back to the plain radial centre+advance. Both
    motions are in-place-turn or turn=0 straight-in — no turn+forward lunge — so neither can drive the base
    into the target. The square tolerance is ``grasp_square_tol_rad`` (must exceed the footprint-yaw jitter,
    else a jittery target only ever rotates and never advances). Leaves the converged fresh detection
    cached for the grasp tool (no separate re-detect).
    """
    tuning = api.approach_tuning()
    # box may arrive as a stringified bind name (the compiler can't pass a whole detection dict through the
    # scalar-only param resolver) — ignore non-dicts and use the cached preceding detection.
    det = box if isinstance(box, dict) else api.last_detection
    if not det or not det.get("center_mm"):
        return {"ok": False, "reason": "no_detection"}
    obj = det.get("object", "box")
    # Keep the grounding sticky: if the target was found via a grounded locate_for_grasp, every post-move
    # re-detect must stay grounded on that same reference AND relation, else it can lock onto a different
    # same-class object mid-approach.
    on = det.get("reference")
    rel = det.get("relation", "on")

    # Single-shot approach (opt-in; OFF by default → the discrete polling loop below). Rather than POLL,
    # compute the whole RIGHT-ANGLE route from ONE detection, drive it open-loop, then take ONE final
    # re-detect + a single alignment. Returns None to DECLINE (untrustworthy normal) → discrete loop.
    if tuning.grasp_arc_enabled:
        # Grounded 'X on Y': the single-shot commits to a whole OPEN-LOOP route from ONE detection, so
        # CONFIRM the target-on-reference relation with a strict grounded re-detect BEFORE planning — never
        # drive toward a target not verified on its reference (the face_object handoff can fall back to a
        # plain, ungrounded detection while far).
        if on:
            det = api.locate_for_grasp(obj, reference=on, relation=rel)
            if not det.get("ok"):
                return {"ok": False, "reason": det.get("reason", "not_confirmed_on_reference"),
                        "object": obj, "reference": on, "relation": rel}
        out = approach_single_shot(api, det, obj, on, rel)
        if out is not None:
            return out

    square_tol = float(tuning.grasp_square_tol_rad)
    pos_tol = float(tuning.base_pos_tol_m)
    max_iters = int(tuning.approach_converge_iters)
    ak_rot = tuning.approach_k_rot
    ak_slow = tuning.approach_k_rot_slow_rad
    ak_fwd = tuning.approach_k_fwd
    # Face-normal hysteresis (base frame): locked on the first footprint frame so a jittery yaw can't
    # 90°-flip the squared face; re-picked near this lock and frame-compensated after each turn.
    lock: tuple[float, float] | None = None
    for i in range(1, max_iters + 1):
        turn, forward, status = plan_base_goal_for_grasp(det["center_mm"], tuning)   # radial goal
        if status == "too_close":
            return {"ok": False, "reason": "too_close", "turn_rad": turn, "forward_m": forward, "iters": i}
        # Footprint face normal with hysteresis. None only when the frame has no footprint yaw.
        n = (near_face_normal(Footprint.from_detection(det), tuning, prev_normal=lock)
             if "yaw_rad" in det else None)
        if n is None:
            # No footprint this frame → plain radial centre+advance; a held lock is kept + compensated.
            logger.info("[approach] approach_for_grasp iter %d/%d turn=%.3f forward=%.3f status=%s square=n/a",
                        i, max_iters, turn, forward, status)
            if status == "in_band":
                api.last_detection = det
                return {"ok": True, "status": "in_band", "turn_rad": turn, "forward_m": forward, "iters": i}
            # Split the forward advance so the base closes in ≤2 moves while the final one lands gently in
            # the decel ramp; non-positive/bearing-only steps pass through.
            step = forward_step(forward, tuning)
            cmd_turn = turn
            res = api.drive_base(step, turn, invalidate=lambda: None,
                                  k_rot=ak_rot, k_rot_slow_rad=ak_slow, k_fwd=ak_fwd)
        else:
            # Lock onto this face, square to it first, then advance straight in (turn=0) to preserve it.
            lock = n
            square_turn = math.atan2(-n[1], -n[0])
            logger.info("[approach] approach_for_grasp iter %d/%d status=%s forward=%.3f square=%.3f",
                        i, max_iters, status, forward, square_turn)
            if abs(square_turn) > square_tol:            # not facing the face → rotate IN PLACE (forward=0)
                cmd_turn = square_turn
                res = api.drive_base(0.0, square_turn, invalidate=lambda: None,
                                      k_rot=ak_rot, k_rot_slow_rad=ak_slow, k_fwd=ak_fwd)
            elif forward > pos_tol:                       # facing the face, still far → STRAIGHT in (turn=0)
                cmd_turn = 0.0
                # Squared + straight → drive the whole leg to the standoff in one continuous move
                # (reserve strategy, ≤2 moves), not the per-0.25 m stop-and-redetect stutter.
                step = grasp_forward_step(forward, tuning)
                res = api.drive_base(step, 0.0, invalidate=lambda: None,
                                      k_rot=ak_rot, k_rot_slow_rad=ak_slow, k_fwd=ak_fwd)
            else:                                         # square + at work distance → the grasp consumes it
                api.last_detection = det
                return {"ok": True, "status": "in_band", "turn_rad": turn, "forward_m": forward,
                        "square_rad": square_turn, "iters": i}
        if not res.get("ok"):
            return {**res, "iters": i}
        if lock is not None and cmd_turn:
            # Base yawed +cmd_turn ⇒ rotate the base-frame lock by −cmd_turn to keep it in the new frame.
            c, s = math.cos(-cmd_turn), math.sin(-cmd_turn)
            lock = (lock[0] * c - lock[1] * s, lock[0] * s + lock[1] * c)
        det = api.redetect(obj, on, rel)   # fresh, post-move; grounded (degrades to plain while far)
        if not det.get("ok"):
            return {"ok": False, "reason": det.get("reason", "lost_after_move"), "iters": i}
    # Out of iterations without landing square+in_band: keep the last fresh detection + report residual.
    api.last_detection = det
    turn, forward, _ = plan_base_goal_for_grasp(det["center_mm"], tuning)
    return {"ok": True, "status": "max_iters", "iters": max_iters, "turn_rad": turn, "forward_m": forward}


def approach_single_shot(api: Any, det: dict, obj: str, on: str | None, rel: str = "on") -> dict | None:
    """Compute-once base fine-approach (opt-in via ``grasp_arc_enabled``): from ONE precise detection, drive
    the whole route to the work pose — a RIGHT-ANGLE (L-shaped) route onto the target's face-normal line —
    then take ONE final re-detect + a single small alignment. NO per-step re-detect polling.

    The route (``plan_grasp_right_angle``) is two perpendicular straight legs joined by in-place turns, so a
    differential base needs no strafe and it works at close range where an arc has no curving room: (1) a
    LATERAL leg ⊥ to the normal that moves ONTO the line (consumes no forward distance-to-target), then (2)
    an APPROACH leg that turns to face the target (−n) and drives straight in to the standoff entry point.

    Returns the result dict, or ``None`` to DECLINE — when the face normal is untrustworthy — so
    :func:`approach_for_grasp` falls back to its discrete loop. A decline always happens BEFORE any motion, so
    the loop restarts from the original detection. The final cached ``_last_detection`` reflects the ALIGNED
    pose, which the grasp tool consumes directly (it does not re-detect).

    Safety is geometric: the lateral leg is ⊥ to the approach so it never drives toward the target; the
    approach leg stops at the standoff entry point; the final square is only a small residual on the line;
    and a ``too_close`` guard after the re-detect aborts if the open-loop route still overshot.
    """
    tuning = api.approach_tuning()
    square_tol = float(tuning.grasp_square_tol_rad)
    yaw_tol = float(tuning.base_yaw_tol_rad)
    ak_rot = tuning.approach_k_rot
    ak_slow = tuning.approach_k_rot_slow_rad
    ak_fwd = tuning.approach_k_fwd

    def _drive(forward: float, turn: float) -> dict:
        return api.drive_base(forward, turn, invalidate=lambda: None,
                               k_rot=ak_rot, k_rot_slow_rad=ak_slow, k_fwd=ak_fwd)

    _t, _f, status = plan_base_goal_for_grasp(det["center_mm"], tuning)
    if status == "too_close":
        return {"ok": False, "reason": "too_close", "turn_rad": _t, "forward_m": _f, "iters": 0}
    n = select_grasp_normal(det, tuning)         # point-cloud face_normal preferred, else footprint
    if n is None:
        return None                              # untrustworthy → DECLINE (discrete loop closes it)

    # 1) Route: right-angle L onto the face-normal line, open-loop (no re-detect between the legs).
    plan = plan_grasp_right_angle(det["center_mm"], n, tuning)
    if plan["mode"] == "reject":
        return None
    heading = 0.0
    if plan["lat_dist"] > 0.0:                    # lateral leg — turn ⊥ to the normal, drive onto the line
        # Scale the geometric perpendicular offset by grasp_lat_gain (default 1.0): odom under-travel /
        # wheel slip / a y-biased centroid leave the real lateral move short of the line; a gain > 1
        # compensates. Overshoot is absorbed by the final re-detect + align below.
        lat_dist = plan["lat_dist"] * float(tuning.grasp_lat_gain)
        logger.info("[approach] approach_for_grasp SINGLE-SHOT L lateral turn=%.3f dist=%.3f (geom=%.3f)",
                    plan["lat_turn"], lat_dist, plan["lat_dist"])
        res = _drive(lat_dist, plan["lat_turn"])
        if not res.get("ok"):
            return {"ok": False, "reason": res.get("reason", "nav_failed"), "iters": 0, "move": res}
        heading = plan["lat_turn"]
    app_turn = math.atan2(math.sin(plan["app_turn"] - heading), math.cos(plan["app_turn"] - heading))
    # Servo terminal (opt-in) owns the ENTIRE forward approach as one continuous creep: turn to face the
    # target here (the forward worker can only steer, not do the ~90° face-turn) but DEFER the forward
    # drive so there is no discrete approach-leg stop before the servo — the near-distance "stop then go".
    servo_on = bool(tuning.grasp_servo_enabled)
    app_dist = 0.0 if servo_on else plan["app_dist"]
    if app_dist > 0.0 or abs(app_turn) > 1e-3:   # approach leg — face the target; servo owns the forward
        logger.info("[approach] approach_for_grasp SINGLE-SHOT L approach turn=%.3f dist=%.3f", app_turn, app_dist)
        res = _drive(app_dist, app_turn)
        if not res.get("ok"):
            return {"ok": False, "reason": res.get("reason", "nav_failed"), "iters": 0, "move": res}

    # Continuous visual-servo terminal (opt-in): the L-route above squared the base onto the face-normal
    # line and faced the target; hand off to a drive-while-detecting loop that drives the whole forward
    # approach with no stop-to-look freeze. Off → the two static re-detects below.
    if servo_on:
        return approach_continuous_servo(api, det, obj, on, rel)

    # 2) ONE final re-detect — now on the line and close, the most reliable view.
    det = api.redetect(obj, on, rel)
    if not det.get("ok"):
        return {"ok": False, "reason": det.get("reason", "lost_after_move"), "iters": 0}

    # 3) Final alignment from this frame: square to −n2 (residual — we are ON the line, so it does not
    #    throw the target off-centre) + straight-in to the work distance, in ONE move.
    turn2, fwd2, status2 = plan_base_goal_for_grasp(det["center_mm"], tuning)
    if status2 == "too_close":   # route overshot inside the band despite the standoff → don't grasp blind
        return {"ok": False, "reason": "too_close", "turn_rad": turn2, "forward_m": fwd2, "iters": 1}
    n2 = select_grasp_normal(det, tuning)
    square2 = math.atan2(-n2[1], -n2[0]) if n2 is not None else turn2
    if status2 == "in_band" and abs(square2) <= square_tol and abs(turn2) <= yaw_tol:
        api.last_detection = det                # already at the work pose → cache, no extra move
        return {"ok": True, "status": "in_band", "turn_rad": turn2, "forward_m": fwd2,
                "square_rad": square2, "iters": 1}
    logger.info("[approach] approach_for_grasp SINGLE-SHOT align square=%.3f forward=%.3f", square2, fwd2)
    res = _drive(forward_step(fwd2, tuning), square2)
    if not res.get("ok"):
        return {**res, "iters": 1}

    # 4) One more re-detect so the cached detection reflects the ALIGNED pose for the grasp tool.
    det = api.redetect(obj, on, rel)
    if not det.get("ok"):
        return {"ok": False, "reason": det.get("reason", "lost_after_move"), "iters": 1}
    api.last_detection = det
    turn3, forward3, _s = plan_base_goal_for_grasp(det["center_mm"], tuning)
    return {"ok": True, "status": "in_band", "turn_rad": turn3, "forward_m": forward3, "iters": 2}


def approach_continuous_servo(api: Any, det: dict, obj: str, on: str | None, rel: str = "on") -> dict:
    """Continuous visual-servo terminal (opt-in via ``grasp_servo_enabled``): drive the base to the work
    pose WHILE re-detecting, replacing :func:`approach_single_shot`'s freeze-inch-freeze tail (two static
    re-detects with the base hard-stopped). Reuses the steerable non-blocking forward worker
    (``start/steer/hold/stop_base_drive``); the MAIN thread polls the precise detector each loop while the
    worker keeps creeping, so there is no drive/detection thread and no lock. Steers toward the LATEST
    target direction, so a target moved during the final stretch is tracked. The L-route in
    :func:`approach_single_shot` already squared the base onto the face-normal line; this only closes the
    standoff.

    Safety is geometric: the work band stops the creep, the worker self-bounds at ``grasp_servo_fwd_max_m``,
    a target lost AFTER a lock HOLDs (no blind creep) until re-acquired or within
    ``grasp_servo_commit_dist_m`` (then the final short segment commits open-loop and MUST re-detect before
    grasping), and a ``too_close`` frame aborts — never grasp blind. Leaves the freshest detection in
    ``_last_detection``. Called only after :func:`approach_single_shot` has committed to approach (its
    DECLINE precedes any motion), so it always returns a result dict, never ``None``.
    """
    tuning = api.approach_tuning()
    yaw_tol = float(tuning.base_yaw_tol_rad)
    square_tol = float(tuning.grasp_square_tol_rad)
    commit = float(tuning.grasp_servo_commit_dist_m)
    max_polls = int(tuning.grasp_servo_max_polls)
    lost_hold = bool(tuning.grasp_servo_lost_hold)
    ak_rot = tuning.approach_k_rot
    ak_slow = tuning.approach_k_rot_slow_rad
    ak_fwd = tuning.approach_k_fwd

    drv = api.base_driver()
    # The servo owns the whole forward approach (the L route deferred it), so bound the worker's open-loop
    # forward to just reach the work point from here + a reserve margin, capped by the absolute ceiling
    # grasp_servo_fwd_max_m. This is the geometric backstop if detection dies mid-creep. The lateral leg
    # consumed no forward, so the seed's forward-to-work still holds after squaring.
    _t0, fwd0, _s0 = plan_base_goal_for_grasp(det["center_mm"], tuning)
    reserve = float(tuning.grasp_approach_forward_reserve_m)
    fwd_cap = min(max(0.0, fwd0) + reserve, float(tuning.grasp_servo_fwd_max_m))
    handle = drv.start_base_drive(k_fwd=float(tuning.grasp_servo_creep_k_fwd), fwd_max_m=fwd_cap)
    last_good: dict | None = None    # only ever a FRESH in-loop detection — never the stale seed
    polls = 0
    reached_band = False
    while drv.base_drive_running(handle) and polls < max_polls:
        polls += 1
        d = api.redetect(obj, on, rel)      # blocking grab+detect WHILE the base keeps creeping
        if d.get("ok"):
            turn, _fwd, status = plan_base_goal_for_grasp(d["center_mm"], tuning)
            if status == "too_close":   # crept inside the band from too close → never grasp blind
                drv.stop_base_drive(handle)
                return {"ok": False, "reason": "too_close", "iters": polls}
            last_good = d
            if status == "in_band" and abs(turn) <= yaw_tol:
                reached_band = True
                break                   # at the work pose + centred → one clean stop below
            drv.steer_base_drive(handle, turn)   # curve toward the LATEST target direction
        elif last_good is None:
            continue                    # not acquired yet → keep creeping straight (like coarse approach)
        else:
            _t, fwd_rem, _s = plan_base_goal_for_grasp(last_good["center_mm"], tuning)
            if not lost_hold or fwd_rem <= commit:
                break                   # had a lock, now lost within commit → commit final segment below
            drv.hold_base_drive(handle)  # had a lock, lost while still far → pause wheels, keep polling
    res = drv.stop_base_drive(handle) or {}
    if last_good is None:               # never acquired during the creep → one last look at the closest point
        d = api.redetect(obj, on, rel)
        if d.get("ok"):
            last_good = d
    logger.info("[approach] approach_for_grasp SERVO %s after %d polls; drive reason=%s dist=%.2fm",
                "in_band" if reached_band else "terminal", polls,
                res.get("reason", "?"), float(res.get("dist_traveled", 0.0) or 0.0))
    if last_good is None:
        return {"ok": False, "reason": "servo_never_acquired", "iters": polls}

    turn, forward, status = plan_base_goal_for_grasp(last_good["center_mm"], tuning)
    if status == "too_close":
        return {"ok": False, "reason": "too_close", "turn_rad": turn, "forward_m": forward, "iters": polls}
    n = select_grasp_normal(last_good, tuning)
    square = math.atan2(-n[1], -n[0]) if n is not None else turn
    if reached_band and abs(square) <= square_tol and abs(turn) <= yaw_tol:
        # last_good was detected mid-creep; the base then decelerated to rest, so its base-frame coords lag
        # the work pose. Refresh once at rest (base static) and abort on a miss — never act on a stale frame.
        fresh = api.redetect(obj, on, rel)
        if not fresh.get("ok"):
            return {"ok": False, "reason": fresh.get("reason", "lost_at_rest"), "iters": polls}
        api.last_detection = fresh
        t2, f2, _s = plan_base_goal_for_grasp(fresh["center_mm"], tuning)
        return {"ok": True, "status": "in_band", "turn_rad": t2, "forward_m": f2,
                "square_rad": square, "iters": polls}
    # Residual square and/or an un-closed final segment (commit / worker self-stop): ONE combined move,
    # then a refresh re-detect so the cache reflects the moved pose — abort rather than act on a stale one.
    logger.info("[approach] approach_for_grasp SERVO align square=%.3f forward=%.3f", square, forward)
    move = api.drive_base(grasp_forward_step(forward, tuning), square, invalidate=lambda: None,
                           k_rot=ak_rot, k_rot_slow_rad=ak_slow, k_fwd=ak_fwd)
    if not move.get("ok"):
        return {**move, "iters": polls}
    d = api.redetect(obj, on, rel)
    if not d.get("ok"):
        return {"ok": False, "reason": d.get("reason", "lost_after_move"), "iters": polls}
    api.last_detection = d
    turn2, forward2, _s = plan_base_goal_for_grasp(d["center_mm"], tuning)
    return {"ok": True, "status": "in_band", "turn_rad": turn2, "forward_m": forward2, "iters": polls}


# --------------------------------------------------------------------------------------
# Place-side approach: square to the near edge, then straight in.
# --------------------------------------------------------------------------------------


def approach_for_place(api: Any, object_name: str = "table", reference: str | None = None,
                     relation: str = "on") -> dict:
    """Drive the base so the support surface's near edge sits ``place_approach_edge_m`` ahead (within arm
    reach) AND squarely faced — but ONLY if it's out of range / off-square; no-op when already there
    (``status="in_range"``).

    SQUARE FIRST, then advance STRAIGHT in (turn=0) — the SAME structure as :func:`approach_for_grasp`. On a
    differential base (no strafe) "square to the edge" and "facing the centroid" only coincide when the base
    sits on the edge normal; chasing the centroid (radial) *after* squaring rotates the base back
    off-square. So each iteration: rotate in place to the near-edge normal (forward=0), then drive straight
    at it (turn=0), never re-centring by turning. The base is deliberately NOT driven laterally to the edge
    midpoint — that needs a ~90° sideways swing that flings a carried payload; the payload is instead
    centred ON the surface by the place tool. The near-edge normal is preferred from the near-edge line fit,
    else the hysteresis-locked footprint normal; only a frame with no footprint falls back to plain radial
    centre+advance. The straight-in advance uses the reserve strategy (:func:`place_forward_step`): ≤2
    moves. Invalidates the cached surface on move so the place tool re-senses.
    """
    tuning = api.approach_tuning()
    pos_tol = float(tuning.base_pos_tol_m)
    yaw_tol = float(tuning.base_yaw_tol_rad)
    square_tol = float(tuning.place_square_tol_rad)
    max_turn = float(tuning.place_max_turn_step_rad)   # per-step base-turn cap (fail-safe)
    max_iters = int(tuning.approach_converge_iters)
    def _invalidate() -> None:
        api.last_surface = None

    # Keep the grounding sticky: every post-move re-sense must stay grounded on the same reference object
    # AND relation, else it can lock onto a different same-class surface mid-approach. Prefer the caller's
    # arguments; fall back to what a prior grounded sense cached. Captured before the loop invalidates it.
    cached = api.last_surface or {}
    if not reference:
        reference, relation = cached.get("reference"), cached.get("relation", relation)
    ak_rot = tuning.approach_k_rot
    ak_slow = tuning.approach_k_rot_slow_rad
    ak_fwd = tuning.approach_k_fwd
    # Iterate (re-sense + correct) to converge — a single move's residual yaw/offset doesn't leave the base
    # off. Near-edge normal is HYSTERESIS-LOCKED across iterations (base frame) so a jittery/near-square
    # footprint can't 90°-flip the edge we square to.
    lock: tuple[float, float] | None = None
    for i in range(1, max_iters + 1):
        s = api.locate_for_place(object_name, reference=reference, relation=relation)
        if not s.get("ok") and reference:
            # A grounded relation resolves only up close; while still far, keep converging on a PLAIN
            # re-sense (carry the reference so the next, closer iteration re-attempts the grounding).
            plain = api.locate_for_place(object_name)
            if plain.get("ok"):
                plain["reference"], plain["relation"] = reference, relation
                s = plain
        if not s.get("ok"):
            return {"ok": False, "reason": s.get("reason", "no_surface"), "object": object_name, "iters": i}
        front_x = float(s["front_x_mm"]) / 1000.0
        cx, cy = float(s["center_mm"][0]) / 1000.0, float(s["center_mm"][1]) / 1000.0
        bearing = math.atan2(cy, cx)
        forward = front_x - float(tuning.place_approach_edge_m)  # >0: too far, must drive up
        # Near-edge outward normal (toward robot): prefer the near-edge line fit, else the footprint
        # principal-axis normal (hysteresis-locked). None only when the footprint is degenerate/absent
        # → nothing to square to, so treat as squared (no aspect gate — near-square surfaces square too).
        n = select_surface_square_normal(s, tuning, prev_normal=lock)
        square_turn = math.atan2(-n[1], -n[0]) if n is not None else 0.0
        logger.info("[approach] approach_for_place %s iter %d/%d front_x=%.3f forward=%.3f "
                    "bearing=%.3f square=%.3f%s", object_name, i, max_iters, front_x, forward,
                    bearing, square_turn, "" if n is not None else "(n/a)")
        if n is not None:
            lock = n                                          # commit to this face; hysteresis holds it
            if abs(square_turn) > square_tol:                 # not squared → rotate IN PLACE (forward=0)
                cmd_turn = max(-max_turn, min(max_turn, square_turn))   # cap per-step swing (no fling)
                res = api.drive_base(0.0, cmd_turn, invalidate=_invalidate,
                                      k_rot=ak_rot, k_rot_slow_rad=ak_slow, k_fwd=ak_fwd)
            elif forward > pos_tol:                           # squared, still far → STRAIGHT in (turn=0)
                if bool(tuning.place_servo_enabled):
                    # Base is squared to the near edge; hand the whole straight-in advance to a continuous
                    # creep so it never stops-to-look between legs. It drives STRAIGHT with NO steer —
                    # steering would rotate the squared base off the edge normal and fling the payload.
                    return approach_for_place_continuous_servo(api, object_name, reference, relation, tuning)
                cmd_turn = 0.0
                step = place_forward_step(forward, tuning)
                res = api.drive_base(max(step, 0.0), 0.0, invalidate=_invalidate,
                                      k_rot=ak_rot, k_rot_slow_rad=ak_slow, k_fwd=ak_fwd)
            else:                                             # squared + at the edge → reachable
                return {"ok": True, "status": "in_range", "front_x_m": front_x, "forward_m": forward,
                        "bearing_rad": bearing, "square_rad": square_turn, "iters": i}
        elif forward > pos_tol or abs(bearing) > yaw_tol:     # no footprint/normal → plain radial approach
            cmd_turn = max(-max_turn, min(max_turn, bearing))          # cap per-step swing (no fling)
            step = forward_step(forward, tuning)
            res = api.drive_base(max(step, 0.0), cmd_turn, invalidate=_invalidate,
                                  k_rot=ak_rot, k_rot_slow_rad=ak_slow, k_fwd=ak_fwd)
        else:                                                # no normal, positioned → reachable
            return {"ok": True, "status": "in_range", "front_x_m": front_x, "forward_m": forward,
                    "bearing_rad": bearing, "square_rad": square_turn, "iters": i}
        if not res.get("ok"):
            return {**res, "iters": i}
        if lock is not None and cmd_turn:
            # Base yawed +cmd_turn ⇒ rotate the base-frame lock by −cmd_turn to keep it in the new frame.
            c, s2 = math.cos(-cmd_turn), math.sin(-cmd_turn)
            lock = (lock[0] * c - lock[1] * s2, lock[0] * s2 + lock[1] * c)
    return {"ok": True, "status": "max_iters", "iters": max_iters,
            "front_x_m": front_x, "bearing_rad": bearing, "square_rad": square_turn}


def approach_for_place_continuous_servo(api: Any, object_name: str, reference: str | None,
                                     relation: str, tuning: Any) -> dict:
    """Continuous straight-in creep to the placement edge (opt-in via ``place_servo_enabled``), replacing
    :func:`approach_for_place`'s discrete freeze-inch-freeze straight-in tail. Called ONLY once the base is
    already SQUARED to the near edge, so it creeps STRAIGHT with NO turn: a real turn would rotate the
    squared base off the edge normal and fling the carried payload. The main thread polls the precise
    detector each loop while the non-blocking forward worker keeps creeping; it stops ONCE when the near
    edge reaches ``place_approach_edge_m``.

    Safety is geometric: the place band stops the creep, the worker self-bounds at
    ``fwd_cap = min(forward0+reserve, grasp_servo_fwd_max_m)`` (stops near the surface even if detection
    dies), and a surface lost within ``grasp_servo_commit_dist_m`` (or ``grasp_servo_lost_hold`` off)
    commits the remaining segment open-loop, while a far loss HOLDs the wheels until re-acquired. Since the
    creep is pure-straight, a held worker is resumed with ``steer_base_drive(0.0)`` — bearing 0 is "keep
    going straight", the worker's documented resume, NOT a payload-flinging turn. Leaves ``_last_surface``
    invalidated so the place tool's own re-sense provides the fresh landing surface.
    """
    pos_tol = float(tuning.base_pos_tol_m)
    edge_m = float(tuning.place_approach_edge_m)
    commit = float(tuning.grasp_servo_commit_dist_m)
    max_polls = int(tuning.grasp_servo_max_polls)
    lost_hold = bool(tuning.grasp_servo_lost_hold)
    reserve = float(tuning.grasp_approach_forward_reserve_m)
    ak_rot = tuning.approach_k_rot
    ak_slow = tuning.approach_k_rot_slow_rad
    ak_fwd = tuning.approach_k_fwd

    def _resense() -> dict:
        s = api.locate_for_place(object_name, reference=reference, relation=relation)
        if not s.get("ok") and reference:
            plain = api.locate_for_place(object_name)     # grounded resolves only up close → degrade
            if plain.get("ok"):
                plain["reference"], plain["relation"] = reference, relation
                s = plain
        return s

    s0 = _resense()
    if not s0.get("ok"):
        return {"ok": False, "reason": s0.get("reason", "no_surface"), "object": object_name, "iters": 0}
    forward0 = float(s0["front_x_mm"]) / 1000.0 - edge_m
    if forward0 <= pos_tol:                               # already at the edge → no creep
        return {"ok": True, "status": "in_range", "front_x_m": float(s0["front_x_mm"]) / 1000.0,
                "forward_m": forward0, "iters": 0}
    api.last_surface = None                              # about to move → the place tool must re-sense

    drv = api.base_driver()
    fwd_cap = min(max(0.0, forward0) + reserve, float(tuning.grasp_servo_fwd_max_m))
    handle = drv.start_base_drive(k_fwd=float(tuning.grasp_servo_creep_k_fwd), fwd_max_m=fwd_cap)
    last_good = s0
    polls = 0
    reached = False
    held = False
    while drv.base_drive_running(handle) and polls < max_polls:
        polls += 1
        s = _resense()                                    # blocking grab+detect WHILE the base creeps
        if s.get("ok"):
            last_good = s
            if float(s["front_x_mm"]) / 1000.0 - edge_m <= pos_tol:
                reached = True
                break                                     # near edge at the place gap → one clean stop
            if held:                                      # re-acquired after a pause → resume STRAIGHT
                drv.steer_base_drive(handle, 0.0)         # bearing 0 = keep going straight (not a turn)
                held = False
            # else: keep creeping straight; NO turn feed keeps the payload square to the edge
        else:
            forward_rem = float(last_good["front_x_mm"]) / 1000.0 - edge_m
            if not lost_hold or forward_rem <= commit:
                break                                     # lost within commit → finish open-loop below
            drv.hold_base_drive(handle)                   # lost while still far → pause wheels, keep polling
            held = True
    worker_alive = drv.base_drive_running(handle)         # True → we interrupted a live creep (commit/cap)
    res = drv.stop_base_drive(handle) or {}
    front_x = float(last_good["front_x_mm"]) / 1000.0
    forward = front_x - edge_m
    api.last_surface = None                              # moved → the place tool re-senses fresh
    logger.info("[approach] approach_for_place SERVO %s after %d polls; drive reason=%s dist=%.2fm front_x=%.3f",
                "in_range" if reached else "terminal", polls,
                res.get("reason", "?"), float(res.get("dist_traveled", 0.0) or 0.0), front_x)
    if not reached and worker_alive and forward > pos_tol:
        # We stopped a still-running creep short of the edge (surface lost, or poll cap) → finish the
        # remaining gap open-loop STRAIGHT (turn=0, no fling); the runner re-senses before placing. A
        # self-completed worker (worker_alive False = crept its full fwd_cap) is already at the edge.
        move = api.drive_base(forward, 0.0, invalidate=lambda: None,
                               k_rot=ak_rot, k_rot_slow_rad=ak_slow, k_fwd=ak_fwd)
        if not move.get("ok"):
            return {**move, "iters": polls}
    return {"ok": True, "status": "in_range", "front_x_m": front_x, "forward_m": forward, "iters": polls}
