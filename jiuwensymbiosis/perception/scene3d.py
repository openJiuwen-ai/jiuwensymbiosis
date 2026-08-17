# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""3-D scene sensing over one calibrated frame — robot-agnostic.

Everything here is a pure function of ``(rgb, depth, intrinsics, base←camera)``
plus a segmentation callable: no robot, no config object, no cached state. That
is what lets a second body reuse the whole grounded-detection pipeline by
supplying a frame and a detector, and it is what makes these functions testable
without hardware.

Two shapes of answer, because callers need different payloads from the same
geometry: an **object** answer (centre / extents / face normal — what a grasp
planner consumes) and a **surface** answer (footprint + near-edge line — what a
place planner consumes).

Both come in a plain and a *grounded* variant. Grounding resolves "the white box
**on** the brown table", "the box **beside** the hat" and their mirrors by sensing
both things from the same frame and keeping only the pair that satisfies the named
relation (:func:`relation_holds`, over the closed set :data:`SPATIAL_RELATIONS`).
One frame, not two: a second grab would be taken from a slightly different instant
and the relation could disagree with itself.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any, Literal, NamedTuple, Optional, TypedDict

logger = logging.getLogger(__name__)

# Detector scores below this are noise; used for every candidate sweep here.
DEFAULT_SCORE_MIN = 0.05


# ---------------------------------------------------------------------------
# Result shapes. Declared next to the functions that BUILD them, so the shape a
# planner is promised and the dict actually returned cannot drift apart. Fed to
# ``ActionSpec.result`` (api/actions.py), which turns them into the JSON Schema
# that ``parse_sequence`` checks every ``<bind>.field`` reference against.
# ---------------------------------------------------------------------------
class SensingFailure(TypedDict):
    """Shape every 3-D sensing action returns when it cannot answer."""

    ok: Literal[False]
    reason: str
    object: str


class ObjectGeometryResult(TypedDict, total=False):
    """Success shape of an object detection — see :func:`object_geometry_fields`.

    ``total=False`` because ``reference`` / ``relation`` are only present on the grounded
    path; every other field is always populated on success.
    """

    ok: Literal[True]
    reason: str
    object: str
    reference: str  # grounded path only: what the target was measured against
    relation: str  # grounded path only: which relation to the reference was confirmed
    center_mm: list[float]  # [x, y, z] base frame
    width_mm: float
    height_mm: float
    front_x_mm: float  # near face (smallest forward X)
    back_x_mm: float
    top_z_mm: float
    n_points: int
    yaw_rad: float  # footprint orientation
    long_mm: float
    short_mm: float
    face_normal: list[float]  # [nx, ny], outward from the near face
    face_flatness: float


class SurfaceGeometryResult(TypedDict, total=False):
    """Success shape of a support-surface sensing — see :func:`surface_footprint_fields`."""

    ok: Literal[True]
    object: str
    reference: str  # grounded path only: what the surface was measured against
    relation: str  # grounded path only: which relation to the reference was confirmed
    surface_z_mm: float  # the height a payload lands on
    center_mm: list[float]
    front_x_mm: float
    back_x_mm: float
    width_mm: float
    n_points: int
    yaw_rad: float
    long_mm: float
    short_mm: float
    face_normal: list[float]
    edge_midpoint_mm: list[float]  # near-edge line the base squares up to
    edge_normal: list[float]
    edge_quality: float
    edge_len_mm: float


class SceneScanResult(TypedDict):
    """Success shape of a whole-scene multi-instance scan."""

    ok: Literal[True]
    object: str
    count: int
    objects: list[dict]  # per-instance geometry, nearest first


def on_surface(
    px: float,
    py: float,
    pz: float,
    *,
    front_x: float,
    back_x: float,
    center_y: float,
    half_width: float,
    top_z: float,
    margin: float,
) -> bool:
    """True if point (px, py, pz) rests ON a surface: its (x, y) is inside the surface
    footprint (± margin) and its z is at/above the surface top. The 'on' relation shared
    by both directions — target-on-reference (grasp) and reference-on-surface (place)."""
    return (front_x - margin <= px <= back_x + margin
            and abs(py - center_y) <= half_width + margin
            and pz >= top_z - margin)


# ---------------------------------------------------------------------------
# Spatial relations — how a task phrase pins down WHICH instance is meant.
# ---------------------------------------------------------------------------
# Closed set, deliberately small and deliberately viewpoint-INDEPENDENT: "on" / "under" /
# "in" / "beside" / "near" mean the same thing from wherever the robot happens to stand, so
# a plan carrying one is still true after the base moves. "left of" / "in front of" are NOT
# here — they are only defined relative to an observer, and the observer moves.
SPATIAL_RELATIONS: tuple[str, ...] = ("on", "under", "in", "beside", "near")


class Extent(NamedTuple):
    """The axis-aligned base-frame box a spatial relation needs — nothing else."""

    center_mm: tuple[float, float, float]
    front_x_mm: float  # near face (smallest forward X)
    back_x_mm: float
    width_mm: float  # along Y, about center_mm[1]
    top_z_mm: float
    bottom_z_mm: float


def has_extent(record: Any) -> bool:
    """True if ``record`` carries the bounds a relation is judged on.

    ``extent_of`` defaults absent bounds to 0.0 — harmless for a full measurement, but on a
    partial record that silently parks the object at the base origin and the predicate then
    answers confidently about geometry nobody measured. Callers holding records of uncertain
    provenance must ask this first and report "cannot judge" rather than take the answer.
    """
    get = record.get if isinstance(record, dict) else lambda k, d=None: getattr(record, k, d)
    if get("center_mm") is None:
        return False
    if get("top_z_mm") is None and get("surface_z_mm") is None:
        return False
    return get("front_x_mm") is not None and get("back_x_mm") is not None


def extent_of(record: Any) -> Extent:
    """Read an :class:`Extent` off an ``ObjectGeometry3D`` or a result dict.

    Both sides of a relation may arrive in either shape (a target measured as an object,
    a reference sensed as a surface), so the predicate takes neither and this bridges. A
    surface reports its top as ``surface_z_mm`` and states no thickness, so its bottom is
    its top — a plane, which is what it is.
    """
    get = record.get if isinstance(record, dict) else lambda k, d=None: getattr(record, k, d)
    center = tuple(float(v) for v in (get("center_mm") or (0.0, 0.0, 0.0)))
    top = get("top_z_mm")
    if top is None:
        top = get("surface_z_mm", 0.0)
    height = get("height_mm")
    return Extent(
        center_mm=(center[0], center[1], center[2]),  # type: ignore[arg-type]
        front_x_mm=float(get("front_x_mm", 0.0)),
        back_x_mm=float(get("back_x_mm", 0.0)),
        width_mm=float(get("width_mm", 0.0)),
        top_z_mm=float(top),
        bottom_z_mm=float(top) - float(height or 0.0),
    )


def _footprint_gap_mm(a: Extent, b: Extent) -> float:
    """Shortest horizontal distance between two footprints; 0 when they overlap in XY."""
    dx = max(0.0, max(a.front_x_mm, b.front_x_mm) - min(a.back_x_mm, b.back_x_mm))
    dy = max(0.0, abs(a.center_mm[1] - b.center_mm[1]) - (a.width_mm + b.width_mm) / 2.0)
    return math.hypot(dx, dy)


def _vertical_gap_mm(a: Extent, b: Extent) -> float:
    """Shortest vertical distance between two z-extents; 0 when they overlap."""
    return max(0.0, max(a.bottom_z_mm, b.bottom_z_mm) - min(a.top_z_mm, b.top_z_mm))


def _rests_on(upper: Extent, lower: Extent, margin: float) -> bool:
    return on_surface(
        upper.center_mm[0], upper.center_mm[1], upper.center_mm[2],
        front_x=lower.front_x_mm, back_x=lower.back_x_mm, center_y=lower.center_mm[1],
        half_width=lower.width_mm / 2.0, top_z=lower.top_z_mm, margin=margin,
    )


def _one_over_the_other(a: Extent, b: Extent) -> bool:
    """True if either centre lies within the other's footprint — i.e. they are stacked, not
    side by side. Deliberately margin-free: the ``on`` margin exists to absorb depth noise
    when deciding "is this resting on that", and inflating a *lateral* test by it would call
    two neighbours a hand's width apart a stack."""
    def over(p: Extent, q: Extent) -> bool:
        return (q.front_x_mm <= p.center_mm[0] <= q.back_x_mm
                and abs(p.center_mm[1] - q.center_mm[1]) <= q.width_mm / 2.0)

    return over(a, b) or over(b, a)


def _inside(inner: Extent, outer: Extent, margin: float) -> bool:
    """True if ``inner`` sits within ``outer``'s extent on all three axes.

    Containment, not support: the thing is in the box rather than on top of it. Judged on
    the whole extent (not just the centre, as ``on`` is) because half an apple hanging out
    of a drawer is not in the drawer.
    """
    # Horizontal axes get the noise margin; the TOP does not. Allowing slack above the rim
    # is what lets something resting ON the lid read as inside it — and a grasp planned on
    # that answer descends onto a closed drawer.
    return (outer.front_x_mm - margin <= inner.front_x_mm
            and inner.back_x_mm <= outer.back_x_mm + margin
            and abs(inner.center_mm[1] - outer.center_mm[1]) + inner.width_mm / 2.0
            <= outer.width_mm / 2.0 + margin
            and inner.top_z_mm <= outer.top_z_mm
            and inner.bottom_z_mm >= outer.bottom_z_mm - margin)


def relation_holds(
    target: Any,
    reference: Any,
    relation: str,
    *,
    margin_mm: float = 80.0,
    beside_max_gap_mm: float = 400.0,
    near_max_dist_mm: float = 1000.0,
) -> bool:
    """True if ``target <relation> reference`` holds, judged on measured base-frame geometry.

    One direction only — the phrase is always read as *target relates to reference* — so
    "the white box ON the brown table" and "the table UNDER the cup" are the same predicate
    with the arguments swapped, rather than two hand-written groundings that can drift.

    ``in`` is containment and is checked FIRST, because it excludes ``on`` / ``under``: an
    apple down inside a drawer is not on the drawer, and treating it as if it were would
    aim the grasp at the closed lid.

    ``beside`` additionally demands the two z-extents overlap (within ``margin_mm``): a box
    on the shelf ABOVE the hat is not beside the hat, however close its footprint is.
    ``near`` drops that and asks only for horizontal proximity — it is the loose one, for
    when the task says "by" / "around" rather than a definite arrangement.
    """
    if relation not in SPATIAL_RELATIONS:
        raise ValueError(f"unknown spatial relation {relation!r}; known: {list(SPATIAL_RELATIONS)}")
    t, r = extent_of(target), extent_of(reference)
    if relation == "in":
        return _inside(t, r, margin_mm)
    # "on" and "in" must not both hold: something down inside a drawer is not on the drawer,
    # and a plan that grasps it as if it were on top would descend onto the closed lid.
    if relation == "on":
        return _rests_on(t, r, margin_mm) and not _inside(t, r, margin_mm)
    if relation == "under":
        return _rests_on(r, t, margin_mm) and not _inside(t, r, margin_mm)
    if relation == "beside":
        if _one_over_the_other(t, r):
            return False  # stacked, not side by side
        return _footprint_gap_mm(t, r) <= beside_max_gap_mm and _vertical_gap_mm(t, r) <= margin_mm
    return math.hypot(t.center_mm[0] - r.center_mm[0], t.center_mm[1] - r.center_mm[1]) <= near_max_dist_mm


def object_geometry_fields(geo: Any) -> dict:
    """Base-frame geometry payload for a detected object (mm), shared by the plain and the
    grounded (``on=``) detect paths so they never drift. ``geo`` is an ``ObjectGeometry3D``."""
    return {
        "center_mm": list(geo.center_mm),
        "width_mm": geo.width_mm,
        "height_mm": geo.height_mm,
        "front_x_mm": geo.front_x_mm,
        "back_x_mm": geo.back_x_mm,
        "top_z_mm": geo.top_z_mm,
        "n_points": geo.n_points,
        "yaw_rad": geo.yaw_rad,
        "long_mm": geo.long_mm,
        "short_mm": geo.short_mm,
        "face_normal": [geo.face_normal_x, geo.face_normal_y],
        "face_flatness": geo.face_flatness,
    }


def surface_footprint_fields(surf: Any) -> dict:
    """Common base-frame footprint payload for a sensed support surface, shared by the plain and the
    grounded (``has=``) sense paths so they NEVER drift — the place-side squaring reads
    ``yaw_rad`` / ``edge_normal`` and a path that omits them silently disables squaring (the exact bug
    this centralises away). ``surf`` is an ``ObjectGeometry3D``."""
    return {
        "surface_z_mm": surf.top_z_mm,
        "center_mm": list(surf.center_mm),
        "front_x_mm": surf.front_x_mm, "back_x_mm": surf.back_x_mm,
        "width_mm": surf.width_mm, "n_points": surf.n_points,
        "yaw_rad": surf.yaw_rad, "long_mm": surf.long_mm, "short_mm": surf.short_mm,
        "face_normal": [surf.face_normal_x, surf.face_normal_y],
        "edge_midpoint_mm": [surf.edge_mid_x_mm, surf.edge_mid_y_mm],
        "edge_normal": [surf.edge_normal_x, surf.edge_normal_y],
        "edge_quality": surf.edge_quality, "edge_len_mm": surf.edge_len_mm,
    }


def edge_log_str(surf: Any) -> str:
    """Human-readable near-edge fit summary for the surface-sensing log — so the table-edge normal the
    base squares to can be CONFIRMED from the log. Empty when the fit is untrusted (zero normal)."""
    if math.hypot(surf.edge_normal_x, surf.edge_normal_y) <= 1e-6:
        return " edge=untrusted"
    return " edge_mid=(%.0f,%.0f) edgeN=%.0fdeg q=%.3f len=%.0fmm" % (
        surf.edge_mid_x_mm, surf.edge_mid_y_mm,
        math.degrees(math.atan2(surf.edge_normal_y, surf.edge_normal_x)),
        surf.edge_quality, surf.edge_len_mm)


def color_stats_str(rgb: Any, mask: Any) -> str:
    """Brightness / saturation / mean-RGB of a masked region for the color-mismatch reject LOG only,
    mirroring region_color_matches' math (same mask resize + mean/bright/sat) so a rejection can be
    diagnosed against its gate (e.g. 'white' ⇒ sat<0.25 & bright>0.35) instead of guessed by eye."""
    import numpy as np

    m = np.asarray(mask).astype(bool)
    rgb = np.asarray(rgb)
    if m.shape[:2] != rgb.shape[:2]:
        ys = (np.arange(rgb.shape[0]) * (m.shape[0] / rgb.shape[0])).astype(int).clip(0, m.shape[0] - 1)
        xs = (np.arange(rgb.shape[1]) * (m.shape[1] / rgb.shape[1])).astype(int).clip(0, m.shape[1] - 1)
        m = m[np.ix_(ys, xs)]
    px = rgb[m].astype(np.float64)
    if px.shape[0] == 0:
        return " [empty mask]"
    mean = px.mean(0)
    bright = float(mean.mean()) / 255.0
    sat = float(((px.max(1) - px.min(1)) / (px.max(1) + 1e-6)).mean())
    return " [bright=%.2f sat=%.2f rgb=(%d,%d,%d) n=%d]" % (
        bright, sat, int(mean[0]), int(mean[1]), int(mean[2]), px.shape[0])


def candidate_geometries(
    rgb: Any,
    depth_m: Any,
    intrinsics: Any,
    tf_base_cam: Any,
    *,
    seg_fn: Optional[Callable[..., list[dict]]],
    object_name: str,
    min_z_mm: Optional[float] = None,
    score_threshold: float = DEFAULT_SCORE_MIN,
) -> list[tuple[Any, dict]]:
    """All ``object_name`` detections in the frame → colour-verified base-frame geometries.

    ``min_z_mm`` (grounded 'X on Y'): the reference surface top — the target's face normal is then
    computed only from points ABOVE it, so a mask that bleeds onto the reference doesn't make the
    edge fit latch onto the REFERENCE's edge (giving a normal for the surface, not the target).

    Returns ``(geometry, detection)`` pairs; ``detection`` is the source ``{mask, box, score}``
    kept so the grounded callers can overlay the picked target's mask in the debug window."""
    import numpy as np

    from jiuwensymbiosis.perception.object_geometry import object_geometry_from_mask
    from jiuwensymbiosis.perception.vision import extract_color_word, region_color_matches

    if seg_fn is None:
        return []
    cw = extract_color_word(object_name)
    geos = []
    for r in seg_fn(rgb, text_prompt=object_name):
        if r.get("score", 0.0) < score_threshold:
            continue
        m = np.asarray(r["mask"])
        if cw and not region_color_matches(rgb, m, cw):  # wrong colour → skip
            continue
        g = object_geometry_from_mask(m, depth_m, np.asarray(intrinsics), np.asarray(tf_base_cam),
                                      min_z_mm=min_z_mm, debug_label=object_name)
        if g.ok:
            geos.append((g, {"mask": m, "box": r.get("box"), "score": r.get("score")}))
    return geos


def _best_geometry(
    rgb: Any,
    depth_m: Any,
    intrinsics: Any,
    tf_base_cam: Any,
    *,
    seg_fn: Optional[Callable[..., list[dict]]],
    object_name: str,
    score_threshold: float,
    log_prefix: str,
    debug_label: str,
    on_pick: Optional[Callable[[Optional[dict]], None]],
) -> tuple[Any, Optional[dict]]:
    """Highest-scoring colour-verified detection → base-frame geometry.

    Returns ``(geometry, None)`` or ``(None, failure_dict)``. ``on_pick`` receives the raw
    detector result (or the failure dict) so a caller can mirror it into a debug window.
    """
    import numpy as np

    from jiuwensymbiosis.perception.object_geometry import object_geometry_from_mask
    from jiuwensymbiosis.perception.vision import (
        _run_detect_pick_best, extract_color_word, region_color_matches)

    if seg_fn is None:
        return None, {"ok": False, "reason": "no_detector", "object": object_name}
    best = _run_detect_pick_best(rgb, seg_fn, object_name, score_threshold, log_prefix)
    if on_pick is not None:
        on_pick(best)
    if best.get("ok") is False:
        return None, best
    # Colour-verify: reject a detection whose pixels contradict the prompt colour (e.g. "white box"
    # grounded on a brown box) so a fresh sense fails cleanly instead of handing on a wrong target.
    cw = extract_color_word(object_name)
    if cw and not region_color_matches(rgb, np.asarray(best["mask"]), cw):
        logger.info("%s %r: color mismatch (want %s) → reject%s",
                    log_prefix, object_name, cw, color_stats_str(rgb, best["mask"]))
        return None, {"ok": False, "reason": "color_mismatch", "object": object_name}
    geo = object_geometry_from_mask(
        np.asarray(best["mask"]), depth_m,
        np.asarray(intrinsics), np.asarray(tf_base_cam), debug_label=debug_label,
    )
    return geo, None


def detect_object_geometry(
    rgb: Any,
    depth_m: Any,
    intrinsics: Any,
    tf_base_cam: Any,
    *,
    seg_fn: Optional[Callable[..., list[dict]]],
    object_name: str,
    score_threshold: float = DEFAULT_SCORE_MIN,
    log_prefix: str = "[scene3d-object]",
    on_pick: Optional[Callable[[Optional[dict]], None]] = None,
) -> dict:
    """Detect ``object_name`` and return its base-frame 3-D geometry (mm), or a failure dict."""
    geo, fail = _best_geometry(
        rgb, depth_m, intrinsics, tf_base_cam, seg_fn=seg_fn, object_name=object_name,
        score_threshold=score_threshold, log_prefix=log_prefix,
        debug_label=object_name, on_pick=on_pick,
    )
    if fail is not None:
        return fail
    return {"ok": geo.ok, "reason": geo.reason, "object": object_name, **object_geometry_fields(geo)}


def sense_surface_geometry(
    rgb: Any,
    depth_m: Any,
    intrinsics: Any,
    tf_base_cam: Any,
    *,
    seg_fn: Optional[Callable[..., list[dict]]],
    object_name: str,
    score_threshold: float = DEFAULT_SCORE_MIN,
    log_prefix: str = "[scene3d-surface]",
    on_pick: Optional[Callable[[Optional[dict]], None]] = None,
) -> dict:
    """Detect a support surface and return its base-frame footprint + top height (mm).

    Full footprint, not just the height: a place planner uses near/far edge (front_x/back_x),
    side centre and width to land a payload fully ON the surface clear of the edges, and the
    near-edge midpoint+normal to square the base to that edge.
    """
    surf, fail = _best_geometry(
        rgb, depth_m, intrinsics, tf_base_cam, seg_fn=seg_fn, object_name=object_name,
        score_threshold=score_threshold, log_prefix=log_prefix,
        debug_label="surface_" + object_name, on_pick=on_pick,
    )
    if fail is not None:
        return fail
    if not surf.ok:
        return {"ok": False, "reason": surf.reason, "object": object_name}
    logger.info("%s %s: surface_z=%.1fmm x=[%.1f,%.1f] cy=%.1f w=%.1f n=%d%s",
                log_prefix, object_name, surf.top_z_mm, surf.front_x_mm, surf.back_x_mm,
                surf.center_mm[1], surf.width_mm, surf.n_points, edge_log_str(surf))
    return {"ok": True, "object": object_name, **surface_footprint_fields(surf)}


def log_grounded_pick(log_prefix: str, object_name: str, on: str, n_cand: int, n_picks: int,
                      geo: Any, *, face_flatness_max: float, square_min_aspect: float) -> None:
    """One line covering everything a grounded pick can go wrong in: how many candidates survived the
    'on' relation, the footprint the base will square to, and whether the point-cloud face normal is
    trusted. The face normal should point roughly BACK at the robot, so its angle and the
    target→robot bearing are logged together — a wildly-off normal is then obvious from the log."""
    lo, hi = min(geo.long_mm, geo.short_mm), max(geo.long_mm, geo.short_mm)
    trusted = math.hypot(geo.face_normal_x, geo.face_normal_y) > 0.5 and geo.face_flatness <= face_flatness_max
    face_ang = math.degrees(math.atan2(geo.face_normal_y, geo.face_normal_x))
    to_robot = math.degrees(math.atan2(-geo.center_mm[1], -geo.center_mm[0]))
    logger.info(
        "%s %r on %r: %d cand → %d on-surface, picked center=%s "
        "footprint yaw=%.0f° long=%.0f short=%.0f aspect=%.2f (square-up needs aspect≥%.2f) | "
        "face n=(%.2f,%.2f) angle=%.0f° (target→robot=%.0f°) flatness=%.2f trusted=%s (needs flatness≤%.2f)",
        log_prefix, object_name, on, n_cand, n_picks, [round(v, 1) for v in geo.center_mm],
        math.degrees(geo.yaw_rad), geo.long_mm, geo.short_mm,
        (hi / lo if lo > 1e-6 else 0.0), square_min_aspect,
        geo.face_normal_x, geo.face_normal_y, face_ang, to_robot,
        geo.face_flatness, trusted, face_flatness_max)
