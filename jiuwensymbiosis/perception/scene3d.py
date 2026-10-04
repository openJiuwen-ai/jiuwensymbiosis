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
import time
import uuid
from collections.abc import Callable
from functools import partial, wraps
from typing import Any, NamedTuple, cast

from jiuwensymbiosis.contracts import SPATIAL_RELATIONS
from jiuwensymbiosis.errors import InferenceServiceError
from jiuwensymbiosis.perception.detector_client import DetectorClient, segment_image
from jiuwensymbiosis.perception.frame import CameraFrame

logger = logging.getLogger(__name__)

# Detector scores below this are noise; used for every candidate sweep here.
DEFAULT_SCORE_MIN = 0.05


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
    by both directions — target-on-reference (grasp) and reference-on-surface (place).
    """
    return (
        front_x - margin <= px <= back_x + margin and abs(py - center_y) <= half_width + margin and pz >= top_z - margin
    )


# ---------------------------------------------------------------------------
# Spatial relations — how a task phrase pins down WHICH instance is meant. The closed
# set itself is ``contracts.py:SPATIAL_RELATIONS``, because an ActionSpec advertises it
# as an enum; what lives here is the geometry that decides whether one holds.
# ---------------------------------------------------------------------------
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
        upper.center_mm[0],
        upper.center_mm[1],
        upper.center_mm[2],
        front_x=lower.front_x_mm,
        back_x=lower.back_x_mm,
        center_y=lower.center_mm[1],
        half_width=lower.width_mm / 2.0,
        top_z=lower.top_z_mm,
        margin=margin,
    )


def _one_over_the_other(a: Extent, b: Extent) -> bool:
    """True if either centre lies within the other's footprint — i.e. they are stacked, not
    side by side. Deliberately margin-free: the ``on`` margin exists to absorb depth noise
    when deciding "is this resting on that", and inflating a *lateral* test by it would call
    two neighbours a hand's width apart a stack.
    """

    def over(p: Extent, q: Extent) -> bool:
        return (
            q.front_x_mm <= p.center_mm[0] <= q.back_x_mm and abs(p.center_mm[1] - q.center_mm[1]) <= q.width_mm / 2.0
        )

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
    return (
        outer.front_x_mm - margin <= inner.front_x_mm
        and inner.back_x_mm <= outer.back_x_mm + margin
        and abs(inner.center_mm[1] - outer.center_mm[1]) + inner.width_mm / 2.0 <= outer.width_mm / 2.0 + margin
        and inner.top_z_mm <= outer.top_z_mm
        and inner.bottom_z_mm >= outer.bottom_z_mm - margin
    )


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
    grounded (``on=``) detect paths so they never drift. ``geo`` is an ``ObjectGeometry3D``.
    """
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
    this centralises away). ``surf`` is an ``ObjectGeometry3D``.
    """
    return {
        "surface_z_mm": surf.top_z_mm,
        "center_mm": list(surf.center_mm),
        "front_x_mm": surf.front_x_mm,
        "back_x_mm": surf.back_x_mm,
        "width_mm": surf.width_mm,
        "n_points": surf.n_points,
        "yaw_rad": surf.yaw_rad,
        "long_mm": surf.long_mm,
        "short_mm": surf.short_mm,
        "face_normal": [surf.face_normal_x, surf.face_normal_y],
        "edge_midpoint_mm": [surf.edge_mid_x_mm, surf.edge_mid_y_mm],
        "edge_normal": [surf.edge_normal_x, surf.edge_normal_y],
        "edge_quality": surf.edge_quality,
        "edge_len_mm": surf.edge_len_mm,
    }


def edge_log_str(surf: Any) -> str:
    """Human-readable near-edge fit summary for the surface-sensing log — so the table-edge normal the
    base squares to can be CONFIRMED from the log. Empty when the fit is untrusted (zero normal).
    """
    if math.hypot(surf.edge_normal_x, surf.edge_normal_y) <= 1e-6:
        return " edge=untrusted"
    angle = math.degrees(math.atan2(surf.edge_normal_y, surf.edge_normal_x))
    return (
        f" edge_mid=({surf.edge_mid_x_mm:.0f},{surf.edge_mid_y_mm:.0f}) "
        f"edgeN={angle:.0f}deg q={surf.edge_quality:.3f} len={surf.edge_len_mm:.0f}mm"
    )


def color_stats_str(rgb: Any, mask: Any) -> str:
    """Brightness / saturation / mean-RGB of a masked region for the color-mismatch reject LOG only,
    mirroring region_color_matches' math (same mask resize + mean/bright/sat) so a rejection can be
    diagnosed against its gate (e.g. 'white' ⇒ sat<0.25 & bright>0.35) instead of guessed by eye.
    """
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
    return f" [bright={bright:.2f} sat={sat:.2f} rgb=({int(mean[0])},{int(mean[1])},{int(mean[2])}) n={px.shape[0]}]"


def candidate_geometries(
    rgb: Any,
    depth_m: Any,
    intrinsics: Any,
    tf_base_cam: Any,
    *,
    seg_fn: Callable[..., list[dict]] | None,
    object_name: str,
    min_z_mm: float | None = None,
    score_threshold: float = DEFAULT_SCORE_MIN,
) -> list[tuple[Any, dict]]:
    """All ``object_name`` detections in the frame → colour-verified base-frame geometries.

    ``min_z_mm`` (grounded 'X on Y'): the reference surface top — the target's face normal is then
    computed only from points ABOVE it, so a mask that bleeds onto the reference doesn't make the
    edge fit latch onto the REFERENCE's edge (giving a normal for the surface, not the target).

    Returns ``(geometry, detection)`` pairs; ``detection`` is the source ``{mask, box, score}``
    kept so the grounded callers can overlay the picked target's mask in the debug window.
    """
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
        g = object_geometry_from_mask(
            m, depth_m, np.asarray(intrinsics), np.asarray(tf_base_cam), min_z_mm=min_z_mm, debug_label=object_name
        )
        if g.ok:
            geos.append((g, {"mask": m, "box": r.get("box"), "score": r.get("score")}))
    return geos


def _best_geometry(
    rgb: Any,
    depth_m: Any,
    intrinsics: Any,
    tf_base_cam: Any,
    *,
    seg_fn: Callable[..., list[dict]] | None,
    object_name: str,
    score_threshold: float,
    log_prefix: str,
    debug_label: str,
    on_pick: Callable[[dict | None], None] | None,
) -> tuple[Any, dict | None]:
    """Highest-scoring colour-verified detection → base-frame geometry.

    Returns ``(geometry, None)`` or ``(None, failure_dict)``. ``on_pick`` receives the raw
    detector result (or the failure dict) so a caller can mirror it into a debug window.
    """
    import numpy as np

    from jiuwensymbiosis.perception.object_geometry import object_geometry_from_mask
    from jiuwensymbiosis.perception.vision import _run_detect_pick_best, extract_color_word, region_color_matches

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
        logger.info(
            "%s %r: color mismatch (want %s) → reject%s",
            log_prefix,
            object_name,
            cw,
            color_stats_str(rgb, best["mask"]),
        )
        return None, {"ok": False, "reason": "color_mismatch", "object": object_name}
    geo = object_geometry_from_mask(
        np.asarray(best["mask"]),
        depth_m,
        np.asarray(intrinsics),
        np.asarray(tf_base_cam),
        debug_label=debug_label,
    )
    return geo, None


def detect_object_geometry(
    rgb: Any,
    depth_m: Any,
    intrinsics: Any,
    tf_base_cam: Any,
    *,
    seg_fn: Callable[..., list[dict]] | None,
    object_name: str,
    score_threshold: float = DEFAULT_SCORE_MIN,
    log_prefix: str = "[scene3d-object]",
    on_pick: Callable[[dict | None], None] | None = None,
) -> dict:
    """Detect ``object_name`` and return its base-frame 3-D geometry (mm), or a failure dict."""
    geo, fail = _best_geometry(
        rgb,
        depth_m,
        intrinsics,
        tf_base_cam,
        seg_fn=seg_fn,
        object_name=object_name,
        score_threshold=score_threshold,
        log_prefix=log_prefix,
        debug_label=object_name,
        on_pick=on_pick,
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
    seg_fn: Callable[..., list[dict]] | None,
    object_name: str,
    score_threshold: float = DEFAULT_SCORE_MIN,
    log_prefix: str = "[scene3d-surface]",
    on_pick: Callable[[dict | None], None] | None = None,
) -> dict:
    """Detect a support surface and return its base-frame footprint + top height (mm).

    Full footprint, not just the height: a place planner uses near/far edge (front_x/back_x),
    side centre and width to land a payload fully ON the surface clear of the edges, and the
    near-edge midpoint+normal to square the base to that edge.
    """
    surf, fail = _best_geometry(
        rgb,
        depth_m,
        intrinsics,
        tf_base_cam,
        seg_fn=seg_fn,
        object_name=object_name,
        score_threshold=score_threshold,
        log_prefix=log_prefix,
        debug_label="surface_" + object_name,
        on_pick=on_pick,
    )
    if fail is not None:
        return fail
    if not surf.ok:
        return {"ok": False, "reason": surf.reason, "object": object_name}
    logger.info(
        "%s %s: surface_z=%.1fmm x=[%.1f,%.1f] cy=%.1f w=%.1f n=%d%s",
        log_prefix,
        object_name,
        surf.top_z_mm,
        surf.front_x_mm,
        surf.back_x_mm,
        surf.center_mm[1],
        surf.width_mm,
        surf.n_points,
        edge_log_str(surf),
    )
    return {"ok": True, "object": object_name, **surface_footprint_fields(surf)}


def log_grounded_pick(
    log_prefix: str,
    object_name: str,
    on: str,
    *,
    n_cand: int,
    n_picks: int,
    geo: Any,
    face_flatness_max: float,
    square_min_aspect: float,
) -> None:
    """One line covering everything a grounded pick can go wrong in: how many candidates survived the
    'on' relation, the footprint the base will square to, and whether the point-cloud face normal is
    trusted. The face normal should point roughly BACK at the robot, so its angle and the
    target→robot bearing are logged together — a wildly-off normal is then obvious from the log.
    """
    lo, hi = min(geo.long_mm, geo.short_mm), max(geo.long_mm, geo.short_mm)
    trusted = math.hypot(geo.face_normal_x, geo.face_normal_y) > 0.5 and geo.face_flatness <= face_flatness_max
    face_ang = math.degrees(math.atan2(geo.face_normal_y, geo.face_normal_x))
    to_robot = math.degrees(math.atan2(-geo.center_mm[1], -geo.center_mm[0]))
    logger.info(
        "%s %r on %r: %d cand → %d on-surface, picked center=%s "
        "footprint yaw=%.0f° long=%.0f short=%.0f aspect=%.2f (square-up needs aspect≥%.2f) | "
        "face n=(%.2f,%.2f) angle=%.0f° (target→robot=%.0f°) flatness=%.2f trusted=%s (needs flatness≤%.2f)",
        log_prefix,
        object_name,
        on,
        n_cand,
        n_picks,
        [round(v, 1) for v in geo.center_mm],
        math.degrees(geo.yaw_rad),
        geo.long_mm,
        geo.short_mm,
        (hi / lo if lo > 1e-6 else 0.0),
        square_min_aspect,
        geo.face_normal_x,
        geo.face_normal_y,
        face_ang,
        to_robot,
        geo.face_flatness,
        trusted,
        face_flatness_max,
    )


# ---------------------------------------------------------------------------
# Body hooks — the seam an adapter may override, resolved HERE.
# ---------------------------------------------------------------------------
# These, and the three actions below, used to live on a ``Scene3D`` component the adapter
# held. Resolving them here lets the actions reach this module the same way every other
# action reaches its implementation: adapter method -> ``api.defaults`` -> here.
#
# ``api`` is duck-typed on purpose: this module must not import the api layer
# (tests/unit_tests/test_layering.py enforces it), and it needs nothing from it but
# ``env``, ``last_detection`` / ``last_surface``, and these four hook names.


def _scene_camera(api: Any) -> Any:
    """Pane/sensor name for bodies with several cameras; also labels debug overlays."""
    return getattr(api, "scene_camera", "scene")


def _grab_frame(api: Any, camera: str | None = None) -> Any:
    """One rgb + depth + intrinsics + base<-camera frame (default: the Env verb)."""
    started = time.monotonic()
    override = getattr(api, "_grab_calibrated_frame", None)
    frame = override(camera) if override else api.env.grab_calibrated_frame(camera)
    if frame is not None and getattr(frame, "captured_monotonic_s", None) is not None:
        detector = _seg_fn(api)
        if isinstance(detector, DetectorClient):
            detector.check_frame_age(frame.captured_monotonic_s, frame.frame_id or "")
    elif isinstance(frame, CameraFrame):
        frame.captured_monotonic_s = started
    if isinstance(frame, CameraFrame) and frame.frame_id is None:
        frame.frame_id = uuid.uuid4().hex
    return frame


def _seg_fn(api: Any, frame: CameraFrame | None = None) -> Any:
    """The open-vocabulary segmentation callable, or None when no detector is up."""
    override = getattr(api, "detector_seg_fn", None)
    fn = override() if override else getattr(api, "_seg_fn", None)
    if frame is not None and isinstance(fn, DetectorClient):
        if frame.captured_monotonic_s is None:
            frame.captured_monotonic_s = time.monotonic()
        if frame.frame_id is None:
            frame.frame_id = uuid.uuid4().hex
        return partial(segment_image, fn, captured_monotonic_s=frame.captured_monotonic_s, frame_id=frame.frame_id)
    return fn


def _check_frame_completion(api: Any, frame: CameraFrame) -> None:
    """Check the original capture deadline after geometry, before publishing caches."""
    detector = _seg_fn(api)
    if isinstance(detector, DetectorClient):
        captured = frame.captured_monotonic_s
        detector.check_frame_age(time.monotonic() if captured is None else captured, frame.frame_id or "")


def _viz(api: Any, camera: str, prompt: str, rgb: Any, best: dict | None) -> None:
    """Debug-overlay hook; no-op unless the body provides a viewer."""
    override = getattr(api, "viz_update", None)
    if override:
        override(camera, prompt, rgb, best)


def _unknown_relation(relation: str, object_name: str) -> dict | None:
    """Failure dict when ``relation`` is outside the closed set, else None.

    The schema advertises the enum, but nothing enforces a schema at dispatch, so a
    grounded sensing action checks before measuring anything — an unknown relation must
    fail by name rather than quietly behave like ``on``.
    """
    if relation in SPATIAL_RELATIONS:
        return None
    return {
        "ok": False,
        "reason": f"unknown_relation:{relation}",
        "object": object_name,
        "known_relations": list(SPATIAL_RELATIONS),
    }


# ------------------------------------------------------- action implementations


def _service_result(fn: Callable) -> Callable:
    """Translate service failures and discard any abandoned geometry caches."""

    @wraps(fn)
    def run(api: Any, *args: Any, **kwargs: Any) -> dict:
        try:
            return cast(dict, fn(api, *args, **kwargs))
        except InferenceServiceError as exc:
            api.last_detection = None
            api.last_surface = None
            return {
                "ok": False,
                "reason": "detector_unavailable",
                "error_code": exc.code,
                "object": args[0] if args else kwargs.get("object_name", ""),
            }
        except Exception:
            # Includes structural cancellation without importing the agent layer.
            # A geometry helper may have populated either cache before the final guard.
            api.last_detection = None
            api.last_surface = None
            raise

    return run


@_service_result
def locate_for_grasp(api: Any, object_name: str = "box", reference: str | None = None, relation: str = "on") -> dict:
    """Detect a target and return its 3-D geometry in the robot base frame (mm).

    With ``reference`` set, accept only the candidate standing in ``relation`` to it —
    a coarse-to-fine grounding that disambiguates same-class targets (see
    ``_detect_related``).
    """
    # Invalidate any prior detection up front: a failed detect must not leave a stale
    # geometry that the next grasp would then act on. Only a fresh success re-fills it.
    api.last_detection = None
    bad = _unknown_relation(relation, object_name)
    if bad is not None:
        return bad
    frame, fail = _calibrated_frame_or_reason(api, object_name)
    if fail is not None:
        return fail
    if reference:  # fine-grained: keep only the candidate related to the reference
        result = _detect_related(api, object_name, reference, relation, frame)
    else:
        result = detect_object_geometry(
            frame.rgb,
            frame.depth_m,
            frame.intrinsics,
            frame.tf_base_cam,
            seg_fn=_seg_fn(api, frame),
            object_name=object_name,
            log_prefix="[scene3d-object]",
            on_pick=lambda best: _viz(api, _scene_camera(api), object_name, frame.rgb, best),
        )
    _check_frame_completion(api, frame)
    if result.get("ok"):
        api.last_detection = result
    return result


@_service_result
def locate_for_place(api: Any, object_name: str = "table", reference: str | None = None, relation: str = "on") -> dict:
    """Detect a support surface and return its base-frame top height in mm.

    With ``reference`` set, keep only the surface standing in ``relation`` to it. The
    phrase reads the same way round as everywhere else — the table that *has* a cup on
    it is the table ``relation="under"`` the cup (see ``_sense_surface_related``).
    """
    # Same invariant as ``locate_for_grasp``: a failed sense must not leave a surface the
    # next place would land on. Only a fresh success re-fills it.
    api.last_surface = None
    bad = _unknown_relation(relation, object_name)
    if bad is not None:
        return bad
    frame, fail = _calibrated_frame_or_reason(api, object_name)
    if fail is not None:
        return fail
    if reference:  # fine-grained: keep only the surface related to the reference object
        result = _sense_surface_related(api, object_name, reference, relation, frame)
    else:
        result = _sense_surface_plain(
            api, object_name, frame.rgb, frame.depth_m, intr=frame.intrinsics, tf=frame.tf_base_cam, frame=frame
        )
    _check_frame_completion(api, frame)
    if result.get("ok"):
        api.last_surface = result
    return result


@_service_result
def analyze_scene(api: Any, object_name: str = "box") -> dict:
    """Every instance of ``object_name`` in view, nearest-first."""
    import numpy as np

    from jiuwensymbiosis.perception.vision import detect_all_object_geometry

    frame, fail = _calibrated_frame_or_reason(api, object_name)
    if fail is not None:
        return fail
    objs = detect_all_object_geometry(
        frame.rgb,
        frame.depth_m,
        np.asarray(frame.intrinsics),
        np.asarray(frame.tf_base_cam),
        seg_fn=_seg_fn(api, frame),
        object_name=object_name,
    )
    _check_frame_completion(api, frame)
    return {"ok": True, "object": object_name, "count": len(objs), "objects": objs}


# ------------------------------------------------------ shared internals


def _calibrated_frame_or_reason(api: Any, object_name: str) -> tuple[Any, dict | None]:
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
    for camera in getattr(api.env, "cameras", None) or (_scene_camera(api),):
        frame = _grab_frame(api, camera)
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
    api: Any,
    object_name: str,
    rgb: Any,
    depth_m: Any,
    *,
    intr: Any,
    tf_base_cam: Any,
    min_z_mm: float | None = None,
    frame: CameraFrame | None = None,
) -> list:
    """All ``object_name`` detections in the frame → colour-verified base-frame geometries."""
    return candidate_geometries(
        rgb,
        depth_m,
        intr,
        tf_base_cam,
        seg_fn=_seg_fn(api, frame),
        object_name=object_name,
        min_z_mm=min_z_mm,
    )


def _sense_surface_plain(
    api: Any, object_name: str, rgb: Any, depth_m: Any, *, intr: Any, tf: Any, frame: CameraFrame | None = None
) -> dict:
    """Detect a plain support surface from an ALREADY-GRABBED frame. Factored out of
    ``locate_for_place`` so a caller already holding a frame (``_detect_on_surface``) reuses
    it instead of grabbing again. Does NOT cache — that is the caller's decision.
    """
    return sense_surface_geometry(
        rgb,
        depth_m,
        intr,
        tf,
        seg_fn=_seg_fn(api, frame),
        object_name=object_name,
        log_prefix="[scene3d-surface]",
        on_pick=lambda best: _viz(api, _scene_camera(api), object_name, rgb, best),
    )


def _relation_thresholds(api: Any) -> dict[str, float]:
    """Tolerances the relation predicate reads off the body config (missing → defaults)."""
    cfg = getattr(api.env, "cfg", None)
    return {
        "margin_mm": float(getattr(cfg, "on_surface_margin_mm", 80.0)),
        "beside_max_gap_mm": float(getattr(cfg, "beside_max_gap_mm", 400.0)),
        "near_max_dist_mm": float(getattr(cfg, "near_max_dist_mm", 1000.0)),
    }


def _measure_reference(api: Any, reference: str, relation: str, frame: CameraFrame) -> Any:
    """The reference's geometry, measured the way this relation needs it.

    ``on`` / ``under`` are relations to a *support surface*, so the reference is sensed
    as one — that also yields the top height the target's face-normal fit must stay
    above. Every other relation is to an ordinary object, which must be measured as
    one: a surface record states no thickness, and ``beside`` compares z-extents.
    Returns None when the reference is not in view.
    """
    rgb, depth_m, intr, tf_base_cam = frame.rgb, frame.depth_m, frame.intrinsics, frame.tf_base_cam
    if relation in ("on", "under"):
        ref = _sense_surface_plain(api, reference, rgb, depth_m, intr=intr, tf=tf_base_cam, frame=frame)
        return ref if ref.get("ok") else None
    found = _candidate_geometries(api, reference, rgb, depth_m, intr=intr, tf_base_cam=tf_base_cam, frame=frame)
    refs = [g for g, _d in found]
    return min(refs, key=lambda g: g.center_mm[0]) if refs else None  # nearest


def _detect_related(api: Any, object_name: str, reference: str, relation: str, frame: CameraFrame) -> dict:
    """Two-stage 'X <relation> Y' grounding. Measure the reference, detect **all**
    ``object_name`` candidates from the given frame, and return the one that stands in
    ``relation`` to it. Nearest-to-robot wins; the caller validates freshness and caches it.
    """
    rgb, depth_m, intr, tf_base_cam = frame.rgb, frame.depth_m, frame.intrinsics, frame.tf_base_cam
    cfg = getattr(api.env, "cfg", None)
    # Reuse the frame already grabbed by locate_for_grasp rather than re-grabbing a second
    # one (halves grounded detection's camera cost). This is the target's reference, not a
    # place surface, so it must NOT touch api.last_surface.
    ref = _measure_reference(api, reference, relation, frame)
    if ref is None:
        return {"ok": False, "reason": f"reference_not_found:{reference}", "object": object_name}
    # Face normal from the target's own wall: for an ON relation drop points at/below the
    # reference surface top (+ a small margin) so the edge fit is the target's, not the
    # surface's dominant edge. No support surface is implied by the other relations.
    min_z = None
    if relation == "on":
        min_z = float(ref["surface_z_mm"]) + float(getattr(cfg, "grasp_face_above_surface_mm", 20.0))
    cands = _candidate_geometries(
        api, object_name, rgb, depth_m, intr=intr, tf_base_cam=tf_base_cam, min_z_mm=min_z, frame=frame
    )
    thresholds = _relation_thresholds(api)
    picks = [(g, d) for (g, d) in cands if relation_holds(g, ref, relation, **thresholds)]
    prompt = f"{object_name} {relation} {reference}"
    if not picks:
        _viz(api, _scene_camera(api), prompt, rgb, None)
        logger.info(
            "[scene3d] locate_for_grasp %r %s %r: %d cand → 0 matched", object_name, relation, reference, len(cands)
        )
        return {
            "ok": False,
            "reason": "no_target_matching_reference",
            "object": object_name,
            "reference": reference,
            "relation": relation,
        }
    geo, det = min(picks, key=lambda gd: gd[0].center_mm[0])  # nearest (smallest forward X)
    _viz(api, _scene_camera(api), prompt, rgb, {"ok": True, **det})
    result = {
        "ok": True,
        "reason": "",
        "object": object_name,
        "reference": reference,
        "relation": relation,
        **object_geometry_fields(geo),
    }
    log_grounded_pick(
        "[scene3d] locate_for_grasp",
        object_name,
        f"{relation} {reference}",
        n_cand=len(cands),
        n_picks=len(picks),
        geo=geo,
        face_flatness_max=float(getattr(cfg, "grasp_face_flatness_max", 0.15)),
        square_min_aspect=float(getattr(cfg, "grasp_square_min_aspect", 1.2)),
    )
    return result


def _sense_surface_related(api: Any, object_name: str, reference: str, relation: str, frame: CameraFrame) -> dict:
    """The place-side mirror of ``_detect_related``: among all ``object_name`` surface
    candidates keep the one standing in ``relation`` to the reference. So "the table
    that has the cup on it" arrives here as ``relation='under'``, reference='cup'.
    Nearest matching surface wins; the caller validates freshness and caches it.
    """
    rgb, depth_m, intr, tf_base_cam = frame.rgb, frame.depth_m, frame.intrinsics, frame.tf_base_cam
    found_refs = _candidate_geometries(api, reference, rgb, depth_m, intr=intr, tf_base_cam=tf_base_cam, frame=frame)
    refs = [g for g, _d in found_refs]
    if not refs:
        return {"ok": False, "reason": f"reference_not_found:{reference}", "object": object_name}
    ref = min(refs, key=lambda g: g.center_mm[0])  # nearest reference
    thresholds = _relation_thresholds(api)
    found = _candidate_geometries(api, object_name, rgb, depth_m, intr=intr, tf_base_cam=tf_base_cam, frame=frame)
    cands = [g for g, _d in found]
    picks = [s for s in cands if relation_holds(s, ref, relation, **thresholds)]
    if not picks:
        logger.info(
            "[scene3d] locate_for_place %r %s %r: %d surf → 0 matched", object_name, relation, reference, len(cands)
        )
        return {
            "ok": False,
            "reason": "no_surface_matching_reference",
            "object": object_name,
            "reference": reference,
            "relation": relation,
        }
    surf = min(picks, key=lambda s: s.center_mm[0])  # nearest matching surface
    # Same footprint payload as the plain path (via surface_footprint_fields) — the grounded path
    # MUST carry yaw_rad/edge_normal too, else the place-side squaring reads no normal.
    result = {
        "ok": True,
        "object": object_name,
        "reference": reference,
        "relation": relation,
        **surface_footprint_fields(surf),
    }
    logger.info(
        "[scene3d] locate_for_place %r %s %r: %d surf → %d matched, picked cy=%.1f z=%.1f%s",
        object_name,
        relation,
        reference,
        len(cands),
        len(picks),
        surf.center_mm[1],
        surf.top_z_mm,
        edge_log_str(surf),
    )
    return result
