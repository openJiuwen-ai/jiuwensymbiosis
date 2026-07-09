# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Cross-vendor vision helpers used by RobotApi.get_grasp_info_simple
(and reusable by future adapters with detector + RGB-D + back-projection).

Three building blocks:

* :func:`detect_and_centroid` — detector detect + median pixel centroid +
  median depth in a small window. Hardware-agnostic.
* :func:`apply_xy_correction` — apply a multi-point xy_transform or
  legacy xy_correction_mm to a back-projected base-frame XYZ. The
  preference order matches the existing robot behavior verbatim.
* :func:`dump_grasp_debug` — write raw RGB + detector overlay + a JSON snapshot
  to disk, for offline correlation with the live test_detector_live.py output.
  Takes an ``extra_info`` dict so adapters can stuff any kinematic details
  the detector detection didn't see (e.g. joint angles at projection time).

Per-vendor projection stays in the
per-vendor api module; this module deliberately knows nothing about it.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result type contract for vision / grasp detection.
#
# Adapters and the LLM both read these dicts; pinning the reason strings as a
# constant (and the shapes as TypedDicts) gives both sides a stable contract
# instead of free-form strings that silently drift (``detector_unavailable``
# vs ``no_detection`` etc.). ``detect_and_centroid`` and the default eye-in-hand
# helpers emit only reasons from this set; ``no_camera`` is the adapter-level
# reason (the frame grab returned None before detection ran).
# ---------------------------------------------------------------------------
DETECTION_REASONS = frozenset(
    {
        "no_camera",
        "no_detection",
        "empty_mask",
        "no_valid_depth",
        "detector_unavailable",
    }
)

DetectionReason = Literal[
    "no_camera",
    "no_detection",
    "empty_mask",
    "no_valid_depth",
    "detector_unavailable",
]


class GraspFailure(TypedDict):
    """Failure shape returned by grasp/detection tools."""

    ok: Literal[False]
    reason: DetectionReason
    object: str


class GraspResult(TypedDict, total=False):
    """Success shape returned by ``get_grasp_info_simple``.

    ``total=False`` because not every caller populates every field (e.g. some
    adapters omit ``place_z``). ``ok`` is the one always-present key.
    """

    ok: Literal[True]
    object: str
    position: list  # [x, y, z]_mm
    grasp_z: float
    grasp_position: list  # [x, y, z]_mm
    place_z: float
    place_position: list  # [x, y, z]_mm
    score: float
    pixel_uv: list  # [u, v]
    depth_m: float


# Single shared counter so artifacts from multiple sessions don't stomp
# each other. Resets to 1 on each fresh Python process.
_GRASP_DEBUG_COUNTER = itertools.count(1)

# Per-process stamp so the default debug dir lands under a unique subdir
# each time the demo is launched. Computed once at module import.
_RUN_STAMP = time.strftime("%Y-%m-%d_%H-%M-%S")


def _run_detect_pick_best(
    rgb: np.ndarray,
    seg_fn: Callable[..., list[dict]],
    object_name: str,
    score_threshold: float,
    log_prefix: str,
) -> dict:
    """Run detector, log every candidate, and return the highest-score result.

    Returns the chosen detector result dict on success, or a ``{"ok": False, ...}``
    failure dict when nothing scores above ``score_threshold``.
    """
    results_raw = seg_fn(rgb, text_prompt=object_name)
    results = [r for r in results_raw if r.get("score", 0.0) >= score_threshold]
    logger.info(
        "%s detector returned %d raw → %d after score>=%.2f",
        log_prefix,
        len(results_raw),
        len(results),
        score_threshold,
    )
    for i, r in enumerate(results):
        m = r["mask"]
        logger.info(
            "%s   cand[%d] score=%.3f label=%r mask_shape=%s n_pixels=%d coverage=%.1f%% box=%s",
            log_prefix,
            i,
            float(r.get("score", 0.0)),
            r.get("label", ""),
            tuple(m.shape),
            int(m.sum()),
            100.0 * float(m.sum()) / (m.shape[0] * m.shape[1]),
            [round(float(b), 1) for b in r["box"][:4]],
        )
    if not results:
        return {"ok": False, "reason": "no_detection", "object": object_name}
    return max(results, key=lambda r: r["score"])


def _mask_centroid(
    best: dict,
    img_w: int,
    img_h: int,
    log_prefix: str,
) -> dict:
    """Median mask-pixel centroid, scaled from mask coords to image coords.

    detector mask shape may not match the RGB shape; scaling to image coords keeps
    the caller's pixel→base back-projection (which uses image-resolution K)
    consistent. The caller guarantees ``best["mask"]`` is non-empty.
    """
    ys, xs = np.where(best["mask"])
    mask_h, mask_w = best["mask"].shape[:2]
    u_mask = float(np.median(xs))
    v_mask = float(np.median(ys))
    scale_x = img_w / mask_w
    scale_y = img_h / mask_h
    u = u_mask * scale_x
    v = v_mask * scale_y
    logger.info(
        "%s centroid: mask=%dx%d (u,v)_mask=(%.1f, %.1f) scale=(%.3f, %.3f) (u,v)_img=(%.1f, %.1f)",
        log_prefix,
        mask_w,
        mask_h,
        u_mask,
        v_mask,
        scale_x,
        scale_y,
        u,
        v,
    )
    return {
        "u": u,
        "v": v,
        "u_mask": u_mask,
        "v_mask": v_mask,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "mask_h": mask_h,
        "mask_w": mask_w,
    }


def _median_depth_window(
    depth_img_m: np.ndarray,
    u: float,
    v: float,
    log_prefix: str,
) -> float | None:
    """Median of valid depths (m) in a 5x5 window around (u, v); None if none."""
    h, w = depth_img_m.shape
    cu, cv, win = int(round(u)), int(round(v)), 5
    x0, x1 = max(0, cu - win), min(w, cu + win + 1)
    y0, y1 = max(0, cv - win), min(h, cv + win + 1)
    patch = depth_img_m[y0:y1, x0:x1]
    valid = patch[(patch > 0) & np.isfinite(patch)]
    logger.info(
        "%s depth window: x=[%d,%d) y=[%d,%d) patch_size=%d valid=%d depth_range=[%s, %s] median=%s",
        log_prefix,
        x0,
        x1,
        y0,
        y1,
        patch.size,
        int(valid.size),
        f"{float(valid.min()):.4f}" if valid.size else "n/a",
        f"{float(valid.max()):.4f}" if valid.size else "n/a",
        f"{float(np.median(valid)):.4f}" if valid.size else "n/a",
    )
    if valid.size == 0:
        return None
    return float(np.median(valid))


def detect_and_centroid(
    *,
    rgb: np.ndarray,
    depth_img_m: np.ndarray,
    seg_fn: Callable[..., list[dict]] | None,
    object_name: str,
    tcp_at_grab: Any,
    score_threshold: float = 0.05,
    log_prefix: str = "[grasp-debug]",
) -> dict:
    """Run detector, pick the best mask, compute median (u, v) and median depth.

    Returns one of:
      * ``{"ok": False, "reason": "no_detection"|"empty_mask"|"no_valid_depth"|"detector_unavailable", ...}``
      * ``{"ok": True, "u": float, "v": float, "depth_m": float, "best": <detector result dict>,
            "mask_shape": (h, w), "u_mask": float, "v_mask": float,
            "scale_x": float, "scale_y": float, "img_shape": (w, h)}``

    Every ``reason`` value is drawn from ``DETECTION_REASONS`` so callers and
    the LLM share a stable contract.

    ``tcp_at_grab`` is passed in only for the diagnostic log line that says
    where the arm was when the frame was grabbed; this module never reads
    or interprets the pose itself.
    """
    img_h, img_w = rgb.shape[:2]
    dep_h, dep_w = depth_img_m.shape[:2]
    logger.info(
        "%s grab: rgb=%dx%d depth=%dx%d tcp=(%.2f, %.2f, %.2f, %.2f) obj=%r",
        log_prefix,
        img_w,
        img_h,
        dep_w,
        dep_h,
        tcp_at_grab.x,
        tcp_at_grab.y,
        tcp_at_grab.z,
        tcp_at_grab.r,
        object_name,
    )
    if (img_w, img_h) != (dep_w, dep_h):
        logger.warning(
            "%s RGB / depth shapes differ — depth lookup at (u,v) assumes the same pixel grid. RGB=%dx%d depth=%dx%d",
            log_prefix,
            img_w,
            img_h,
            dep_w,
            dep_h,
        )

    if seg_fn is None:
        return {"ok": False, "reason": "detector_unavailable"}

    best = _run_detect_pick_best(
        rgb,
        seg_fn,
        object_name,
        score_threshold,
        log_prefix,
    )
    if best.get("ok") is False:
        return best
    if not best["mask"].any():
        return {"ok": False, "reason": "empty_mask"}

    centroid = _mask_centroid(best, img_w, img_h, log_prefix)
    depth_m = _median_depth_window(
        depth_img_m,
        centroid["u"],
        centroid["v"],
        log_prefix,
    )
    if depth_m is None:
        return {"ok": False, "reason": "no_valid_depth"}

    return {
        "ok": True,
        "u": centroid["u"],
        "v": centroid["v"],
        "depth_m": depth_m,
        "best": best,
        "mask_shape": (centroid["mask_h"], centroid["mask_w"]),
        "u_mask": centroid["u_mask"],
        "v_mask": centroid["v_mask"],
        "scale_x": centroid["scale_x"],
        "scale_y": centroid["scale_y"],
        "img_shape": (img_w, img_h),
    }


def apply_xy_correction(
    xyz_raw: np.ndarray,
    *,
    xy_transform: dict | None = None,
    xy_correction_mm: list[float] | None = None,
) -> tuple[np.ndarray, str]:
    """Apply a 2D linear xy correction to a back-projected base-frame XYZ.

    Preference (matches the existing robot behavior):
      1. ``xy_transform`` — multi-sample affine/similarity/translation fit
         from ``robot calibrate --correct``. Captures translation AND
         rotation/scale residuals in T_J3link_cam.
      2. ``xy_correction_mm`` — legacy single-point translation, kept for
         backward compatibility. Used only when ``xy_transform`` is None.

    Returns ``(xyz_final, corr_desc)`` where ``corr_desc`` is a short human
    string describing which correction (if any) was applied — useful for
    the debug log line.
    """
    xyz_final = xyz_raw.copy()
    if xy_transform is not None:
        a_mat = np.asarray(xy_transform["A"], dtype=np.float64)
        b_vec = np.asarray(xy_transform["b"], dtype=np.float64)
        xy_new = a_mat @ np.array([xyz_raw[0], xyz_raw[1]], dtype=np.float64) + b_vec
        xyz_final = np.array([xy_new[0], xy_new[1], xyz_raw[2]], dtype=np.float64)
        corr_desc = (
            f"xy_transform({xy_transform.get('method')}, "
            f"N={xy_transform.get('n_samples')}, "
            f"rms={xy_transform.get('rms_residual_mm'):.2f}mm, "
            f"Δ=({xyz_final[0] - xyz_raw[0]:+.2f}, "
            f"{xyz_final[1] - xyz_raw[1]:+.2f}))"
        )
    elif xy_correction_mm is not None:
        xyz_final[0] += float(xy_correction_mm[0])
        xyz_final[1] += float(xy_correction_mm[1])
        corr_desc = f"xy_correction_mm={list(xy_correction_mm)}"
    else:
        corr_desc = "none"
    return xyz_final, corr_desc


# ---------------------------------------------------------------------------
# Default eye-in-hand implementations.
#
# ``get_grasp_info_simple`` / ``pixel_to_base_xyz`` cannot have a *generic*
# default on the mixin because they depend on the adapter's hand-eye
# calibration — but the ~130-line pipeline (grab → detect → centroid → depth
# → project → xy-correct → grasp/place geometry) is identical for every
# eye-in-hand camera robot. These helpers factor it out; an adapter only
# supplies:
#   * its detector ``seg_fn`` (lazy-bound like PiperApi._ensure_detector),
#   * a ``pose_to_tf(flange_pose) -> 4x4`` callback (the one truly vendor-
#     specific piece: how the vendor's flange pose becomes a base-frame
#     transform; Piper uses FlangePose(...).to_tf_base_flange()).
# and reads calibration (``tf_flange_cam`` / ``intrinsics`` / ``calibration`` /
# ``grab_frames``) straight off ``api.env.low_level`` — already a VisionDriver
# Protocol surface.
# ---------------------------------------------------------------------------


def _resolve_intrinsics(ll: Any) -> np.ndarray | None:
    """Intrinsics from calibration (preferred) else live camera."""
    calib = getattr(ll, "calibration", None)
    intrinsics = calib.get("intrinsics") if calib is not None else None
    if intrinsics is None:
        intrinsics = getattr(ll, "intrinsics", None)
    return intrinsics


def default_pixel_to_base_xyz(
    api: Any, u: float, v: float, depth_m: float, *, pose_to_tf: Callable[[Any], np.ndarray]
) -> dict:
    """Back-project (u, v, depth_m) → base-frame XYZ (mm) via eye-in-hand math.

    Reads ``tf_flange_cam`` / ``intrinsics`` from ``api.env.low_level``; applies
    the calibration's ``xy_transform``/``xy_correction_mm`` when present.
    ``pose_to_tf`` converts the env's vendor flange pose to a 4x4 base←flange
    transform (the one vendor-specific step).
    """
    from jiuwensymbiosis.utils.geometry import (
        apply_transform,
        pixel_and_depth_to_camera_xyz,
    )

    ll = api.env.low_level
    if ll.tf_flange_cam is None:
        raise RuntimeError("pixel_to_base_xyz needs a loaded calibration (set calib_path in YAML).")
    intrinsics = _resolve_intrinsics(ll)
    if intrinsics is None:
        raise RuntimeError("camera intrinsics unavailable (no calibration, no live camera)")
    tf_base_flange = pose_to_tf(api.env.get_flange_pose())
    tf_base_cam = tf_base_flange @ ll.tf_flange_cam
    xyz = apply_transform(tf_base_cam, pixel_and_depth_to_camera_xyz((u, v), depth_m, intrinsics))
    calib = getattr(ll, "calibration", None)
    if calib is not None:
        xyz, _desc = apply_xy_correction(
            np.asarray(xyz, dtype=np.float64),
            xy_transform=calib.get("xy_transform"),
            xy_correction_mm=calib.get("xy_correction_mm"),
        )
    return {"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])}


def default_get_grasp_info_simple(
    api: Any,
    object_name: str,
    *,
    seg_fn: Callable[..., list[dict]] | None,
    pose_to_tf: Callable[[Any], np.ndarray],
    z_correction_mm: float = 0.0,
    grasp_z_offset_mm: float = -25.0,
    chip_thickness_mm: float = 75.0,
    score_threshold: float = 0.05,
) -> dict:
    """Default ``get_grasp_info_simple`` for an eye-in-hand camera robot.

    Runs the standard detect → centroid → depth → back-project → xy-correct →
    grasp/place-geometry pipeline, returning the same shape Piper does:
    ``{ok, object, position, grasp_z, grasp_position, place_z, place_position,
    score, pixel_uv, depth_m}``. On failure returns a ``GraspFailure`` whose
    ``reason`` is drawn from ``DETECTION_REASONS``.

    Args:
      api: an Api-like object exposing ``env`` (with ``low_level`` a
        VisionDriver and ``get_flange_pose``/``z_min_safe``).
      seg_fn: the detector segmentation callable (``None`` → detector_unavailable).
      pose_to_tf: vendor flange-pose → 4x4 base←flange transform.
      z_correction_mm / grasp_z_offset_mm / chip_thickness_mm: grasp geometry
        constants (see PiperConfig for semantics).
    """
    from types import SimpleNamespace

    from jiuwensymbiosis.utils.geometry import (
        apply_transform,
        pixel_and_depth_to_camera_xyz,
    )

    ll = api.env.low_level
    frames = ll.grab_frames()
    if frames is None:
        return {"ok": False, "reason": "no_camera", "object": object_name}
    rgb, depth_img_m = frames

    det = detect_and_centroid(
        rgb=rgb,
        depth_img_m=depth_img_m,
        seg_fn=seg_fn,
        object_name=object_name,
        tcp_at_grab=SimpleNamespace(x=0.0, y=0.0, z=0.0, r=0.0),
        score_threshold=score_threshold,
    )
    if not det.get("ok"):
        return det

    if ll.tf_flange_cam is None:
        raise RuntimeError("get_grasp_info_simple needs a loaded calibration (set calib_path in YAML).")
    intrinsics = _resolve_intrinsics(ll)
    if intrinsics is None:
        raise RuntimeError("camera intrinsics unavailable (no calibration, no live camera)")

    u, v, depth_m = det["u"], det["v"], det["depth_m"]
    tf_base_flange = pose_to_tf(api.env.get_flange_pose())
    tf_base_cam = tf_base_flange @ ll.tf_flange_cam
    xyz_raw = apply_transform(tf_base_cam, pixel_and_depth_to_camera_xyz((u, v), depth_m, intrinsics))

    calib = getattr(ll, "calibration", None)
    xy_transform = calib.get("xy_transform") if calib is not None else None
    xy_corr = calib.get("xy_correction_mm") if (calib is not None and xy_transform is None) else None
    xyz_final, _corr_desc = apply_xy_correction(xyz_raw, xy_transform=xy_transform, xy_correction_mm=xy_corr)
    if z_correction_mm:
        xyz_final = np.asarray(xyz_final, dtype=np.float64).copy()
        xyz_final[2] += z_correction_mm

    top_z = float(xyz_final[2])
    z_floor = api.env.z_min_safe
    grasp_z = top_z + grasp_z_offset_mm
    if z_floor is not None:
        grasp_z = max(grasp_z, float(z_floor))
    place_z = top_z + chip_thickness_mm
    x_f, y_f = float(xyz_final[0]), float(xyz_final[1])
    best = det["best"]
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


def _default_debug_dir() -> Path:
    """Resolve where ``dump_grasp_debug`` writes artifacts.

    Resolution order (first match wins):
      1. ``$JIUWEN_GRASP_DEBUG_DIR`` (verbatim) — explicit user override.
      2. ``$JIUWEN_MOTION_LOG_RUN_DIR/grasp_debug`` — current motion-log run.
      3. ``$JIUWEN_GRASP_DEBUG_ROOT/<run-stamp>`` — legacy explicit root.
      4. ``$JIUWEN_CMD_LOG_DIR/<run-stamp>/grasp_debug``.
      5. ``./jiuwen_motion_log/<run-stamp>/grasp_debug``.

    The ``<run-stamp>`` is computed once at module import (``YYYY-MM-DD_HH-MM-SS``)
    so every invocation of the demo gets its own subdirectory, with detections
    accumulating in order inside it. Previous runs are NOT overwritten.
    """
    explicit = os.environ.get("JIUWEN_GRASP_DEBUG_DIR")
    if explicit:
        return Path(explicit)
    motion_run_dir = os.environ.get("JIUWEN_MOTION_LOG_RUN_DIR")
    if motion_run_dir:
        return Path(motion_run_dir) / "grasp_debug"
    legacy_root = os.environ.get("JIUWEN_GRASP_DEBUG_ROOT")
    if legacy_root:
        return Path(legacy_root) / _RUN_STAMP
    motion_root = os.environ.get("JIUWEN_CMD_LOG_DIR", "./jiuwen_motion_log")
    return Path(motion_root) / _RUN_STAMP / "grasp_debug"


def _save_raw_and_depth(
    rgb: np.ndarray,
    depth_img: np.ndarray | None,
    debug_dir: Path,
    idx: int,
) -> None:
    """Save ``raw_{idx}.jpg`` and, best-effort, ``depth_{idx}.npy``."""
    from PIL import Image

    Image.fromarray(rgb).convert("RGB").save(debug_dir / f"raw_{idx:03d}.jpg")
    # Persist the depth frame (metres) so the slot post-processing — which now
    # keys the through-hole removal off depth — can be re-tuned offline against
    # recorded runs (RGB alone can't reproduce the depth path).
    if depth_img is not None:
        try:
            np.save(debug_dir / f"depth_{idx:03d}.npy", np.asarray(depth_img))
        except Exception:  # noqa: BLE001 - depth dump is best-effort
            pass


def _composite_red_mask(pil_img, mask: np.ndarray):
    """Composite a red translucent detector mask onto ``pil_img`` (returns a new image)."""
    from PIL import Image

    # PIL stub omits legacy module-level NEAREST constant (value 0); runtime-correct
    mask_pil = Image.fromarray(mask.astype(np.uint8)).resize(
        pil_img.size,
        Image.NEAREST,  # type: ignore[attr-defined]
    )
    overlay = Image.new("RGB", pil_img.size, (255, 0, 0))
    return Image.composite(overlay, pil_img, mask_pil.point(lambda p: int(p * 128)))


def _annotate_detection(
    draw,
    best: dict,
    uv: tuple[float, float],
    rejected: bool,
    box_color: tuple,
) -> None:
    """Draw the detection box, label, and yellow centroid crosshair."""
    x1, y1, x2, y2 = (int(round(float(b))) for b in best["box"][:4])
    draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
    status_prefix = "REFINE REJECTED keep coarse  " if rejected else ""
    draw.text(
        (x1 + 2, max(y1 - 22, 2)),
        f"{status_prefix}{best.get('label', '')} {float(best['score']):.2f}",
        fill=box_color,
    )
    cu_i, cv_i = int(round(uv[0])), int(round(uv[1]))
    rad = 6
    draw.ellipse(
        [cu_i - rad, cv_i - rad, cu_i + rad, cv_i + rad],
        outline=(255, 255, 0),
        width=2,
    )
    draw.line([cu_i - rad - 4, cv_i, cu_i + rad + 4, cv_i], fill=(255, 255, 0), width=2)
    draw.line([cu_i, cv_i - rad - 4, cu_i, cv_i + rad + 4], fill=(255, 255, 0), width=2)


def _overlay_slot_surface(pil_img, draw, surface_mask):
    """Composite the cyan slot-surface overlay (best-effort); returns (img, draw)."""
    from PIL import Image, ImageDraw

    if surface_mask is None:
        return pil_img, draw
    try:
        surface_arr = np.asarray(surface_mask).astype(np.uint8)
        if surface_arr.size:
            surface_pil = Image.fromarray(surface_arr * 96).resize(
                pil_img.size,
                Image.NEAREST,
            )
            surface_overlay = Image.new("RGB", pil_img.size, (0, 220, 255))
            pil_img = Image.composite(
                surface_overlay,
                pil_img,
                surface_pil.point(int),
            )
            draw = ImageDraw.Draw(pil_img)
    except Exception as exc:  # noqa: BLE001 - surface overlay is best-effort
        logger.debug("[grasp-debug] slot-surface overlay skipped: %s", exc)
    return pil_img, draw


def _draw_core_markers(draw, core_metrics: dict, img_height: int) -> None:
    """Draw the cyan core box, orange raw-centroid dot, and the legend text."""
    core_box = core_metrics.get("surface_box") or core_metrics.get("dumbbell_box") or core_metrics.get("core_box")
    if isinstance(core_box, (list, tuple)) and len(core_box) >= 4:
        cx1, cy1, cx2, cy2 = (int(round(float(b))) for b in core_box[:4])
        draw.rectangle([cx1, cy1, cx2, cy2], outline=(0, 220, 255), width=3)
    mask_uv = core_metrics.get("raw_mask_centroid_uv") or core_metrics.get("mask_centroid_uv")
    if isinstance(mask_uv, (list, tuple)) and len(mask_uv) >= 2:
        mu, mv = int(round(float(mask_uv[0]))), int(round(float(mask_uv[1])))
        draw.ellipse(
            [mu - 5, mv - 5, mu + 5, mv + 5],
            outline=(255, 165, 0),
            width=2,
        )
    draw.text(
        (4, img_height - 18),
        ("red=detector mask  cyan=slot surface used  yellow=place uv  orange=raw center/status"),
        fill=(0, 220, 255),
    )


def _annotate_slot_core(pil_img, draw, extra: dict, rejected: bool, box_color: tuple):
    """Draw slot-core annotations; returns the (possibly recomposited) (img, draw)."""
    core_metrics = extra.get("slot_core_metrics")
    if isinstance(core_metrics, dict) and core_metrics.get("used"):
        pil_img, draw = _overlay_slot_surface(
            pil_img,
            draw,
            extra.get("slot_surface_mask"),
        )
        _draw_core_markers(draw, core_metrics, pil_img.height)
    elif rejected:
        draw.text(
            (4, pil_img.height - 18),
            "orange=rejected refine candidate; final placement keeps coarse slot",
            fill=box_color,
        )
    return pil_img, draw


def dump_grasp_debug(
    *,
    rgb: np.ndarray,
    object_name: str,
    best: dict,
    u: float,
    v: float,
    depth_m: float,
    tcp_grab: Any,
    tcp_proj: Any,
    xyz_raw: np.ndarray,
    xyz_final: np.ndarray,
    xy_corr: Any = None,
    xy_transform: Any = None,
    intrinsics_src: str = "",
    intrinsics: list | None = None,
    img_shape: tuple[int, int] = (0, 0),
    mask_shape: tuple[int, int] = (0, 0),
    extra_info: dict | None = None,
    debug_dir: Path | None = None,
    depth_img: np.ndarray | None = None,
) -> None:
    """Save raw + detector overlay + JSON snapshot for offline comparison.

    Adapter-specific debug payload (e.g. joint angles, T_J3link_cam) goes
    in ``extra_info`` and is merged into the JSON file.

    Never raises — failure to dump must not break the live pipeline.
    """
    try:
        from PIL import Image, ImageDraw  # local import: keep cold path cheap
    except ImportError:
        return
    if debug_dir is None:
        debug_dir = _default_debug_dir()
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        idx = next(_GRASP_DEBUG_COUNTER)
        _save_raw_and_depth(rgb, depth_img, debug_dir, idx)

        # Red translucent mask + status-coloured box + yellow centroid marker —
        # same style as scripts/test_detector_live.py for direct comparison.
        extra = extra_info or {}
        rejected = str(extra.get("debug_visual_state") or "").startswith("rejected")
        box_color = (255, 128, 0) if rejected else (0, 255, 0)
        pil_img = _composite_red_mask(
            Image.fromarray(rgb).convert("RGB"),
            best["mask"],
        )
        draw = ImageDraw.Draw(pil_img)
        _annotate_detection(draw, best, (u, v), rejected, box_color)
        pil_img, draw = _annotate_slot_core(pil_img, draw, extra, rejected, box_color)
        pil_img.save(debug_dir / f"det_{idx:03d}.jpg")

        info: dict[str, Any] = {
            "idx": idx,
            "object": object_name,
            "img_shape_wh": list(img_shape),
            "mask_shape_wh": list(mask_shape),
            "pixel_uv": [float(u), float(v)],
            "depth_m": float(depth_m),
            "score": float(best["score"]),
            "box": [float(b) for b in best["box"][:4]],
            "tcp_at_grab": [tcp_grab.x, tcp_grab.y, tcp_grab.z, tcp_grab.r],
            "tcp_at_proj": [tcp_proj.x, tcp_proj.y, tcp_proj.z, tcp_proj.r],
            "K_src": intrinsics_src,
            "K": list(intrinsics) if intrinsics is not None else None,
            "xyz_raw_mm": [float(c) for c in xyz_raw],
            "xy_correction_mm": list(xy_corr) if xy_corr is not None else None,
            "xy_transform": xy_transform,
            "xy_correction_effective_mm": [
                float(xyz_final[0] - xyz_raw[0]),
                float(xyz_final[1] - xyz_raw[1]),
            ],
            "xyz_final_mm": [float(c) for c in xyz_final],
        }
        if extra_info:
            info.update({key: value for key, value in extra_info.items() if key != "slot_surface_mask"})
        info_path = debug_dir / f"info_{idx:03d}.json"
        info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False))
        logger.info(
            "[grasp-debug] dumped #%d → raw_%03d.jpg / det_%03d.jpg / %s",
            idx,
            idx,
            idx,
            info_path.name,
        )
    except Exception as exc:  # noqa: BLE001 - never let debug saving break the pipeline
        logger.warning("[grasp-debug] dump failed: %s", exc)
