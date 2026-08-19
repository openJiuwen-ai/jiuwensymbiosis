# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Cruzr-only one-shot debug tool: visualize how the head RGB, the detection masks, and the head
stereo point cloud line up, so we can tell WHY grounded head verify degrades with "too few valid
cloud points" — a genuine RGB↔cloud misregistration, an untextured-face stereo hole, or a wrong
base←optical TF.

It grabs ONE head frame through the exact runtime path (``CruzrCamera.grab_head_frame`` → the
one-shot subprocess worker, so the cloud/TF are the real ones the verify uses), runs the detector
for the target + reference nouns, and renders a 3-panel PNG:

  [ head RGB ] [ cloud range-image ] [ RGB×range blend + mask contours + per-mask stats ]

Panel 2 colourises the ORGANIZED cloud by point range (metres); NaN holes on untextured faces show
as black, so sparse coverage is obvious at a glance. Panel 3 blends it onto the RGB and draws each
mask's contour (target = green, reference = red): if the cloud is registered, the contour hugs the
same object silhouette in the depth image; if it's offset, they don't line up. Each mask is
annotated with ``n_valid`` (finite in-mask cloud points, the number the DEGRADE log hides) and its
base-frame centroid in mm — a centroid far from the box's real position points at the TF, not the
registration.

Run at the robot's current pose (needs ROS + head topics live + detector :8114):

    python -m scripts.cruzr.debug_align \
        --config configs/cruzr/cruzr.yaml --object "white box" --on "brown box" \
        --out /tmp/head_align.png

Rendering (:func:`render_alignment`) is pure numpy+cv2 and offline-testable; only ``grab_and_render``
/ ``main`` touch ROS + the detector.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _reshape_flat_cloud(cloud: np.ndarray, rgb_hw: Optional[tuple[int, int]]) -> np.ndarray:
    """Mirror ``head_geometry_from_mask``: recover an organized H×W×3 grid from a flat/unorganized
    cloud when its point count matches the RGB. Leaves an already-organized cloud untouched.
    """
    if rgb_hw is None or cloud.ndim not in (2, 3) or cloud.shape[-1] < 3:
        return cloud
    rh, rw = int(rgb_hw[0]), int(rgb_hw[1])
    n_pts = int(cloud.size // cloud.shape[-1])
    is_flat = cloud.ndim == 2 or cloud.shape[0] == 1
    if is_flat and min(rh, rw) > 0 and rh * rw == n_pts:
        return cloud.reshape(rh, rw, cloud.shape[-1])
    return cloud


def _range_colormap(cloud: np.ndarray, out_hw: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Colourise an organized cloud (H×W×3, metres) by per-point range; NaN holes → black. Resized
    (nearest, to preserve holes) to ``out_hw``. Returns ``(bgr_image, valid_fraction)``.
    """
    import cv2

    xyz = cloud[..., :3].astype(np.float64)
    finite = np.isfinite(xyz).all(axis=2)
    rng = np.sqrt((xyz ** 2).sum(axis=2))
    valid_frac = float(finite.mean()) if finite.size else 0.0

    vis = np.zeros(cloud.shape[:2], dtype=np.uint8)
    if finite.any():
        vals = rng[finite]
        lo, hi = np.percentile(vals, 2), np.percentile(vals, 98)   # robust contrast
        if hi <= lo:
            hi = lo + 1e-6
        norm = np.clip((rng - lo) / (hi - lo), 0.0, 1.0)
        vis[finite] = (norm[finite] * 255).astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    color[~finite] = (0, 0, 0)                                     # holes stay black
    oh, ow = out_hw
    color = cv2.resize(color, (ow, oh), interpolation=cv2.INTER_NEAREST)
    return color, valid_frac


def _put_lines(img: np.ndarray, lines: list[str], org: tuple[int, int],
               color: tuple[int, int, int]) -> None:
    import cv2

    x, y = org
    for i, text in enumerate(lines):
        yy = y + i * 20
        cv2.putText(img, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def render_alignment(rgb: np.ndarray, cloud: Optional[np.ndarray], tf: Optional[np.ndarray], *,
                     dets: list[dict], min_valid: int = 30) -> np.ndarray:
    """Render the RGB | cloud-range | blend+contours+stats triptych as one BGR image.

    ``rgb`` is the head RGB (H×W×3, RGB order). ``cloud`` is the organized (or flat, auto-reshaped)
    head cloud in metres, ``tf`` the base←optical 4×4 (mm translation). ``dets`` is a list of
    ``{"role", "name", "mask", "box", "score"}`` (role="target"/"reference"); each mask is bool at
    RGB resolution. ``min_valid`` is the runtime's ``head_cloud_min_valid_points`` gate — reported
    per mask so it's obvious which detections fall under it. Pure numpy+cv2; no ROS/GUI.
    """
    import cv2

    from scripts.cruzr._head_cloud import head_geometry_from_mask

    rgb = np.ascontiguousarray(rgb)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    rgb_bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Panel 2: cloud range image (or a placeholder when no cloud arrived).
    if cloud is not None:
        cloud = _reshape_flat_cloud(np.asarray(cloud, dtype=np.float64), (h, w))
    if cloud is not None and cloud.ndim == 3 and cloud.shape[2] >= 3:
        ch, cw = int(cloud.shape[0]), int(cloud.shape[1])
        range_img, valid_frac = _range_colormap(cloud, (h, w))
        cloud_note = f"cloud {ch}x{cw}  rgb {h}x{w}  valid {valid_frac * 100:.0f}%"
    else:
        ch = cw = 0
        range_img = np.zeros_like(rgb_bgr)
        valid_frac = 0.0
        cloud_note = "NO CLOUD"

    # Panel 3: blend RGB (dimmed) with the range image, then draw mask contours + per-mask stats.
    blend = cv2.addWeighted(rgb_bgr, 0.5, range_img, 0.6, 0.0)
    role_color = {"target": (0, 255, 0), "reference": (0, 0, 255)}   # BGR
    stat_lines: list[str] = [cloud_note]
    for d in dets:
        color = role_color.get(d.get("role", "target"), (0, 255, 255))
        mask = d.get("mask")
        name = d.get("name", d.get("role", "?"))
        if mask is None:
            stat_lines.append(f"{d.get('role','?')} {name}: no mask")
            continue
        m = np.asarray(mask).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blend, cnts, -1, color, 2)
        g = None
        if cloud is not None and tf is not None:
            g = head_geometry_from_mask(np.asarray(mask), cloud, np.asarray(tf),
                                        min_valid_points=1, rgb_hw=(h, w))
        if g is None:
            stat_lines.append(f"{d.get('role','?')} {name}: n_valid=0 (<{min_valid}) NO GEOM")
        else:
            flag = "" if g.n_valid >= min_valid else f" (<{min_valid} DEGRADE)"
            stat_lines.append(
                f"{d.get('role','?')} {name}: n_valid={g.n_valid}{flag} "
                f"c=({g.center_x:.0f},{g.center_y:.0f},{g.center_z:.0f})mm")
    _put_lines(blend, stat_lines, (8, 24), (255, 255, 255))

    # Panel labels.
    for panel, label, col in ((rgb_bgr, "1 head RGB", (255, 255, 255)),
                              (range_img, "2 cloud range (NaN=black)", (255, 255, 255)),
                              (blend, "3 blend + masks (green=target red=ref)", (255, 255, 255))):
        cv2.putText(panel, label, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(panel, label, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    return np.hstack([rgb_bgr, range_img, blend])


def grab_and_render(cfg: Any, detector_url: str, object_name: str, on: str, out_path: str, *,
                    show: bool = False) -> Optional[str]:
    """Grab one live head frame (rgb+cloud+tf), detect the target + reference nouns, render the
    alignment triptych, and write it to ``out_path`` (also imshow when ``show`` + a display exist).
    Returns the written path, or None if the frame/detector was unavailable.
    """
    import cv2

    from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrCamera
    from jiuwensymbiosis.perception.detector_client import init_detector

    frame = CruzrCamera(cfg).grab_head_frame()
    if frame is None:
        logger.error("grab_head_frame() returned None — no head frame (topics live? warmup?)")
        return None
    rgb, cloud, tf = frame
    if rgb is None:
        logger.error("head frame had no RGB")
        return None
    if cloud is None or tf is None:
        logger.warning("cloud=%s tf=%s — rendering RGB/mask only",
                       "ok" if cloud is not None else "None", "ok" if tf is not None else "None")

    seg = init_detector(detector_url)
    dets: list[dict] = []
    for role, name in (("target", object_name), ("reference", on)):
        if not name:
            continue
        best = None
        for r in seg(rgb, text_prompt=name):
            if r.get("score", 0.0) < 0.05:
                continue
            if best is None or r.get("score", 0.0) > best.get("score", 0.0):
                best = r
        if best is None:
            logger.info("%s %r: no detection", role, name)
            dets.append({"role": role, "name": name, "mask": None})
        else:
            dets.append({"role": role, "name": name, "mask": best.get("mask"),
                         "box": best.get("box"), "score": best.get("score")})

    min_valid = int(getattr(cfg, "head_cloud_min_valid_points", 30))
    img = render_alignment(rgb, cloud, tf, dets=dets, min_valid=min_valid)
    cv2.imwrite(out_path, img)
    logger.info("wrote %s  (%dx%d)", out_path, img.shape[1], img.shape[0])
    if show and os.environ.get("DISPLAY"):
        try:
            cv2.imshow("cruzr head cloud alignment", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as exc:  # noqa: BLE001 — headless / GUI-less is fine, PNG already saved
            logger.info("imshow skipped (%s)", exc)
    return out_path


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Cruzr head RGB↔mask↔cloud alignment debug view")
    p.add_argument("--config", default="configs/cruzr/cruzr.yaml", help="Cruzr YAML config")
    p.add_argument("--object", default="box", help="target noun (English) to detect")
    p.add_argument("--on", default="table", help="reference surface noun (English) to detect")
    p.add_argument("--out", default="/tmp/cruzr_head_align.png", help="output PNG path")
    p.add_argument("--detector-url", default="http://127.0.0.1:8114", help="detector service URL")
    p.add_argument("--show", action="store_true", help="also imshow (needs a display)")
    a = p.parse_args(argv)

    from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig

    cfg = CruzrConfig.from_yaml(a.config)
    out = grab_and_render(cfg, a.detector_url, a.object, a.on, a.out, show=a.show)
    return 0 if out else 1


if __name__ == "__main__":
    raise SystemExit(main())
