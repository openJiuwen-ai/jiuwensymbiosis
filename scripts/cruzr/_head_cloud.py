# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Cruzr-only: turn a head detection mask + the head stereo point cloud into a base-frame
footprint geometry, so the wide-FOV head camera can verify the "target on reference surface"
relation in 3-D during search (reuses the shared ``_on_surface`` predicate).

Pure numpy, offline-testable. The head point cloud
(``/sensor/camera/stereo/pointcloud/jazzy``) is row-major per-pixel, in METERS, in the same optical
frame as the head detection RGB, so a full-res mask maps onto it by uniform downsampling. It is
normally ORGANIZED (H×W×3); when it arrives UNORGANIZED (height=1 → ``(1, N, 3)``) but its point
count matches the RGB resolution we reshape it back to the image grid (see ``rgb_hw`` below) —
otherwise the mask would collapse onto row 0. Holes on untextured surfaces come through as NaN and
are dropped. Coordinates are
converted to millimetres (×1000) before applying ``tf_base_cam`` (whose translation is already in
mm) so the result is the same scale as the waist pixel-reprojected geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class HeadGeom:
    """Base-frame (millimetres) footprint of a masked region, from the head point cloud.

    ``center_*`` is the point-set centroid (used as a candidate's position for on-surface tests);
    ``front_x``/``back_x``/``half_width``/``top_z`` describe the footprint (used as a reference
    surface's extent). ``n_valid`` is the number of finite in-mask cloud points that fed it."""

    center_x: float
    center_y: float
    center_z: float
    front_x: float
    back_x: float
    half_width: float
    top_z: float
    n_valid: int


def head_geometry_from_mask(mask: Any, cloud_xyz: Any, tf_base_cam: Any, *,
                            min_valid_points: int,
                            rgb_hw: Optional[tuple[int, int]] = None) -> Optional[HeadGeom]:
    """Project a full-resolution detection ``mask`` onto the organized ``cloud_xyz`` (H×W×3,
    metres, optical frame), keep the finite in-mask points, transform them to the base frame
    (millimetres) via ``tf_base_cam``, and summarize as a :class:`HeadGeom`.

    ``rgb_hw`` = the detection RGB's ``(height, width)``. A stereo cloud published UNORGANIZED
    (``height=1`` → arriving here as ``(1, N, 3)`` or a bare ``(N, 3)`` list) is still row-major
    per-pixel; when ``N`` matches the RGB resolution we reshape it back onto the image grid so the
    mask maps correctly. Without this a flat cloud collapses onto image row 0 and yields ~0 in-mask
    points — a spurious "too few valid cloud points" degrade even though the geometry was available.

    Returns ``None`` (the degrade signal → caller falls back to bearing-only) when the cloud or TF
    is missing, the cloud is malformed, or fewer than ``min_valid_points`` finite in-mask points
    survive (untextured NaN holes)."""
    if cloud_xyz is None or tf_base_cam is None:
        return None
    mask = np.asarray(mask)
    cloud = np.asarray(cloud_xyz, dtype=np.float64)
    tf = np.asarray(tf_base_cam, dtype=np.float64)
    if mask.ndim != 2:
        return None
    # Recover an organized H×W×3 grid from a flat/unorganized cloud when its point count matches the
    # RGB (see rgb_hw above) — this is the head "cloud size ≠ image size" mismatch that otherwise
    # zeroes out the in-mask points.
    if rgb_hw is not None and cloud.ndim in (2, 3) and cloud.shape[-1] >= 3:
        rh, rw = int(rgb_hw[0]), int(rgb_hw[1])
        n_pts = int(cloud.size // cloud.shape[-1])
        is_flat = cloud.ndim == 2 or cloud.shape[0] == 1
        if is_flat and rh > 0 and rw > 0 and rh * rw == n_pts:
            cloud = cloud.reshape(rh, rw, cloud.shape[-1])
    if cloud.ndim != 3 or cloud.shape[2] < 3:
        return None
    ch, cw = int(cloud.shape[0]), int(cloud.shape[1])
    mh, mw = int(mask.shape[0]), int(mask.shape[1])
    if ch == 0 or cw == 0 or mh == 0 or mw == 0:
        return None

    # Downsample the full-res mask onto the cloud grid (uniform nearest; registration = risk ①).
    rows = np.minimum(np.arange(ch) * mh // ch, mh - 1)
    cols = np.minimum(np.arange(cw) * mw // cw, mw - 1)
    mask_ds = mask[np.ix_(rows, cols)].astype(bool)   # (ch, cw)

    xyz = cloud[..., :3]
    finite = np.isfinite(xyz).all(axis=2)             # drop NaN holes on untextured faces
    sel = mask_ds & finite
    n_valid = int(sel.sum())
    if n_valid < int(min_valid_points):
        return None

    pts_mm = xyz[sel] * 1000.0                          # metres → millimetres (matches tf/base scale)
    homog = np.concatenate([pts_mm, np.ones((n_valid, 1))], axis=1)   # (N, 4)
    base = (tf @ homog.T).T[:, :3]                      # (N, 3) base-frame millimetres
    xs, ys, zs = base[:, 0], base[:, 1], base[:, 2]
    return HeadGeom(
        center_x=float(xs.mean()), center_y=float(ys.mean()), center_z=float(zs.mean()),
        front_x=float(xs.min()), back_x=float(xs.max()),
        half_width=float((ys.max() - ys.min()) / 2.0),
        top_z=float(zs.max()), n_valid=n_valid,
    )
