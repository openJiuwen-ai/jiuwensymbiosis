# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""把分割 mask + 深度 + 标定投影成 base 系物体 3D 几何（轴对齐包围盒）。

对任意被分割的物体成立（不限于箱体）：复用 utils/geometry 的针孔反投影/SE(3)，
base 系点单位 mm（与现有 detect 一致），跨度用分位数抗噪。front/top/back 为
base 系包围盒的对应面。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from jiuwensymbiosis.utils.geometry import apply_transform

_NORMAL_DBG_N = 0   # rolling counter for face-normal debug PNG filenames


@dataclass
class ObjectGeometry3D:
    ok: bool
    reason: str
    center_mm: tuple[float, float, float]
    width_mm: float
    height_mm: float
    front_x_mm: float
    top_z_mm: float
    n_points: int
    back_x_mm: float = 0.0  # far (q_hi) X face; with front_x gives the depth extent
    # Ground-plane oriented footprint (PCA of the base-frame x-y points).
    yaw_rad: float = 0.0    # long-axis orientation in the base frame, in (-π/2, π/2]
    long_mm: float = 0.0    # oriented extent along the long principal axis
    short_mm: float = 0.0   # oriented extent along the short principal axis
    # Robot-facing vertical-face normal in the ground plane (from the face's straight EDGE, aspect-free
    # — works for a cube; see face_normal_ground). Lets a base square up to the face instead of the
    # diagonal to the centre. ``face_flatness`` = edge_thickness/edge_length (small ⇒ a clean straight
    # face ⇒ trustworthy); ``1.0`` marks an untrustworthy normal (no vertical face / no clean edge /
    # a near-horizontal top surface).
    face_normal_x: float = 0.0
    face_normal_y: float = 0.0
    face_flatness: float = 1.0
    # Ground-plane NEAR-EDGE fit of the footprint boundary nearest the robot (the front lip of a table
    # top). Unlike face_normal_ground this needs NO vertical face, so it works on a horizontal table top:
    # (edge_mid_x_mm, edge_mid_y_mm) is the near edge's midpoint; (edge_normal_x, edge_normal_y) its unit
    # normal pointing toward the robot; ``edge_quality`` = residual_thickness/edge_length (small ⇒ a clean
    # straight edge ⇒ trustworthy, mirrors face_flatness); ``edge_len_mm`` the fitted edge length. Lets the
    # place-side base square to the table's near edge (see near_edge_line). ``1.0`` quality / zero normal
    # marks an untrustworthy fit (too few slices / short edge / centre at the origin).
    edge_mid_x_mm: float = 0.0
    edge_mid_y_mm: float = 0.0
    edge_normal_x: float = 0.0
    edge_normal_y: float = 0.0
    edge_quality: float = 1.0
    edge_len_mm: float = 0.0


_EMPTY = (0.0, 0.0, 0.0)


class NearEdge(NamedTuple):
    """``near_edge_line`` result: near-edge midpoint, robot-facing unit normal, fit quality.

    A NamedTuple (not a plain 7-tuple) so callers can unpack or index exactly as before while
    the fields carry their own names. ``ok=False`` marks an untrustworthy fit.
    """

    mid_x_mm: float
    mid_y_mm: float
    normal_x: float
    normal_y: float
    quality: float
    length_mm: float
    ok: bool


def face_normal_ground(
    pts_base: np.ndarray, *, inlier_tol_mm: float = 12.0, min_inlier_frac: float = 0.12,
    ransac_iters: int = 300, max_points: int = 2000, seed: int = 1234,
    debug: "dict | None" = None,
) -> tuple[float, float, float]:
    """Ground-plane heading of the box's robot-facing VERTICAL face normal — from the face's straight
    edge, NOT the footprint aspect, so it works even for a cube.

    The old approach PCA'd the WHOLE point cloud and took the smallest-variance axis as the plane
    normal. That only holds when a single flat face dominates the cloud: on an oblique/corner view
    (front + side + top all visible) the smallest axis is a blend of several planes → ``flatness≈1``,
    untrusted; and it can NEVER orient a near-cube (a square footprint carries no long-axis to key off).

    Instead, use that each vertical box face is a vertical plane whose GROUND TRACE is a straight LINE.
    Drop the horizontal top cap (it fills the footprint interior and has no edge), then RANSAC the
    dominant straight edge(s) of the ground-projected wall points and keep the NEAREST robot-facing
    edge — its perpendicular is the front-face normal, recovered from edge geometry alone (aspect-free).

    Returns ``(nx, ny, flatness)``: unit ground normal pointing toward the robot, and a quality metric
    ``flatness = edge_thickness / edge_length`` (small ⇒ a clean straight face ⇒ trustworthy; kept on
    the same ``≤ grasp_face_flatness_max`` gate the caller already uses). A horizontal-only patch (box
    or table TOP, no vertical face) or a cloud with no clean edge returns ``(0.0, 0.0, 1.0)`` so the
    caller falls back to a plain radial approach.
    """
    pts = np.asarray(pts_base, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3 or pts.shape[0] < 30:
        return 0.0, 0.0, 1.0
    if not np.all(np.isfinite(pts)):
        return 0.0, 0.0, 1.0
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    z_lo, z_hi = np.quantile(z, 0.05), np.quantile(z, 0.95)
    x_ext = float(np.quantile(x, 0.95) - np.quantile(x, 0.05))
    y_ext = float(np.quantile(y, 0.95) - np.quantile(y, 0.05))
    # Horizontal-only patch (box/table TOP): tiny vertical extent vs its ground spread → no vertical face.
    if (z_hi - z_lo) < 0.15 * max(x_ext, y_ext, 1e-6):
        return 0.0, 0.0, 1.0
    # Keep vertical-WALL points: drop the top cap (a horizontal slab filling the footprint interior that
    # would dilute the edge lines). The lower walls' ground trace IS the box edges.
    wall = pts[z <= z_hi - 0.15 * (z_hi - z_lo)]
    xy = wall[:, :2] if wall.shape[0] >= 30 else pts[:, :2]
    rng = np.random.default_rng(seed)
    if xy.shape[0] > max_points:                       # downsample: RANSAC is O(iters·N)
        xy = xy[rng.choice(xy.shape[0], max_points, replace=False)]
    n_pts = xy.shape[0]
    cxy = np.array([float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))])  # box centre (ground)
    if float(np.hypot(cxy[0], cxy[1])) < 1e-6:
        return 0.0, 0.0, 1.0
    u = cxy / float(np.hypot(cxy[0], cxy[1]))          # robot→box unit
    min_inliers = max(20, int(min_inlier_frac * n_pts))

    # RANSAC up to two dominant straight edges (remove inliers between rounds); keep the NEAREST
    # robot-facing one as the FRONT face.
    remaining = np.ones(n_pts, dtype=bool)
    best_face: tuple[float, float, float, float] | None = None   # (nx, ny, flatness, depth)
    best_edge = None                                             # inlier xy of the chosen (nearest) edge
    for _round in range(2):
        idx = np.flatnonzero(remaining)
        if idx.size < min_inliers:
            break
        sub = xy[idx]
        best_inl: np.ndarray | None = None
        best_cnt = 0
        for _ in range(ransac_iters):
            a, b = rng.integers(0, sub.shape[0], size=2)
            if a == b:
                continue
            d = sub[b] - sub[a]
            length = float(np.hypot(d[0], d[1]))
            if length < 5.0:
                continue
            d = d / length
            nrm = np.array([-d[1], d[0]])
            inl = np.abs((sub - sub[a]) @ nrm) < inlier_tol_mm
            cnt = int(inl.sum())
            if cnt > best_cnt:
                best_cnt, best_inl = cnt, inl
        if best_inl is None or best_cnt < min_inliers:
            break
        edge = sub[best_inl]
        e = edge - edge.mean(0)
        _ev, evec = np.linalg.eigh(np.cov(e, rowvar=False))
        tangent = evec[:, -1]                          # edge direction (largest spread)
        nrm = np.array([-tangent[1], tangent[0]])      # perpendicular = face normal
        if float(nrm @ (-u)) < 0.0:                    # orient outward (toward the robot/viewer)
            nrm = -nrm
        span = float(np.ptp(e @ tangent))
        thick = float(np.std(e @ nrm))
        flatness = min(thick / span, 1.0) if span > 1e-6 else 1.0
        depth = float(edge.mean(0) @ u)                # smaller ⇒ nearer the robot ⇒ the FRONT face
        if best_face is None or depth < best_face[3]:
            best_face = (float(nrm[0]), float(nrm[1]), flatness, depth)
            best_edge = edge
        remaining[idx[best_inl]] = False

    if debug is not None:                              # diagnostics for confirming the normal (viz/log)
        debug.update(wall_xy=xy, centre_xy=cxy,
                     edge_xy=(best_edge if best_edge is not None else np.empty((0, 2))),
                     normal=(np.array(best_face[:2]) if best_face is not None else np.zeros(2)),
                     flatness=(best_face[2] if best_face is not None else 1.0),
                     n_inliers=(0 if best_edge is None else int(best_edge.shape[0])),
                     edge_len_mm=(0.0 if best_edge is None else float(
                         np.ptp(best_edge[:, 0])) + float(np.ptp(best_edge[:, 1]))))
    if best_face is None:
        return 0.0, 0.0, 1.0
    return best_face[0], best_face[1], best_face[2]


def near_edge_line(
    pts_base: np.ndarray, *, n_bins: int = 24, min_slices: int = 6, min_span_mm: float = 150.0,
    debug: "dict | None" = None,
) -> NearEdge:
    """Fit the footprint's NEAR edge — the boundary nearest the robot (a table top's front lip) — and
    return its midpoint, outward normal (toward the robot) and a quality metric.

    Needs NO vertical face (``face_normal_ground`` bails on a horizontal table top): a filled top-surface
    footprint still has a clean front boundary. We slice the ground-projected points across the edge's
    tangential direction and, per slice, keep the point NEAREST the robot — that front contour IS the near
    edge — then total-least-squares fit a line through it.

    Returns ``(mid_x, mid_y, nx, ny, quality, edge_len_mm, ok)`` in mm / unit: the near-edge midpoint,
    the unit ground normal pointing toward the robot (sign-fixed like ``face_normal_ground`` /
    ``_grasp_near_face_normal``), ``quality`` = residual_thickness/edge_length (small ⇒ clean straight edge)
    and the fitted edge length. ``ok=False`` (zeros, quality ``1.0``) when the fit is untrustworthy: too
    few slices, an edge shorter than ``min_span_mm``, or a footprint centred at the origin.
    """
    fail = NearEdge(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, False)
    pts = np.asarray(pts_base, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2 or pts.shape[0] < 30:
        return fail
    if not np.all(np.isfinite(pts[:, :2])):
        return fail
    xy = pts[:, :2]
    c = np.array([float(np.median(xy[:, 0])), float(np.median(xy[:, 1]))])  # footprint centre (ground)
    c_norm = float(np.hypot(c[0], c[1]))
    if c_norm < 1e-6:
        return fail
    u = c / c_norm                                   # robot→centre (inward); the near edge faces −u
    t0 = np.array([-u[1], u[0]])                     # tangential (along the edge) direction
    s = xy @ t0                                       # tangential coordinate
    d = xy @ u                                        # depth coordinate (larger ⇒ deeper/farther)
    s_lo, s_hi = float(np.quantile(s, 0.05)), float(np.quantile(s, 0.95))
    if s_hi - s_lo < 1e-6:
        return fail
    edges = np.linspace(s_lo, s_hi, n_bins + 1)
    slot = np.clip(np.digitize(s, edges) - 1, 0, n_bins - 1)
    contour = []                                      # per-slice nearest-to-robot point = the front contour
    for b in range(n_bins):
        sel = np.flatnonzero(slot == b)
        if sel.size == 0:
            continue
        contour.append(xy[sel[np.argmin(d[sel])]])
    if len(contour) < min_slices:
        return fail
    contour = np.asarray(contour, dtype=np.float64)
    mid = contour.mean(0)                             # on the TLS line ⇒ the near edge's midpoint
    e = contour - mid
    _ev, evec = np.linalg.eigh(np.cov(e, rowvar=False))
    tangent = evec[:, -1]                            # edge direction (largest spread)
    nrm = np.array([-tangent[1], tangent[0]])        # perpendicular = edge normal
    if float(nrm @ (-mid)) < 0.0:                    # orient toward the robot (origin) — same convention
        nrm = -nrm                                    # as face_normal_ground / _grasp_near_face_normal
    span = float(np.ptp(e @ tangent))
    thick = float(np.std(e @ nrm))
    if span < min_span_mm:
        return fail
    quality = min(thick / span, 1.0) if span > 1e-6 else 1.0
    if debug is not None:
        debug.update(edge_contour_xy=contour, edge_mid_xy=mid, edge_normal=nrm,
                     ne_quality=quality, ne_len_mm=span)
    return NearEdge(float(mid[0]), float(mid[1]), float(nrm[0]), float(nrm[1]), quality, span, True)


def save_face_normal_topdown(debug: dict, out_path: str) -> bool:
    """Render a TOP-DOWN (bird's-eye) debug PNG to CONFIRM the edge-based face normal: the box's
    ground-projected wall points (grey), the RANSAC-fitted front edge (green), the box centre (red),
    the computed normal arrow toward the robot (cyan), and the robot at the origin (yellow, +x up).

    Eyeball check: the arrow should be ⊥ to the visible front edge (green) and point back at the robot.
    If the green edge is the wrong side, short, or the arrow is skewed, the normal is untrustworthy.
    No-op (returns False) if cv2 is unavailable or the debug dict has no points.
    """
    try:
        import cv2
    except Exception:
        return False
    wall = np.asarray(debug.get("wall_xy", np.empty((0, 2))), dtype=np.float64)
    if wall.shape[0] == 0:
        return False
    centre = np.asarray(debug.get("centre_xy", (0.0, 0.0)), dtype=np.float64).reshape(2)
    edge = np.asarray(debug.get("edge_xy", np.empty((0, 2))), dtype=np.float64)
    normal = np.asarray(debug.get("normal", (0.0, 0.0)), dtype=np.float64).reshape(2)
    h = w = 600
    mg = 40
    allp = np.vstack([wall, centre.reshape(1, 2), np.zeros((1, 2))])
    xmn, xmx = float(allp[:, 0].min()), float(allp[:, 0].max())
    ymn, ymx = float(allp[:, 1].min()), float(allp[:, 1].max())
    scale = min((w - 2 * mg) / max(ymx - ymn, 1.0), (h - 2 * mg) / max(xmx - xmn, 1.0))

    def px(p):
        return int(mg + (ymx - p[1]) * scale), int(mg + (xmx - p[0]) * scale)  # +y→left, +x→up

    img = np.full((h, w, 3), 30, dtype=np.uint8)
    for p in wall:
        cv2.circle(img, px(p), 1, (150, 150, 150), -1)
    for p in edge:
        cv2.circle(img, px(p), 2, (0, 220, 0), -1)                              # fitted front edge = green
    cv2.circle(img, px(centre), 5, (0, 0, 255), -1)                            # box centre = red
    cv2.arrowedLine(img, px(centre), px(centre + 250.0 * normal), (255, 255, 0), 2, tipLength=0.25)  # normal
    # Near-edge fit (near_edge_line): front contour = magenta, midpoint = orange, edge normal = orange arrow
    edge_contour = np.asarray(debug.get("edge_contour_xy", np.empty((0, 2))), dtype=np.float64)
    edge_mid = np.asarray(debug.get("edge_mid_xy", (0.0, 0.0)), dtype=np.float64).reshape(2)
    edge_normal = np.asarray(debug.get("edge_normal", (0.0, 0.0)), dtype=np.float64).reshape(2)
    for p in edge_contour:
        cv2.circle(img, px(p), 2, (255, 0, 255), -1)
    if float(np.hypot(edge_normal[0], edge_normal[1])) > 1e-6:
        cv2.circle(img, px(edge_mid), 5, (0, 165, 255), -1)
        cv2.arrowedLine(img, px(edge_mid), px(edge_mid + 250.0 * edge_normal), (0, 165, 255), 2, tipLength=0.25)
    cv2.circle(img, px((0.0, 0.0)), 6, (0, 255, 255), -1)                      # robot = yellow
    cv2.arrowedLine(img, px((0.0, 0.0)), px((150.0, 0.0)), (0, 255, 255), 2, tipLength=0.3)          # robot +x
    ang = float(np.degrees(np.arctan2(normal[1], normal[0])))
    e_ang = float(np.degrees(np.arctan2(edge_normal[1], edge_normal[0])))
    txt = "n=%.0fdeg flat=%.3f inl=%d edge=%.0fmm | edgeN=%.0fdeg q=%.3f len=%.0fmm" % (
        ang, float(debug.get("flatness", 1.0)), int(debug.get("n_inliers", 0)),
        float(debug.get("edge_len_mm", 0.0)),
        e_ang, float(debug.get("ne_quality", 1.0)), float(debug.get("ne_len_mm", 0.0)))
    cv2.putText(img, txt, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    try:
        return bool(cv2.imwrite(out_path, img))
    except Exception:
        return False


def oriented_footprint(xy: np.ndarray, *, q_lo: float = 0.05, q_hi: float = 0.95) -> tuple[float, float, float]:
    """Ground-plane oriented footprint of base-frame ``xy`` points (N,2) via PCA.

    Returns ``(yaw_rad, long_mm, short_mm)``: the **long-axis** heading in the base frame
    (normalised to ``(-π/2, π/2]`` — a box axis is 180°-ambiguous) and the oriented spans
    (``q_lo..q_hi`` quantile, noise-robust) along the long/short principal axes. Degenerate
    (near-collinear) inputs still yield a valid axis from the covariance eigenvectors.
    """
    xy = np.asarray(xy, dtype=np.float64)
    d = xy - xy.mean(0)
    cov = np.cov(d, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return 0.0, 0.0, 0.0
    _evals, evecs = np.linalg.eigh(np.atleast_2d(cov))  # ascending eigenvalues
    long_axis, short_axis = evecs[:, -1], evecs[:, 0]

    def span(proj) -> float:
        return float(np.quantile(proj, q_hi) - np.quantile(proj, q_lo))

    long_mm, short_mm = span(d @ long_axis), span(d @ short_axis)
    yaw = float(np.arctan2(long_axis[1], long_axis[0]))
    if yaw > np.pi / 2:
        yaw -= np.pi
    elif yaw <= -np.pi / 2:
        yaw += np.pi
    return yaw, long_mm, short_mm


def _resize_mask_nearest(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    if mask.shape[:2] == (h, w):
        return mask.astype(bool)
    ys = (np.arange(h) * (mask.shape[0] / h)).astype(int).clip(0, mask.shape[0] - 1)
    xs = (np.arange(w) * (mask.shape[1] / w)).astype(int).clip(0, mask.shape[1] - 1)
    return mask[np.ix_(ys, xs)].astype(bool)


def object_geometry_from_mask(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    tf_base_cam: np.ndarray,
    *,
    q_lo: float = 0.05,
    q_hi: float = 0.95,
    min_points: int = 50,
    min_z_mm: "float | None" = None,
    debug_label: str = "",
) -> ObjectGeometry3D:
    """Back-project masked, valid-depth pixels to base frame and summarize box geometry.

    ``min_z_mm`` (grounded 'X on Y' detection): compute the FACE NORMAL only from points ABOVE this
    height (the reference surface top). The target's mask often bleeds onto the reference surface; the
    edge-based normal would then latch onto the reference's dominant straight edge, giving a normal for
    the REFERENCE, not the target. Dropping surface points keeps the fit on the target's own wall.
    """
    h, w = depth_m.shape[:2]
    m = _resize_mask_nearest(mask, h, w)
    valid = m & (depth_m > 0) & np.isfinite(depth_m)
    n = int(valid.sum())
    if n < min_points:
        return ObjectGeometry3D(False, "too_few_points", _EMPTY, 0.0, 0.0, 0.0, 0.0, n)

    ys, xs = np.where(valid)
    z_mm = depth_m[ys, xs].astype(np.float64) * 1000.0
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    ppx, ppy = intrinsics[0, 2], intrinsics[1, 2]
    x_mm = (xs - ppx) * z_mm / fx
    y_mm = (ys - ppy) * z_mm / fy
    pts_cam = np.stack([x_mm, y_mm, z_mm], axis=1)  # (N,3)
    pts_base = apply_transform(np.asarray(tf_base_cam, dtype=np.float64), pts_cam)  # (N,3) mm

    bx, by, bz = pts_base[:, 0], pts_base[:, 1], pts_base[:, 2]
    x_lo, x_hi = np.quantile(bx, q_lo), np.quantile(bx, q_hi)
    y_lo, y_hi = np.quantile(by, q_lo), np.quantile(by, q_hi)
    z_lo, z_hi = np.quantile(bz, q_lo), np.quantile(bz, q_hi)
    center = (float(np.median(bx)), float(np.median(by)), float(np.median(bz)))
    yaw_rad, long_mm, short_mm = oriented_footprint(pts_base[:, :2], q_lo=q_lo, q_hi=q_hi)
    # Face normal from the TARGET's own wall: when grounded on a surface, drop points at/below the
    # reference surface top so the edge fit is the target's, not the reference's dominant edge.
    fn_pts = pts_base if min_z_mm is None else pts_base[pts_base[:, 2] > float(min_z_mm)]
    # Optional face-normal confirmation: set JIUWEN_CRUZR_NORMAL_DEBUG_DIR to a dir to dump a top-down
    # PNG per detection (footprint + fitted front edge + normal arrow + robot) — see save_face_normal_topdown.
    _dbg_dir = os.environ.get("JIUWEN_CRUZR_NORMAL_DEBUG_DIR")
    _dbg: dict | None = {} if _dbg_dir else None
    nfx, nfy, flatness = face_normal_ground(fn_pts, debug=_dbg)
    emx, emy, enx, eny, e_qual, e_len, _e_ok = near_edge_line(fn_pts, debug=_dbg)
    if _dbg_dir and _dbg:
        global _NORMAL_DBG_N
        _NORMAL_DBG_N += 1
        try:
            os.makedirs(_dbg_dir, exist_ok=True)
            _lbl = "".join(c if c.isalnum() else "_" for c in debug_label)[:20] or "obj"
            save_face_normal_topdown(_dbg, os.path.join(
                _dbg_dir, "face_normal_%s_%03d.png" % (_lbl, _NORMAL_DBG_N % 100)))
        except Exception:  # noqa: BLE001 — debug dump must never break detection
            pass
    return ObjectGeometry3D(
        ok=True, reason="", center_mm=center,
        width_mm=float(y_hi - y_lo), height_mm=float(z_hi - z_lo),
        front_x_mm=float(x_lo), top_z_mm=float(z_hi), n_points=n,
        back_x_mm=float(x_hi), yaw_rad=yaw_rad, long_mm=long_mm, short_mm=short_mm,
        face_normal_x=nfx, face_normal_y=nfy, face_flatness=flatness,
        edge_mid_x_mm=emx, edge_mid_y_mm=emy, edge_normal_x=enx, edge_normal_y=eny,
        edge_quality=e_qual, edge_len_mm=e_len,
    )
