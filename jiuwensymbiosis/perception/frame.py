# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Generic camera-frame container + pixel→base projection (robot-agnostic).

``CameraFrame`` holds one grabbed frame (rgb + optional depth/intrinsics/extrinsics).
``project_to_base`` back-projects a pixel (u,v)+depth to camera XYZ then maps it to
the base frame — pure pinhole + SE(3) over ``utils.geometry``. Any adapter's camera
driver returns a ``CameraFrame`` and reuses ``project_to_base``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from jiuwensymbiosis.utils.geometry import apply_transform, pixel_and_depth_to_camera_xyz


class FrameValidationError(RuntimeError):
    """A captured frame cannot establish a current target location."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def project_to_base(
    uv: tuple[float, float],
    depth_m: float,
    intrinsics: np.ndarray,
    tf_base_cam: np.ndarray,
) -> np.ndarray:
    """Back-project pixel (u,v)+depth to camera XYZ (mm), then map to base via ``tf_base_cam``."""
    xyz_cam = pixel_and_depth_to_camera_xyz(uv, depth_m, np.asarray(intrinsics, dtype=np.float64))
    return apply_transform(np.asarray(tf_base_cam, dtype=np.float64), xyz_cam)


@dataclass
class CameraFrame:
    """One grabbed frame; depth/intrinsics/extrinsics may be None when unavailable.

    RGB-D target association requires a unique ``frame_id`` and acquisition time
    on the host monotonic clock. ``tf_base_cam`` maps camera to base (translation
    in mm); depth is metres. ``metadata['base_sample']`` is measured at capture:
    ``position_m`` is a finite xyz vector in a stable reference, ``body_yaw_rad``
    is finite yaw in that same reference, and ``odom_frame`` identifies its epoch.
    Optional ``scene_instance_id`` identifies the scene/reset epoch independently.
    At least one reference ID is required. A scene-only reference is valid only
    if base_sample uses its stable coordinates and its ID changes on every reset.
    When both IDs exist, neither overrides the other; any change, appearance or
    disappearance invalidates association. Never supply object ground truth here.
    """

    rgb: np.ndarray
    depth_m: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    tf_base_cam: np.ndarray | None = None
    frame_id: str | None = None
    captured_monotonic_s: float | None = None
    metadata: dict = field(default_factory=dict)


def check_observed_rotation(
    frame: CameraFrame, turn_rad: float, body_bounds_m: np.ndarray | None, clearance_m: float
) -> str | None:
    """Conservatively certify a full-body yaw envelope from this RGB-D frame.

    Bounds describe the robot itself in forward/left/up metres, including its
    current upper-body posture. They must not come from a scene obstacle query.
    A circumscribed square prism encloses every intermediate rotation. All of
    its projection must have measured depth beyond the entire prism. Missing
    pixels, occlusion (including the robot itself) and rear blind spots cannot
    certify free space. This deliberately over-rejects; it never fills sparse
    depth, exempts an unobserved bounding-box interior, or assumes a circular
    robot. The caller validates frame freshness immediately before execution.
    """
    if not math.isfinite(turn_rad) or not math.isfinite(clearance_m) or clearance_m < 0:
        raise ValueError("Rotation and clearance must be finite, with nonnegative clearance")
    if turn_rad == 0:
        return None
    if body_bounds_m is None:
        return "body_geometry_unavailable"
    bounds = np.asarray(body_bounds_m, dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        return "body_geometry_unavailable"
    if frame.depth_m is None or frame.intrinsics is None or frame.tf_base_cam is None:
        return "turn_clearance_unobserved"
    depth = np.asarray(frame.depth_m)
    k = np.asarray(frame.intrinsics, dtype=float)
    transform = np.asarray(frame.tf_base_cam, dtype=float)
    if depth.ndim != 2 or min(depth.shape) < 2 or depth.shape != frame.rgb.shape[:2]:
        return "turn_clearance_unobserved"
    if k.shape != (3, 3) or transform.shape != (4, 4):
        return "turn_clearance_unobserved"
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        return "turn_clearance_unobserved"
    if not np.isfinite(transform).all() or not np.allclose(k[2], [0, 0, 1]):
        return "turn_clearance_unobserved"
    if not np.allclose(transform[3], [0, 0, 0, 1]):
        return "turn_clearance_unobserved"
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        return "turn_clearance_unobserved"
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        return "turn_clearance_unobserved"
    radius = float(np.linalg.norm(np.max(np.abs(bounds[:, :2]), axis=0))) + clearance_m
    # A rotation does not move the ground-contact surface downwards. Retain
    # the bottom bound, and reserve clearance above the actual top surface.
    corners = np.array(list(product((-radius, radius), (-radius, radius), (bounds[0, 2], bounds[1, 2] + clearance_m))))
    camera = (corners - transform[:3, 3] / 1000) @ transform[:3, :3]
    if np.any(camera[:, 2] <= 0):
        return "turn_clearance_unobserved"
    pixels = camera @ k.T
    pixels = pixels[:, :2] / pixels[:, 2:]
    if not np.isfinite(pixels).all():
        return "turn_clearance_unobserved"
    height, width = depth.shape
    if np.any(pixels < 0) or np.any(pixels >= [width - 1, height - 1]):
        return "turn_clearance_unobserved"
    lo = np.floor(pixels.min(axis=0)).astype(int)
    hi = np.ceil(pixels.max(axis=0)).astype(int)
    rows = slice(lo[1], hi[1] + 1)
    columns = slice(lo[0], hi[0] + 1)
    measured = depth[rows, columns]
    # Checking the complete projected rectangle against its farthest Z is a
    # sufficient, deliberately conservative certificate for the whole prism.
    if not np.isfinite(measured).all() or np.any(measured <= camera[:, 2].max() + 0.02):
        return "turn_clearance_unobserved"
    return None


def check_observed_corridor(
    frame: CameraFrame,
    forward_m: float,
    body_radius_m: float,
    clearance_m: float,
    *,
    ground_z_m: float | None = None,
) -> str | None:
    """Reject measured obstructions in a short forward swept envelope.

    This is an observed-obstacle guard, not a global free-space certificate.
    It never consults simulation entities or an oracle map. Unobserved regions
    remain outside the local approach's supported operating envelope.
    """
    if forward_m <= 0:
        return None
    if frame.depth_m is None or frame.intrinsics is None or frame.tf_base_cam is None:
        return "clearance_unavailable"
    depth = frame.depth_m
    ys, xs = np.nonzero(np.isfinite(depth) & (depth > 0))
    if len(xs) < 50:
        return "clearance_unavailable"
    z = depth[ys, xs]
    k = frame.intrinsics
    camera_points = np.stack(((xs - k[0, 2]) * z / k[0, 0], (ys - k[1, 2]) * z / k[1, 1], z), axis=1)
    transform = frame.tf_base_cam
    points = camera_points @ transform[:3, :3].T + transform[:3, 3] / 1000
    if ground_z_m is None:
        # Base-frame origins differ between bodies. Estimate the dominant, level
        # support below the base from measured depth instead of hardcoding a USD
        # floor height. This local policy supports a level indoor floor only.
        low = points[(points[:, 2] >= -0.4) & (points[:, 2] <= 0.05)]
        if len(low) < 100:
            return "ground_unobserved"
        bins = np.rint(low[:, 2] / 0.01).astype(int)
        heights, counts = np.unique(bins, return_counts=True)
        level = heights[np.argmax(counts)] * 0.01
        support = low[np.abs(low[:, 2] - level) <= 0.015]
        if len(support) < 100 or np.ptp(support[:, 0]) < 0.5 or np.ptp(support[:, 1]) < 0.8:
            return "ground_unobserved"
        ground_z_m = float(np.median(support[:, 2]))
    ahead = (points[:, 0] > body_radius_m) & (np.abs(points[:, 1]) < body_radius_m + clearance_m)
    if np.count_nonzero(ahead) < 20:
        return "clearance_unavailable"
    # A translating circular body sweeps a capsule. A bounding rectangle would
    # incorrectly reject points near its corners even outside the swept body.
    outside_body = np.hypot(points[:, 0], points[:, 1]) > body_radius_m
    swept_distance = np.hypot(np.maximum(points[:, 0] - forward_m, 0), points[:, 1])
    obstacle = (
        outside_body & (points[:, 0] > 0) & (points[:, 2] > ground_z_m + 0.05) & (points[:, 2] < ground_z_m + 1.8)
    )
    if np.any(obstacle & (swept_distance < body_radius_m + clearance_m)):
        return "depth_obstacle_blocked"
    # The next swept strip must have measured support throughout. A few far-away
    # points do not establish clearance in an unobserved near-field or over a gap.
    floor = points[np.abs(points[:, 2] - ground_z_m) <= 0.02, :2]
    end = body_radius_m + clearance_m + forward_m
    gx, gy = np.meshgrid(
        np.linspace(0, end, max(2, math.ceil(end / 0.1) + 1)),
        np.linspace(
            -body_radius_m - clearance_m,
            body_radius_m + clearance_m,
            max(2, math.ceil(2 * (body_radius_m + clearance_m) / 0.1) + 1),
        ),
    )
    for point in np.c_[gx.ravel(), gy.ravel()]:
        if (
            np.linalg.norm(point) <= body_radius_m + clearance_m
            or np.linalg.norm(point - [forward_m, 0.0]) > body_radius_m + clearance_m
        ):
            continue  # Only newly swept support; the currently occupied footprint is stationary.
        if floor.size == 0 or np.min(np.linalg.norm(floor - point, axis=1)) > 0.1:
            return "ground_coverage_incomplete"
    return None
