"""Observable-surface association; labels never determine physical identity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

from jiuwensymbiosis.spatial_memory.models import Detection, Observation


@dataclass(frozen=True)
class MemoryPolicy:
    voxel_m: float = 0.03
    surface_distance_m: float = 0.09
    min_points: int = 12
    max_points: int = 4096
    min_overlap: float = 0.30
    min_score: float = 0.40
    ambiguity_margin: float = 0.12
    confirmation_translation_m: float = 0.05
    confirmation_rotation_rad: float = 0.15
    appearance_weight: float = 0.15

    def __post_init__(self) -> None:
        for name in ("voxel_m", "surface_distance_m", "confirmation_translation_m", "confirmation_rotation_rad"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Invalid policy {name}")
        for name in ("min_overlap", "min_score", "ambiguity_margin"):
            if not np.isfinite(getattr(self, name)) or not 0 < getattr(self, name) < 1:
                raise ValueError(f"Invalid policy {name}")
        if not 3 <= self.min_points <= self.max_points or self.max_points > 100_000:
            raise ValueError("Invalid point budget")
        if not np.isfinite(self.appearance_weight) or not 0 <= self.appearance_weight <= 1:
            raise ValueError("Invalid appearance weight")


def downsample(points: np.ndarray, policy: MemoryPolicy) -> np.ndarray:
    if points.size == 0:
        return points
    _, indices = np.unique(np.floor(points / policy.voxel_m).astype(np.int64), axis=0, return_index=True)
    selected = points[np.sort(indices)]
    if len(selected) > policy.max_points:
        selected = selected[np.linspace(0, len(selected) - 1, policy.max_points, dtype=int)]
    return selected


def project(
    observation: Observation, detection: Detection, policy: MemoryPolicy
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Reject mixed boundary pixels; sparse depth stays sparse."""
    if detection.mask.shape != observation.depth_m.shape:
        raise ValueError("Mask must match original RGB-D dimensions")
    inside = binary_erosion(detection.mask, iterations=1)
    v, u = np.nonzero(inside & np.isfinite(observation.depth_m))
    z = observation.depth_m[v, u]
    quality = {
        "mask_pixels": int(detection.mask.sum()),
        "interior_pixels": int(inside.sum()),
        "valid_depth_pixels": len(z),
        "depth_coverage_fraction": len(z) / max(1, int(inside.sum())),
        "retained_depth_pixels": 0,
        "surface_voxels": 0,
        "depth_range_m": [float(z.min()), float(z.max())] if len(z) else None,
        "material_depth_verified": False,
    }
    if len(z) < policy.min_points:
        return np.empty((0, 3)), np.zeros(24), quality
    lo, hi = np.quantile(z, [0.02, 0.98])
    keep = (z >= lo) & (z <= hi)
    u, v, z = u[keep], v[keep], z[keep]
    k, t = observation.intrinsics, observation.map_from_camera
    camera = np.column_stack(((u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z))
    points = downsample(camera @ t[:3, :3].T + t[:3, 3], policy)
    quality.update(retained_depth_pixels=len(z), surface_voxels=len(points))
    colors = observation.rgb[v, u]
    histogram = np.concatenate(
        [np.histogram(colors[:, channel], bins=8, range=(0, 256))[0] for channel in range(3)]
    ).astype(float)
    histogram /= np.linalg.norm(histogram)
    return points, histogram, quality


def surface_in_view(
    points: np.ndarray, observation: Observation, *, occlusion_margin_m: float | None = None
) -> np.ndarray:
    """Exclude remembered points outside the current camera, never outside a mask.

    Cropping to a detection mask would confuse small contained objects with their
    parent. Optional depth testing removes only independently measured foreground occlusion.
    Missing depth remains unknown; this never reconstructs hidden faces.
    """
    transform, k = observation.map_from_camera, observation.intrinsics
    camera = (points - transform[:3, 3]) @ transform[:3, :3]
    front = camera[:, 2] > 1e-6
    selected = points[front]
    camera = camera[front]
    pixels = camera[:, :2] / camera[:, 2, None] * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]
    height, width = observation.depth_m.shape
    inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    selected = selected[inside]
    if occlusion_margin_m is not None:
        # Only independent scene depth can establish occlusion. Cropping to a
        # detection mask instead would falsely merge a small object with its parent.
        uv = np.floor(pixels[inside]).astype(int)
        measured = observation.depth_m[uv[:, 1], uv[:, 0]]
        occluded = np.isfinite(measured) & (measured > 0) & (camera[inside, 2] > measured + occlusion_margin_m)
        selected = selected[~occluded]
    return np.asarray(selected)


def compare_surfaces(
    points: np.ndarray, histogram: np.ndarray, entity_points: np.ndarray, gallery: list, policy: MemoryPolicy
) -> dict:
    """Expose rejected geometric candidates without relaxing identity gates."""
    gap = np.maximum(0, np.maximum(points.min(0) - entity_points.max(0), entity_points.min(0) - points.max(0)))
    evidence: dict = {"score": 0.0, "bbox_gap_m": float(np.linalg.norm(gap)), "overlap": None}
    if np.linalg.norm(gap) > policy.surface_distance_m:
        return {**evidence, "rejection": "separated_bounds"}
    forward = np.mean(cKDTree(entity_points).query(points)[0] < policy.surface_distance_m)
    reverse = np.mean(cKDTree(points).query(entity_points)[0] < policy.surface_distance_m)
    # A sofa mask can contain all points of a small pillow. Symmetric overlap
    # rejects that containment; a one-sided maximum would call it a perfect match.
    overlap = float(2 * forward * reverse / (forward + reverse)) if forward + reverse else 0.0
    evidence.update(overlap=overlap, forward_overlap=float(forward), reverse_overlap=float(reverse))
    if overlap < policy.min_overlap:
        return {**evidence, "rejection": "insufficient_surface_overlap"}
    invalid_gallery = {**evidence, "rejection": "invalid_appearance_gallery"}
    try:
        features = np.asarray(gallery, dtype=float)
    except (TypeError, ValueError):
        return invalid_gallery
    if features.ndim != 2 or features.size == 0 or features.shape[1:] != histogram.shape:
        return invalid_gallery
    if not np.isfinite(features).all():
        return invalid_gallery
    appearance = max(float(np.dot(histogram, row)) for row in features)
    weight = policy.appearance_weight
    return {**evidence, "score": (1 - weight) * overlap + weight * appearance, "appearance_similarity": appearance}


def different_view(first: list, current: np.ndarray, policy: MemoryPolicy) -> bool:
    first_array = np.asarray(first)
    translation = np.linalg.norm(first_array[:3, 3] - current[:3, 3])
    angle = np.arccos(np.clip((np.trace(first_array[:3, :3].T @ current[:3, :3]) - 1) / 2, -1, 1))
    return bool(translation >= policy.confirmation_translation_m or angle >= policy.confirmation_rotation_rad)
