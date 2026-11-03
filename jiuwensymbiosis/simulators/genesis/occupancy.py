"""Offline geometry maps in the same metre/Z-up frame as Genesis room meshes.

These are evaluator maps, not sensor observations or proof of traversability.
Known free space requires both a room polygon and an authored floor surface.
The runtime planner is authoritative: ``navigation.NavigationMap`` builds its
own floor/obstacle bands from the same meshes (support within 0.015 m of the
floor level, obstacles spanning +0.06 m to the band top). This export mirrors
those band edges — the export script's defaults are 0.06/1.95 for exactly that
reason — but derives its floor level by area vote instead, so an disagreement
between the two is a defect in one of them, never a tiebreak for planning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .indoor import RoomMesh


@dataclass
class OccupancyMap:
    """Arrays use row zero at minimum world Y; image exporters must flip vertically."""

    cells: np.ndarray  # int8: unknown=-1, free=0, occupied=100
    resolution: float
    origin: tuple[float, float]
    floor_z: float
    metadata: dict


def _clip_height(polygon: np.ndarray, z: float, *, above: bool) -> np.ndarray:
    result = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        inside_a = a[2] >= z if above else a[2] <= z
        inside_b = b[2] >= z if above else b[2] <= z
        if inside_a:
            result.append(a)
        if inside_a != inside_b:
            result.append(a + (b - a) * ((z - a[2]) / (b[2] - a[2])))
    return np.asarray(result).reshape(-1, 3)


def _paint(mask: np.ndarray, polygons, origin: np.ndarray, resolution: float) -> None:
    import cv2

    # Fill each polygon separately: cv2.fillPoly(list_of_polygons) uses an
    # even/odd rule that can erase overlapping triangles instead of unioning them.
    for polygon in polygons:
        points = np.floor((np.asarray(polygon)[:, :2] - origin) / resolution).astype(np.int32)
        cv2.fillConvexPoly(mask, points, 1)


def build_occupancy(
    meshes: list[RoomMesh],
    rooms: list[dict],
    *,
    resolution: float = 0.05,
    min_height: float = 0.06,
    max_height: float = 1.95,
    floor_tolerance: float = 0.03,
    padding: float = 0.5,
) -> OccupancyMap:
    """Project height-clipped triangles, preserving thin walls and unknown space.

    The dominant exposed floor level is area-weighted. Other levels remain
    unknown unless within floor_tolerance. Geometry above max_height is excluded;
    closed projected silhouettes of each watertight mesh are filled conservatively.
    Room metadata only bounds known space; it never invents supporting geometry.
    """
    import cv2
    from scipy.ndimage import binary_fill_holes

    values = (resolution, min_height, max_height, floor_tolerance, padding)
    if not all(np.isfinite(values)) or not (resolution > 0 and 0 <= min_height < max_height):
        raise ValueError("Invalid grid resolution or obstacle height interval")
    if not meshes or not rooms:
        raise ValueError("Require meshes, rooms, nonnegative floor tolerance and at least one padding cell")
    if floor_tolerance < 0 or padding < resolution:
        raise ValueError("Require meshes, rooms, nonnegative floor tolerance and at least one padding cell")
    polygons = [np.asarray(room["polygon"], dtype=float) for room in rooms]
    if any(p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all() for p in polygons):
        raise ValueError("Room polygons must contain finite XY coordinates")
    points = np.concatenate(polygons)
    low = np.minimum(points.min(axis=0), np.min([m.mesh.bounds[0, :2] for m in meshes], axis=0))
    high = np.maximum(points.max(axis=0), np.max([m.mesh.bounds[1, :2] for m in meshes], axis=0))
    origin = np.floor((low - padding) / resolution) * resolution
    width, height = np.ceil((high + padding - origin) / resolution).astype(int)
    if width * height > 16_000_000:
        raise ValueError("Map exceeds 16 million cells; increase resolution")

    floor_triangles = []
    for item in meshes:
        if any(part.lower() == "floor" or part.lower().startswith("floor_") for part in item.path.split("/")):
            tri = item.mesh.triangles
            horizontal = np.ptp(tri[:, :, 2], axis=1) < 0.005
            # For a solid slab, the lower face cannot support a robot above it.
            # Open floor sheets can have either winding and remain valid.
            if item.mesh.is_watertight:
                horizontal &= item.mesh.face_normals[:, 2] > 0.5
            floor_triangles.extend(tri[horizontal])
    if not floor_triangles:
        raise ValueError("No authored horizontal floor geometry")
    floors = np.asarray(floor_triangles)
    areas = np.linalg.norm(np.cross(floors[:, 1] - floors[:, 0], floors[:, 2] - floors[:, 0]), axis=1) / 2
    levels = floors[:, :, 2].mean(axis=1)
    bins, inverse = np.unique(np.round(levels, 2), return_inverse=True)
    room_mask = np.zeros((height, width), np.uint8)
    for poly in polygons:
        contour = np.floor((poly - origin) / resolution).astype(np.int32)
        cv2.fillPoly(room_mask, [contour], 1)

    # Vote by exposed XY area, not summed triangle area. Structural slabs can
    # be larger than the finished floor above them, and duplicate finish layers
    # must not receive extra votes. Sorted layers retain the top surface per cell.
    top_level = np.full(room_mask.shape, -1, np.int32)
    elevations = []
    for index in range(len(bins)):
        selected = inverse == index
        elevations.append(float(np.average(levels[selected], weights=areas[selected])))
        mask = np.zeros_like(room_mask)
        _paint(mask, floors[selected], origin, resolution)
        top_level[mask != 0] = index
    valid = (room_mask != 0) & (top_level >= 0)
    weights = np.bincount(top_level[valid], minlength=len(bins)) * resolution**2
    if not weights.any():
        raise ValueError("Room polygons do not overlap authored floor geometry")
    floor_z = elevations[int(np.argmax(weights))]
    support = np.zeros_like(room_mask)
    for index, elevation in enumerate(elevations):
        if abs(elevation - floor_z) <= floor_tolerance:
            support[top_level == index] = 1
    # Erode once to avoid calling cells barely touching a floor edge known free.
    support = cv2.erode(support, np.ones((3, 3), np.uint8))

    obstacles = np.zeros_like(support, dtype=np.uint8)
    z_low, z_high = floor_z + min_height, floor_z + max_height
    used = clipped = filled = 0
    for item in meshes:
        bounds = item.mesh.bounds
        if bounds[1, 2] < z_low or bounds[0, 2] > z_high:
            continue
        tri = item.mesh.triangles
        lo, hi = tri[:, :, 2].min(axis=1), tri[:, :, 2].max(axis=1)
        overlap = (hi >= z_low) & (lo <= z_high)
        inside = (lo >= z_low) & (hi <= z_high)
        projected = np.zeros_like(support, dtype=np.uint8)
        _paint(projected, tri[inside], origin, resolution)
        for triangle in tri[overlap & ~inside]:
            polygon = _clip_height(triangle, z_low, above=True)
            polygon = _clip_height(polygon, z_high, above=False)
            if len(polygon):
                _paint(projected, [polygon], origin, resolution)
            clipped += 1
        if item.mesh.is_watertight:
            solid = binary_fill_holes(projected)
            filled += int(np.count_nonzero(solid & ~projected.astype(bool)))
            projected = solid.astype(np.uint8)
        obstacles |= projected
        used += int(np.count_nonzero(overlap))
    cells = np.full(support.shape, -1, dtype=np.int8)
    cells[(support != 0) & (room_mask != 0)] = 0
    cells[obstacles != 0] = 100
    return OccupancyMap(
        cells,
        resolution,
        (float(origin[0]), float(origin[1])),
        floor_z,
        {
            "method": "authored_geometry_height_clipped_triangle_projection",
            "frame": "source_world_metres_z_up",
            "array_row_order": "increasing_world_y",
            "image_row_order": "decreasing_world_y",
            "height_above_floor_m": [min_height, max_height],
            "floor_tolerance_m": floor_tolerance,
            "floor_level_selection": "dominant_topmost_floor_area_inside_room_polygons",
            "floor_level_area_m2": dict(zip(map(str, bins), weights.tolist(), strict=True)),
            "mesh_count": len(meshes),
            "triangles_in_height_band": used,
            "height_clipped_triangles": clipped,
            "watertight_silhouette_filled_cells": filled,
            "room_count": len(rooms),
            "support_area_m2": float(np.count_nonzero(support) * resolution**2),
            "room_area_raster_m2": float(np.count_nonzero(room_mask) * resolution**2),
            "robot_radius_m": 0,
            "sensor_derived": False,
            "physics_collision_equivalence_verified": False,
            "navigation_map_relationship": (
                "Evaluator export; navigation.NavigationMap is authoritative for runtime planning."
            ),
            "limitations": [
                "Static source geometry, not SLAM or measured robot traversability.",
                "One dominant floor level; stairs and other levels are not represented as traversable.",
                "Rasterization and watertight silhouette filling are conservative approximations.",
                "No robot footprint inflation; use a separate footprint/costmap before planning.",
                "Doors and movable objects retain their authored pose; motion can invalidate this map.",
            ],
        },
    )
