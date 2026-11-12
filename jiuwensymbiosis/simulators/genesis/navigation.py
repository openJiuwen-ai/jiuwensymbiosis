"""Static, supported-floor navigation, independent of the renderer and Agent.

Reference: RoboSim's native_grid_planner (inflated footprint, A*, continuous
segment checks and an exact final heading). This implementation uses Genesis's
world-space collision-source triangles, not Isaac APIs or a teleport operation.
Unknown/unsupported cells are blocked. A circular whole-body envelope allows
holonomic translation and rotation without a separate orientation search.
"""

from __future__ import annotations

import hashlib
import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def finite_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _paint(mask, origin, resolution, triangle, padding):
    """Conservative XY projection, including vertical/degenerate triangles."""
    low = np.maximum(0, np.floor((triangle[:, :2].min(axis=0) - padding - origin) / resolution).astype(int))
    high = np.minimum(
        mask.shape[::-1], np.ceil((triangle[:, :2].max(axis=0) + padding - origin) / resolution).astype(int) + 1
    )
    if np.any(high <= low):
        return
    xx, yy = np.meshgrid(np.arange(low[0], high[0]), np.arange(low[1], high[1]))
    points = np.stack((xx, yy), axis=-1) * resolution + origin + resolution / 2
    signs = []
    edge_hit = np.zeros(xx.shape, dtype=bool)
    for index in range(3):
        a, b = triangle[index, :2], triangle[(index + 1) % 3, :2]
        edge = b - a
        signs.append(edge[0] * (points[..., 1] - a[1]) - edge[1] * (points[..., 0] - a[0]))
        t = np.clip(np.sum((points - a) * edge, axis=-1) / max(float(edge @ edge), 1e-20), 0, 1)
        edge_hit |= np.sum((points - a - t[..., None] * edge) ** 2, axis=-1) <= padding**2
    ab, ac = triangle[1, :2] - triangle[0, :2], triangle[2, :2] - triangle[0, :2]
    area = abs(float(ab[0] * ac[1] - ab[1] * ac[0]))
    inside = (np.minimum.reduce(signs) >= -1e-10) | (np.maximum.reduce(signs) <= 1e-10)
    if area < 1e-12:
        inside[:] = False
    rows, columns = slice(low[1], high[1]), slice(low[0], high[0])
    mask[rows, columns] |= inside | edge_hit


def _paint_obstacles(mask, origin, resolution, triangles):
    """Batch tiny triangles as conservative cell rectangles; retain large-face geometry.

    Indoor source meshes can contain millions of sub-cell triangles. A difference
    array unions their rectangles in O(triangles + cells), without a Python loop
    per face. This can only add blocked cells, never remove an obstacle.
    """
    if triangles.size == 0:
        return
    padding = resolution / math.sqrt(2)
    low = np.maximum(0, np.floor((triangles[:, :, :2].min(axis=1) - padding - origin) / resolution).astype(int))
    high = np.minimum(
        mask.shape[::-1], np.ceil((triangles[:, :, :2].max(axis=1) + padding - origin) / resolution).astype(int) + 1
    )
    small = np.all(high - low <= 4, axis=1)
    valid = small & np.all(high > low, axis=1)
    a, b = low[valid], high[valid]
    counts = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    np.add.at(counts, (a[:, 1], a[:, 0]), 1)
    np.add.at(counts, (a[:, 1], b[:, 0]), -1)
    np.add.at(counts, (b[:, 1], a[:, 0]), -1)
    np.add.at(counts, (b[:, 1], b[:, 0]), 1)
    mask |= counts.cumsum(axis=0).cumsum(axis=1)[:-1, :-1] > 0
    for triangle in triangles[~small]:
        _paint(mask, origin, resolution, triangle, padding)


@dataclass
class NavigationMap:
    origin: np.ndarray
    free: np.ndarray
    resolution: float
    radius: float

    @classmethod
    def from_meshes(
        cls,
        meshes: Sequence[Any],
        *,
        floor_z: float,
        radius: float = 0.47,
        height: float = 1.95,
        resolution: float = 0.025,
    ) -> NavigationMap:
        """Rasterize a single floor and erode support by the full robot envelope."""
        low = np.min([item.mesh.bounds[0, :2] for item in meshes], axis=0) - resolution
        high = np.max([item.mesh.bounds[1, :2] for item in meshes], axis=0) + resolution
        size = np.ceil((high - low) / resolution).astype(int)
        if np.prod(size) > 250_000:
            raise ValueError("navigation_map_too_large")
        support = np.zeros(tuple(size[::-1]), dtype=bool)
        obstacles = np.zeros_like(support)
        for item in meshes:
            triangles = item.mesh.triangles
            floors = triangles[np.all(np.abs(triangles[:, :, 2] - floor_z) <= 0.015, axis=1)]
            for triangle in floors:
                _paint(support, low, resolution, triangle, 0)
            blocked = triangles[
                (triangles[:, :, 2].max(axis=1) > floor_z + 0.06) & (triangles[:, :, 2].min(axis=1) < floor_z + height)
            ]
            _paint_obstacles(obstacles, low, resolution, blocked)
        # Include half a cell diagonal so continuous positions remain conservative.
        cells = math.ceil((radius + resolution / math.sqrt(2)) / resolution)
        raw = support & ~obstacles
        padded = np.pad(raw, cells, constant_values=False)
        free = raw.copy()
        for dy in range(-cells, cells + 1):
            for dx in range(-cells, cells + 1):
                if math.hypot(dx, dy) * resolution <= radius + resolution / math.sqrt(2):
                    rows = slice(cells + dy, cells + dy + size[1])
                    columns = slice(cells + dx, cells + dx + size[0])
                    free &= padded[rows, columns]
        return cls(low, free, resolution, radius)

    @classmethod
    def plane(
        cls, center: ArrayLike, *, radius: float = 0.47, extent: float = 10.0, resolution: float = 0.1
    ) -> NavigationMap:
        """Bounded workspace on the explicit infinite plane used by robot-only scenes."""
        size = round(2 * extent / resolution)
        free = np.ones((size, size), dtype=bool)
        border = math.ceil(radius / resolution) + 1
        free[:border] = free[-border:] = False
        free[:, :border] = free[:, -border:] = False
        return cls(np.asarray(center, dtype=float) - extent, free, resolution, radius)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.free.tobytes() + self.origin.tobytes()).hexdigest()[:16]

    def cell(self, point: ArrayLike) -> tuple[int, int] | None:
        xy = np.floor((np.asarray(point)[:2] - self.origin) / self.resolution).astype(int)
        x, y = int(xy[0]), int(xy[1])
        return (x, y) if 0 <= y < self.free.shape[0] and 0 <= x < self.free.shape[1] else None

    def clear(self, point: ArrayLike) -> bool:
        cell = self.cell(point)
        return cell is not None and bool(self.free[cell[1], cell[0]])

    def segment_clear(self, start: ArrayLike, end: ArrayLike) -> bool:
        start, end = np.asarray(start)[:2], np.asarray(end)[:2]
        count = max(1, math.ceil(float(np.linalg.norm(end - start)) / (self.resolution / 4)))
        return all(self.clear(start + (end - start) * i / count) for i in range(count + 1))

    def path(self, start: Sequence[float], goal: Sequence[float]) -> list[list[float]]:
        start, goal = list(start[:2]), list(goal[:2])
        if not self.clear(start):
            raise ValueError("navigation_start_blocked_or_unsupported")
        if not self.clear(goal):
            raise ValueError("navigation_goal_blocked_or_unsupported")
        if self.segment_clear(start, goal):
            return [start, goal]
        first, last = self.cell(start), self.cell(goal)
        if first is None or last is None:
            raise ValueError("navigation_goal_outside_map")
        queue = [(0.0, first)]
        costs, parents = {first: 0.0}, {}
        visited = set()
        while queue:
            _, current = heapq.heappop(queue)
            if current in visited:
                continue
            visited.add(current)
            if current == last:
                break
            x, y = current
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nx, ny = x + dx, y + dy
                if not (0 <= ny < self.free.shape[0] and 0 <= nx < self.free.shape[1] and self.free[ny, nx]):
                    continue
                if dx and dy and not (self.free[y, nx] and self.free[ny, x]):
                    continue
                cost = costs[current] + math.hypot(dx, dy)
                node = (nx, ny)
                if cost < costs.get(node, math.inf):
                    costs[node], parents[node] = cost, current
                    heapq.heappush(queue, (cost + math.hypot(nx - last[0], ny - last[1]), node))
        if last not in visited:
            raise ValueError("navigation_no_path")
        cells, current = [last], last
        while current != first:
            current = parents[current]
            cells.append(current)
        points = [
            start,
            *[(self.origin + (np.array(c) + 0.5) * self.resolution).tolist() for c in reversed(cells)],
            goal,
        ]
        pruned = [points[0]]
        index = 0
        while index < len(points) - 1:
            following = index + 1
            if not self.segment_clear(points[index], points[following]):
                raise ValueError("navigation_continuous_path_blocked")
            while following + 1 < len(points) and self.segment_clear(points[index], points[following + 1]):
                following += 1
            pruned.append(points[following])
            index = following
        return pruned

    def plan(
        self,
        start: Sequence[float],
        goal: Sequence[float] | None = None,
        *,
        bounds: ArrayLike | None = None,
        stand_off_m: float = 0.4,
    ) -> dict:
        """Choose a reachable object perimeter point, or preserve an exact pose goal."""
        if not self.clear(start):
            raise ValueError("navigation_start_blocked_or_unsupported")
        if bounds is not None:
            gap = finite_number(stand_off_m, "stand_off_m")
            if not 0.1 <= gap <= 2:
                raise ValueError("stand_off_m must be between 0.1 and 2 metres")
            low, high = np.asarray(bounds, dtype=float)[:, :2]
            center = (low + high) / 2
            low, high = low - self.radius - gap, high + self.radius + gap
            candidates: list[list[float]] = []
            for x in np.linspace(low[0], high[0], max(3, math.ceil((high[0] - low[0]) / 0.25) + 1)):
                candidates.extend(([x, low[1]], [x, high[1]]))
            for y in np.linspace(low[1], high[1], max(3, math.ceil((high[1] - low[1]) / 0.25) + 1)):
                candidates.extend(([low[0], y], [high[0], y]))
            candidates.sort(key=lambda point: math.dist(start[:2], point))
            best = None
            for point in candidates:
                if best is not None and math.dist(start[:2], point) >= best[0]:
                    break
                if not self.clear(point):
                    continue
                try:
                    path = self.path(start, point)
                except ValueError:
                    continue
                length = sum(math.dist(a, b) for a, b in zip(path, path[1:], strict=False))
                if best is None or length < best[0]:
                    best = length, path, [*map(float, point), math.atan2(center[1] - point[1], center[0] - point[0])]
            if best is None:
                raise ValueError("navigation_no_reachable_object_standoff")
            length, path, goal = best
        else:
            if goal is None or len(goal) != 3:
                raise ValueError("navigation_goal_requires_x_y_yaw")
            goal = [finite_number(value, "goal") for value in goal]
            path = self.path(start, goal)
            length = sum(math.dist(a, b) for a, b in zip(path, path[1:], strict=False))
        if length > 20:
            raise ValueError("navigation_path_exceeds_20_metres")
        # RelativeMotion remains the sole wheel controller; each subgoal respects its 1 m cap.
        waypoints = []
        for a, b in zip(path, path[1:], strict=False):
            count = max(1, math.ceil(math.dist(a, b) / 0.6))
            waypoints.extend([[a[i] + (b[i] - a[i]) * j / count for i in (0, 1)] for j in range(1, count + 1)])
        if len(waypoints) > 64:
            raise ValueError("navigation_path_too_complex")
        return {
            "planner": "genesis_supported_grid_v1",
            "map_fingerprint": self.fingerprint,
            "frame": "world",
            "goal": goal,
            "waypoints": waypoints,
            "path_length_m": length,
            "footprint_radius_m": self.radius,
            "position_tolerance_m": 0.04,
            "yaw_tolerance_rad": 0.04,
        }
