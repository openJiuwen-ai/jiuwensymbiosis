"""Sensor-only sparse occupancy and online observation candidates.

Candidates use a camera-plane proxy. The execution port MUST independently admit
the complete robot envelope; these candidates never authorize motion.
No scene loader, known map, room polygon or true pose is accepted here.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np

RES = 0.10
NEIGHBORS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_from_pose(pose: np.ndarray) -> float:
    return math.atan2(pose[1, 2], pose[0, 2])


class OnlineMap:
    def __init__(self) -> None:
        self.free: set[tuple[int, ...]] = set()
        self.occupied: set[tuple[int, ...]] = set()
        self.surface: dict[tuple[int, ...], list[float]] = {}  # 5cm voxel -> measured xyzrgb and view count
        self.visits: dict[tuple[int, ...], int] = {}
        self.observed_views: set[tuple[int, ...]] = set()
        self.version = 0
        self.last_new = 0
        self.ground_level: float | None = None

    def integrate(self, rgb: np.ndarray, depth: np.ndarray, k: np.ndarray, pose: np.ndarray) -> None:
        v, u = np.mgrid[0 : depth.shape[0] : 4, 0 : depth.shape[1] : 4]
        z = depth[v, u]
        ok = np.isfinite(z) & (z > 0.08) & (z < 8)
        u, v, z = u[ok], v[ok], z[ok]
        camera = np.column_stack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z])
        world = camera @ pose[:3, :3].T + pose[:3, 3]
        # A low, broad horizontal return is a projection reference, not proof
        # of traversable support. The execution port still checks the whole body.
        if self.ground_level is None and len(world) >= 50:
            level = float(np.quantile(world[:, 2], 0.05))
            plane = world[np.abs(world[:, 2] - level) < 0.03]
            if len(plane) >= 50 and np.all(np.ptp(plane[:, :2], axis=0) > 0.5) and level < pose[2, 3] - 0.7:
                self.ground_level = float(np.median(plane[:, 2]))
        colors = rgb[v, u]
        origin = pose[:3, 3]
        hit = np.floor(world / RES).astype(np.int32)
        self.occupied.update(map(tuple, hit.tolist()))
        vectors = world - origin
        lengths = np.linalg.norm(vectors, axis=1)
        # Half-voxel ray spacing, stop before each observed surface. Invalid
        # returns never clear space; unobserved cells remain absent/unknown.
        distances = np.arange(0, min(8, lengths.max()), RES / 2) if len(lengths) else np.array([])
        for start in range(0, len(vectors), 512):
            vec, lens = vectors[start : start + 512], lengths[start : start + 512]
            valid = distances[None, :] < lens[:, None] - 0.075
            ray = origin + vec[:, None, :] / lens[:, None, None] * distances[None, :, None]
            cells = np.unique(np.floor(ray[valid] / RES).astype(np.int32), axis=0)
            self.free.update(map(tuple, cells.tolist()))
        # A camera was physically placed here at reset. Only its initial
        # occupied envelope (not surrounding room geometry) is given clear.
        if self.version == 0:
            center = np.floor(origin / RES).astype(int)
            for x in range(-2, 3):
                for y in range(-2, 3):
                    for zz in range(-2, 3):
                        if x * x + y * y + zz * zz <= 6:
                            self.free.add(tuple(center + [x, y, zz]))
        old = len(self.surface)
        keys = np.floor(world / 0.05).astype(np.int32)
        for key, point, color in zip(keys.tolist(), world, colors, strict=True):
            key = tuple(key)
            if key in self.surface:
                self.surface[key][6] += 1
            else:
                self.surface[key] = [*point.tolist(), *color.tolist(), 1]
        self.last_new = len(self.surface) - old
        self.version += 1
        yaw = yaw_from_pose(pose)
        self.observed_views.add(
            (round(float(pose[0, 3]) / 0.3), round(float(pose[1, 3]) / 0.3), round(yaw / math.radians(30)))
        )

    def navigation(self, pose: np.ndarray, *, observed_projection: bool = True) -> tuple:
        center = tuple(np.floor(pose[:2, 3] / RES).astype(int))
        height = int(math.floor(pose[2, 3] / RES))
        levels = (height - 1, height, height + 1)
        layers = [{(x, y) for x, y, z in self.free if z == level} for level in levels]
        free = set.intersection(*layers)
        occupied = {(x, y) for x, y, z in self.occupied if z in levels}
        if observed_projection and self.ground_level is not None:
            # Downward-looking cameras see the floor ahead, not a horizontal
            # corridor at eye height. Project observed ray columns for candidate
            # generation only; unknown support/overhangs require independent admission.
            low = self.ground_level + 0.08
            high = pose[2, 3] + 0.2
            free = {(x, y) for x, y, z in self.free if low < (z + 0.5) * RES < high}
            occupied = {(x, y) for x, y, z in self.occupied if low < (z + 0.5) * RES < high}
            # Current physical camera/base proxy, including self returns below
            # the camera. This is not a clearance envelope for the full robot.
            current = {(center[0] + x, center[1] + y) for x in range(-2, 3) for y in range(-2, 3) if x * x + y * y <= 4}
            free.update(current)
            occupied -= current
        free -= occupied
        # Nine-cell square conservatively covers the horizontal camera envelope.
        # conservatism. All required cells must have observed free evidence.
        offsets = [(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1)]
        safe = {p for p in free if all((p[0] + dx, p[1] + dy) in free for dx, dy in offsets)}
        if center not in safe:
            # Do not manufacture free cells to rescue an uncertain map.
            return center, {}, free, occupied
        previous: dict[tuple, tuple | None] = {center: None}
        queue = deque([center])
        while queue:
            p = queue.popleft()
            for dx, dy in NEIGHBORS:
                q = (p[0] + dx, p[1] + dy)
                if q in safe and q not in previous:
                    previous[q] = p
                    queue.append(q)
        return center, previous, free, occupied

    @staticmethod
    def expected_gain(xy: np.ndarray, yaw: float, known_free: set, occupied: set) -> float:
        unknown = set()
        for a in np.linspace(yaw - 0.57, yaw + 0.57, 17):
            for d in np.arange(0.2, 3.1, RES):
                cell = tuple(np.floor((xy + d * np.array([math.cos(a), math.sin(a)])) / RES).astype(int))
                if cell in occupied:
                    break
                if cell not in known_free:
                    unknown.add(cell)
        return len(unknown) * RES**2

    def candidates(self, pose: np.ndarray, tasks=()) -> tuple[list[dict], dict]:
        center, paths, free, occupied = self.navigation(pose)
        yaw = yaw_from_pose(pose)
        position = pose[:2, 3]
        choices = []

        def add(kind, target, heading, path, purpose, semantic=0.0):
            signature = (
                round(float(target[0]) / 0.3),
                round(float(target[1]) / 0.3),
                round(heading / math.radians(30)),
            )
            seen = signature in self.observed_views
            repeats = self.visits.get(signature, 0)
            gain = self.expected_gain(target, heading, free, occupied)
            distance = sum(np.linalg.norm(np.array(b) - a) for a, b in zip(path, path[1:], strict=False))
            # Kinematic camera can translate horizontally without changing yaw.
            turning = abs(wrap(heading - yaw))
            cost = float(distance / 0.15 + turning / math.radians(15) + 1)
            choices.append(
                {
                    "kind": kind,
                    "target_xy_m": np.round(target, 3).tolist(),
                    "heading_rad": round(heading, 5),
                    "path_xy_m": [np.round(x, 3).tolist() for x in path],
                    "purpose": purpose,
                    "estimated_unknown_view_m2": round(gain, 3),
                    "semantic_need": semantic,
                    "path_length_m": round(float(distance), 3),
                    "turn_deg": round(math.degrees(turning), 1),
                    "cost_units": round(cost, 2),
                    "prior_visits": repeats,
                    "view_already_observed": seen,
                    "map_version": self.version,
                    "signature": list(signature),
                    "utility": round((gain + semantic * 0.5) / (1 + cost * 0.10) / (1 + repeats * 2), 5),
                }
            )

        for angle in (-60, -30, 30, 60):
            heading = wrap(yaw + math.radians(angle))
            add("scan", position, heading, [position], "observe a different direction from current position")
        # Frontier stands inside observed free space, facing nearby unknown.
        frontier = []
        for p in sorted(paths):
            if np.linalg.norm(np.array(p) - center) < 3:
                continue
            unseen = [
                (p[0] + dx, p[1] + dy)
                for dx in range(-3, 4)
                for dy in range(-3, 4)
                if (p[0] + dx, p[1] + dy) not in free and (p[0] + dx, p[1] + dy) not in occupied
            ]
            if unseen:
                frontier.append((p, np.mean(unseen, axis=0)))
        groups: dict[tuple, tuple] = {}
        for p, unknown in frontier:
            key = (p[0] // 5, p[1] // 5)
            if key not in groups or np.linalg.norm(np.array(p) - center) < np.linalg.norm(
                np.array(groups[key][0]) - center
            ):
                groups[key] = (p, unknown)
        for goal, unknown in groups.values():
            route = [goal]
            while paths[route[-1]] is not None:
                route.append(paths[route[-1]])
            route.reverse()
            full_length = (len(route) - 1) * RES
            # Execute only first <=0.6m, revisit full frontier on next decision.
            route = route[:7]
            target = (np.array(route[-1]) + 0.5) * RES
            route_xy = [position, *[(np.array(p) + 0.5) * RES for p in route[1:]]]
            # Simplify stair-step paths only when every sampled cell is in
            # the already validated connected component.
            smooth = [route_xy[0]]
            index = 0
            while index < len(route_xy) - 1:
                farthest = index + 1
                for j in range(index + 1, len(route_xy)):
                    line = np.linspace(
                        route_xy[index],
                        route_xy[j],
                        max(2, int(np.linalg.norm(route_xy[j] - route_xy[index]) / 0.02) + 1),
                    )
                    if all(tuple(np.floor(p / RES).astype(int)) in paths for p in line):
                        farthest = j
                smooth.append(route_xy[farthest])
                index = farthest
            route_xy = smooth
            heading = math.atan2(unknown[1] - route[-1][1], unknown[0] - route[-1][0])
            add("advance", target, heading, route_xy, "approach an observed-ray frontier; full-body admission required")
            full_target = (np.array(goal) + 0.5) * RES
            downstream = self.expected_gain(full_target, heading, free, occupied)
            choices[-1].update(
                frontier_distance_before_m=round(full_length, 3),
                frontier_distance_after_m=round(max(0, full_length - choices[-1]["path_length_m"]), 3),
                downstream_unknown_view_m2=round(downstream, 3),
            )
            benefit = choices[-1]["estimated_unknown_view_m2"] + 0.4 * downstream * min(
                1, choices[-1]["path_length_m"] / max(0.1, full_length)
            )
            choices[-1]["utility"] = round(
                benefit / (1 + choices[-1]["cost_units"] * 0.10) / (1 + choices[-1]["prior_visits"] * 2), 5
            )
        # Preserve alternatives with diverse goals before final bounded list.
        moving = sorted([c for c in choices if c["kind"] == "advance"], key=lambda c: -c["utility"])
        selected: list[dict] = []
        for c in moving:
            if all(np.linalg.norm(np.array(c["target_xy_m"]) - s["target_xy_m"]) > 0.25 for s in selected):
                selected.append(c)
            if len(selected) >= 6:
                break
        choices = choices[:4] + selected
        # Semantic tasks come exclusively from observed masks / memory evidence.
        for task in tasks:
            if task.get("state") != "open":
                continue
            target = task.get("centroid_m") or task.get("center_m")
            if target is not None:
                target = np.array(target[:2])
                old_direction = position - target
                options = []
                for cell in sorted(paths):
                    xy = (np.array(cell) + 0.5) * RES
                    travel = np.linalg.norm(xy - position)
                    direction = xy - target
                    if 0.3 < travel < 0.9 and 0.4 < np.linalg.norm(direction) < 3:
                        change = abs(
                            wrap(
                                math.atan2(direction[1], direction[0]) - math.atan2(old_direction[1], old_direction[0])
                            )
                        )
                        if change > math.radians(15):
                            options.append((change / (1 + travel), cell))
                if options:
                    goal = max(options)[1]
                    route = [goal]
                    while paths[route[-1]] is not None:
                        route.append(paths[route[-1]])
                    route.reverse()
                    route = route[:7]
                    route_xy = [position, *[(np.array(p) + 0.5) * RES for p in route[1:]]]
                    viewpoint = route_xy[-1]
                    direction = target - viewpoint
                    actual_change = abs(
                        wrap(math.atan2(-direction[1], -direction[0]) - math.atan2(old_direction[1], old_direction[0]))
                    )
                    if actual_change <= math.radians(15):
                        continue
                    heading = math.atan2(direction[1], direction[0])
                    add(
                        "reobserve",
                        viewpoint,
                        heading,
                        route_xy,
                        "change viewpoint to resolve an observed memory ambiguity",
                        1.0,
                    )
                    choices[-1]["memory_task_id"] = task["task_id"]
                    break
        # Already observed scans with zero estimated new volume and no semantic
        # target are redundant. Apply the same filter to every policy.
        choices = [
            c
            for c in choices
            if not (
                c["kind"] == "scan"
                and c["view_already_observed"]
                and c["estimated_unknown_view_m2"] < 0.01
                and c["semantic_need"] == 0
            )
        ]
        for i, c in enumerate(choices):
            c["id"] = f"a{i}"
        return choices, {
            "reachable_cells": len(paths),
            "free_cells": len(free),
            "occupied_cells": len(occupied),
            "frontier_cells": len(frontier),
            "surface_voxels": len(self.surface),
            "new_surface_voxels": self.last_new,
            "map_version": self.version,
            "candidate_geometry": "observed_ray_projection" if self.ground_level is not None else "camera_height_slice",
            "projection_reference_z_m": self.ground_level,
        }

    def record_choice(self, choice: dict) -> None:
        key = tuple(choice["signature"])
        self.visits[key] = self.visits.get(key, 0) + 1

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            free=np.array(sorted(self.free), dtype=np.int32),
            occupied=np.array(sorted(self.occupied), dtype=np.int32),
            surface=np.array(list(self.surface.values())),
            resolution_m=RES,
        )
