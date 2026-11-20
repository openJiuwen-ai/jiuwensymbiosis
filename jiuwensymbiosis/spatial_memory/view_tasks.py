"""Evidence-backed revisit regions; task identity is not physical object identity."""

from __future__ import annotations

import uuid

import numpy as np
from scipy.spatial import cKDTree

from jiuwensymbiosis.spatial_memory.geometry import MemoryPolicy, different_view, surface_in_view
from jiuwensymbiosis.spatial_memory.models import Observation, canonical


def overlap(first: np.ndarray, second: np.ndarray, distance: float) -> float:
    if not len(first) or not len(second):
        return 0.0
    if np.linalg.norm(np.maximum(0, np.maximum(first.min(0) - second.max(0), second.min(0) - first.max(0)))) > distance:
        return 0.0
    a = float(np.mean(cKDTree(first).query(second)[0] < distance))
    b = float(np.mean(cKDTree(second).query(first)[0] < distance))
    return 2 * a * b / (a + b) if a + b else 0.0


def update_tasks(
    tasks: list[dict], observation: Observation, assignments: list[dict], projected: list, policy: MemoryPolicy
) -> list[dict]:
    """Resolve only from an unambiguous later matched surface; exhaustion stays a gap.

    Three distinct frustum opportunities without resolution defer a region. This
    is a bounded revisit heuristic, not proof of visibility through occlusion.
    Missing-depth tasks cannot invent a 3-D target and are deferred immediately.
    """
    pose = observation.map_from_camera.tolist()
    for task in tasks:
        if task["state"] != "open":
            continue
        points = np.asarray(task["surface_m"])
        visible = surface_in_view(points, observation, occlusion_margin_m=policy.surface_distance_m)
        if len(visible) < policy.min_points or not different_view(
            task["last_attempt_pose"], observation.map_from_camera, policy
        ):
            continue
        task["attempts"] += 1
        task["last_attempt_pose"] = pose
        task["last_attempt_observation"] = observation.observation_id
        candidates = [
            item
            for item, (surface, _, _) in zip(assignments, projected, strict=True)
            if item["status"] == "matched" and overlap(visible, surface, policy.surface_distance_m) >= 0.65
        ]
        if len(candidates) == 1:
            task.update(
                state="resolved",
                resolution="later_unique_surface_match",
                resolved_by={
                    "observation_id": observation.observation_id,
                    "local_id": candidates[0]["local_id"],
                    "entity_id": candidates[0]["entity_id"],
                },
            )
        elif task["attempts"] >= 3:
            task.update(state="deferred", resolution="view_attempt_budget_exhausted")
    for item, (points, _, _) in zip(assignments, projected, strict=True):
        if item["status"] != "needs_view":
            continue
        reference = {"observation_id": observation.observation_id, "local_id": item["local_id"]}
        prior = next(
            (
                task
                for task in tasks
                if task["state"] in {"open", "deferred"}
                and task["reason"] == item["reason"]
                and overlap(points, np.asarray(task["surface_m"]), policy.surface_distance_m) >= 0.65
            ),
            None,
        )
        if prior is not None:
            prior["evidence_count"] += 1
            prior["last_evidence"] = reference
            continue
        localizable = len(points) >= policy.min_points
        sampled = points[np.linspace(0, len(points) - 1, min(512, len(points)), dtype=int)] if len(points) else points
        tasks.append(
            {
                "task_id": str(uuid.uuid5(uuid.NAMESPACE_URL, canonical([observation.scene_epoch, reference]))),
                "state": "open" if localizable else "deferred",
                "reason": item["reason"],
                "resolution": None if localizable else "no_supported_3d_target",
                "first_evidence": reference,
                "last_evidence": reference,
                "evidence_count": 1,
                "surface_m": sampled.tolist(),
                "center_m": np.median(points, axis=0).tolist() if localizable else None,
                "last_attempt_pose": pose,
                "attempts": 0,
                "semantics": "revisit_region_not_object_identity",
            }
        )
    return tasks
