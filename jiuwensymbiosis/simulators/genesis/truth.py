"""Worker-owned logical entities and bounded, immutable physical snapshots.

No model/runtime imports. Queries are executed between physics steps; a reused
snapshot never mixes robot and object coordinates from different steps.
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
import uuid
from collections import OrderedDict
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .navigation import NavigationMap


def _quat_multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return [
        w * v - x * i - y * j - z * k,
        w * i + x * v + y * k - z * j,
        w * j - x * k + y * v + z * i,
        w * k + x * j - y * i + z * v,
    ]


def in_base(entity: dict, robot: dict) -> dict:
    """Transform both the origin pose and all eight bounds corners consistently."""
    result = copy.deepcopy(entity)
    yaw = robot["body_yaw_rad"]
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    anchor = np.asarray(robot["position_m"])
    result["position_m"] = (rotation @ (np.asarray(result["position_m"]) - anchor)).tolist()
    result["quaternion_wxyz"] = _quat_multiply([math.cos(yaw / 2), 0, 0, -math.sin(yaw / 2)], result["quaternion_wxyz"])
    if result["bounds_m"] is not None:
        low, high = result["bounds_m"]
        corners = np.array(list(product(*zip(low, high, strict=True))))
        corners = (corners - anchor) @ rotation.T
        result["bounds_m"] = [corners.min(axis=0).tolist(), corners.max(axis=0).tolist()]
        result["center_m"] = corners.mean(axis=0).tolist()
        result["size_m"] = np.ptp(corners, axis=0).tolist()
    result["frame"] = "base"
    return result


class EntityRegistry:
    """Hide USD/render/collision identities behind stable logical IDs."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.entities: OrderedDict[str, dict] = OrderedDict()
        self.bodies: dict[str, Any] = {}
        self.snapshots: OrderedDict[str, dict] = OrderedDict()
        self.navigation_map: NavigationMap | None = None
        self._build()

    def _add(
        self,
        key,
        name,
        *,
        kind="object",
        position=(0, 0, 0),
        quaternion=(1, 0, 0, 0),
        bounds=None,
        body=None,
        collision=False,
    ):
        entity_id = hashlib.sha256(f"{self.runtime.settings.scene_id}:{key}".encode()).hexdigest()[:20]
        self.entities[entity_id] = {
            "entity_id": entity_id,
            "name": str(name),
            "kind": kind,
            "category": None,
            "semantic_source": "authored_name",
            "dynamic": body is not None and (kind != "robot" or self.runtime.motion is not None),
            "articulated": kind == "robot",
            "collision": bool(collision),
            "position_m": list(position),
            "quaternion_wxyz": list(quaternion),
            "position_reference": "authored_origin",
            "bounds_m": bounds,
            "frame": "world",
        }
        if body is not None:
            self.bodies[entity_id] = body

    def _build(self):
        runtime = self.runtime
        if runtime.source is not None:
            if __package__:
                from .selection import UsdPickIndex
            else:
                from selection import UsdPickIndex
            from pxr import Gf

            index = UsdPickIndex(runtime.source)
            groups = {}
            for prim, bounds in index.meshes:
                model, ancestor = prim, prim.GetParent()
                while ancestor and not ancestor.IsPseudoRoot():
                    if (
                        ancestor.GetMetadata("kind") in {"component", "subcomponent", "assembly"}
                        or ancestor.HasAuthoredReferences()
                    ):
                        model = ancestor
                        break
                    ancestor = ancestor.GetParent()
                key = str(model.GetPath())
                groups.setdefault(key, (model, []))[1].append(bounds)
            for key, (model, boxes) in sorted(groups.items()):
                matrix = index.transforms.GetLocalToWorldTransform(model)
                pos = np.asarray(matrix)[3, :3] * index.scale
                q = Gf.Transform(matrix).GetRotation().GetQuat()
                quat = [float(q.GetReal()), *map(float, q.GetImaginary())]
                if index.y_up:
                    pos = pos[[0, 2, 1]] * [1, -1, 1]
                    quat = _quat_multiply([math.sqrt(0.5), math.sqrt(0.5), 0, 0], quat)
                boxes = np.asarray(boxes)
                bounds = [boxes[:, 0].min(axis=0).tolist(), boxes[:, 1].max(axis=0).tolist()]
                name = model.GetName()
                if name in {"entity", "visual", "body"}:
                    name = model.GetParent().GetName()
                self._add(key, name, position=pos, quaternion=quat, bounds=bounds, collision=runtime.indoor is not None)
        if runtime.robot is not None:
            self._add(
                "robot",
                runtime.robot.name,
                kind="robot",
                body=runtime.robot.entity,
                collision=runtime.motion is not None,
            )
        if runtime.source is None:
            self._add("ground", "地面", kind="ground", collision=True)
        if runtime.bodies:
            self._add(
                "platform", "平台", position=(0, 0, 0.06), bounds=[[-0.8, -0.55, 0], [0.8, 0.55, 0.12]], collision=True
            )
            for key, name, body in zip(
                ("gold_sphere", "blue_box", "copper_sphere"),
                ("金色球", "蓝色方块", "铜色球"),
                runtime.bodies,
                strict=True,
            ):
                self._add(key, name, body=body, collision=True)

    def snapshot(self, snapshot_id: str | None = None) -> dict:
        runtime, now = self.runtime, time.monotonic()
        for key, value in list(self.snapshots.items()):
            if now - value["captured_at"] > 60 or value["reset_generation"] != runtime.cycle:
                del self.snapshots[key]
        if snapshot_id:
            if snapshot_id not in self.snapshots:
                raise ValueError("snapshot_expired_or_scene_reset")
            return self.snapshots[snapshot_id]
        entities = copy.deepcopy(self.entities)
        for key, body in self.bodies.items():
            entity = entities[key]
            entity.update(
                position_m=body.get_pos()[0].cpu().tolist(),
                quaternion_wxyz=body.get_quat()[0].cpu().tolist(),
                bounds_m=body.get_AABB()[0].cpu().tolist(),
            )
        for entity in entities.values():
            if entity["bounds_m"] is not None:
                low, high = np.asarray(entity["bounds_m"])
                entity.update(center_m=((low + high) / 2).tolist(), size_m=(high - low).tolist())
        robot = None
        if runtime.motion is not None:
            if __package__:
                from ..common.base_motion import measure

                robot = measure(runtime.robot)
            else:
                from base_motion import measure as worker_measure

                robot = worker_measure(runtime.robot)
            robot["position_reference"] = "authored_root_body_heading"
        result = {
            "source": "simulation_ground_truth",
            "scene_instance_id": runtime.info["instance_id"],
            "reset_generation": runtime.cycle,
            "snapshot_id": uuid.uuid4().hex,
            "step": runtime.steps,
            "sim_time": runtime.steps * runtime.settings.dt,
            "captured_at": now,
            "frame": "world",
            "units": {"length": "m", "angle": "rad", "time": "s"},
            "robot": robot,
            "entities": entities,
        }
        self.snapshots[result["snapshot_id"]] = result
        while len(self.snapshots) > 16:
            self.snapshots.popitem(last=False)
        return result

    def query(self, operation: str, params: dict) -> dict:
        snapshot = self.snapshot(params.get("snapshot_id"))
        metadata = {key: value for key, value in snapshot.items() if key not in {"robot", "entities"}}
        if operation == "list_entities":
            limit, offset = params.get("limit", 20), params.get("offset", 0)
            if type(limit) is not int or not 1 <= limit <= 50:
                raise ValueError("invalid_pagination")
            if type(offset) is not int or offset < 0:
                raise ValueError("invalid_pagination")
            query, kind = str(params.get("query", "")).casefold(), params.get("kind", "")
            entities = [
                e
                for e in snapshot["entities"].values()
                if query in e["name"].casefold() and (not kind or kind == e["kind"])
            ]
            page = slice(offset, offset + limit)
            return {
                **metadata,
                "entities": [
                    {k: e[k] for k in ("entity_id", "name", "kind", "category", "dynamic", "collision")}
                    for e in entities[page]
                ],
                "total": len(entities),
                "next_offset": offset + limit if offset + limit < len(entities) else None,
            }
        if operation == "robot_state":
            if snapshot["robot"] is None:
                raise ValueError("mobile_robot_state_unavailable")
            return {**metadata, "robot": snapshot["robot"]}
        if operation == "entity_state":
            entity = snapshot["entities"].get(params.get("entity_id"))
            if entity is None:
                raise ValueError("unknown_entity_id")
            frame = params.get("frame", "world")
            if frame not in {"world", "base"}:
                raise ValueError("frame_must_be_world_or_base")
            if frame == "base":
                if snapshot["robot"] is None:
                    raise ValueError("mobile_robot_state_unavailable")
                entity = in_base(entity, snapshot["robot"])
            return {**metadata, "frame": frame, "entity": entity}
        if operation == "plan_navigation":
            return {**metadata, **self._plan(snapshot, params)}
        raise ValueError("unknown_scene_query")

    def _plan(self, snapshot, params):
        if __package__:
            from .navigation import NavigationMap
        else:
            from navigation import NavigationMap
        robot, runtime = snapshot["robot"], self.runtime
        if robot is None:
            raise ValueError("mobile_robot_state_unavailable")
        if self.navigation_map is None:
            # Enclose the current locked upper body through every heading.
            low, high = np.asarray(runtime.robot.bounds)
            pos = np.asarray(robot["position_m"])
            radius = (
                max(0.42, max(math.hypot(x - pos[0], y - pos[1]) for x in (low[0], high[0]) for y in (low[1], high[1])))
                + 0.05
            )
            if runtime.indoor is not None:
                self.navigation_map = NavigationMap.from_meshes(
                    runtime.indoor.meshes,
                    floor_z=runtime.info["spawn_ground_m"][2],
                    radius=radius,
                    height=float(high[2] - runtime.info["spawn_ground_m"][2]) + 0.05,
                )
            else:
                self.navigation_map = NavigationMap.plane(robot["position_m"][:2], radius=radius)
        entity_id = params.get("entity_id")
        if entity_id:
            entity = snapshot["entities"].get(entity_id)
            if entity is None or entity["kind"] != "object" or entity["bounds_m"] is None:
                raise ValueError("navigation_target_must_be_bounded_object")
            if entity["dynamic"]:
                raise ValueError("dynamic_target_navigation_unsupported")
            plan = self.navigation_map.plan(
                robot["position_m"], bounds=entity["bounds_m"], stand_off_m=params.get("stand_off_m", 0.4)
            )
            plan.update(
                entity_id=entity_id, target_center_m=entity["center_m"], stand_off_m=params.get("stand_off_m", 0.4)
            )
        else:
            plan = self.navigation_map.plan(robot["position_m"], [params["x_m"], params["y_m"], params["yaw_rad"]])
        return {"plan": plan, "robot": robot}
