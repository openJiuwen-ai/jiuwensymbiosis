"""Scene-owned serial cuRobo worker. Never owns physical entities or cached plans."""

from __future__ import annotations

import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import fields
from pathlib import Path

import numpy as np
from arm_planning import CuroboPlanner, timed


def prepare_payload_slots(planner, captured):
    """Keep CUDA tensor shapes stable when a configured hand picks or releases."""
    planner.payload_links = set()
    links = {link.get("name") for link in planner.root.findall("link")}
    covered = set()
    for index, config in enumerate(planner.profile.get("effectors", {}).values()):
        tool = config["tool"]
        payload_link = f"__carried_payload_{index}"
        while payload_link in links:
            payload_link += "_"
        links.add(payload_link)
        ET.SubElement(planner.root, "link", name=payload_link)
        joint_name = payload_link + "_fixed"
        while joint_name in planner.joints:
            joint_name += "_"
        joint = ET.SubElement(planner.root, "joint", name=joint_name, type="fixed")
        ET.SubElement(joint, "parent", link=tool)
        ET.SubElement(joint, "child", link=payload_link)
        planner.joints[joint_name] = joint
        planner.payload_links.add(payload_link)
        payloads = [g for g in captured["geometries"].values() if g.get("payload") and g["link"] == tool]
        if len({g["object_id"] for g in payloads}) > 1:
            raise ValueError("one carried body per configured tool required")
        # cuRobo reserves inactive attachment spheres with a negative radius.
        sphere = {"center": [0.0, 0.0, 0.0], "radius": -100.0}
        if payloads:
            points = np.concatenate([g["vertices"] @ g["pose"][:3, :3].T + g["pose"][:3, 3] for g in payloads])
            center = (points.max(0) + points.min(0)) / 2
            sphere = {"center": center.tolist(), "radius": float(np.linalg.norm(points - center, axis=1).max()) + 0.003}
            covered.add(tool)
        planner.profile["collision_spheres"][payload_link] = [sphere]
        planner.profile.setdefault("self_collision_ignore", {})[payload_link] = sum(config["contact_groups"], [])
    if {g["link"] for g in captured["geometries"].values() if g.get("payload")} - covered:
        raise ValueError("payload has no configured effector")


def main(path: Path, previous=None):
    begin = time.monotonic()
    phases = {}
    data = json.loads(path.read_text())
    planner = None
    stage = "snapshot"
    backend_success = False
    try:
        with timed(phases, "model_load_s"):
            planner = CuroboPlanner(data["manifest"], data["profile"], path.parent)
        captured = data["captured"]
        captured["geometries"] = {int(k): v for k, v in captured["geometries"].items()}
        for geom in captured["geometries"].values():
            for field in ("vertices", "pose"):
                if field in geom:
                    geom[field] = np.array(geom[field], dtype=float)
        prepare_payload_slots(planner, captured)
        # cuRobo owns collision-aware planning. Do not add mesh-gated sphere
        # exclusions or a separate geometry veto before returning its path.
        phases["snapshot_prepare_s"] = time.monotonic() - begin
        if previous is not None:
            planner.backend, previous.backend = previous.backend, None
            planner.key = previous.key
            planner.backend_structure = previous.backend_structure
        request = data["request"]
        if request.get("warmup"):
            stage = "curobo_warmup"
            planner.prepare_motion(data["group"], data["state"], captured["world"])
            result = {"ok": True, "operation": "warmup", "robot_motion": False}
        else:
            stage = "curobo_planning"
            result = plan_request(planner, data)
            backend_success = True
    except Exception as exc:
        result = {
            "ok": False,
            "backend_success": backend_success,
            "failure_stage": stage,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    if planner is not None:
        phases.update(planner.prepare_timing)
        phases.update(planner.plan_timing)
    phases.update(worker_total_s=time.monotonic() - begin, worker_pid=os.getpid())
    result.setdefault("timing", {}).update(phases)
    pending = path.parent / "result.pending.json"
    pending.write_text(
        json.dumps(
            result, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else dict(v), indent=2, allow_nan=False
        )
    )
    pending.replace(path.parent / "result.json")
    if previous is not None:
        previous.close()
    return planner


def plan_request(planner, data):
    """Run cuRobo once and serialize its path without a separate collision gate."""
    request = data["request"]
    captured = data["captured"]
    trajectory = planner.plan(
        data["group"],
        data["state"],
        captured["world"],
        data["binding"],
        targets=request.get("targets"),
        poses=request.get("poses"),
        speed=request.get("speed", 1.0),
        timeout_s=60.0,
    )
    return {
        "ok": True,
        "backend_success": True,
        "timing": {
            "requested_speed": request.get("speed", 1.0),
            "retimed_dt_s": trajectory.dt,
            "trajectory_duration_s": trajectory.duration,
            "effective_speed_vs_curobo_samples": 0.025 / trajectory.dt,
            "command_derivative_maxima": [values.tolist() for values in trajectory.derivative_maxima()],
        },
        "collision_validation": {
            "method": "curobo_motion_planner",
            "self_collision_check": True,
            "environment_collision_check": True,
            "independent_geometry_check": False,
        },
        "trajectory": {field.name: getattr(trajectory, field.name) for field in fields(trajectory)},
    }


def serve():
    planners = {}
    try:
        for line in sys.stdin:
            if line.strip():
                path = Path(line.strip())
                group = json.loads(path.read_text())["group"]
                planners[group] = main(path, planners.get(group))
    finally:
        for planner in planners.values():
            if planner is not None:
                planner.close()


if __name__ == "__main__":
    serve()
