"""Disposable cuRobo process. Never imports Genesis or owns physical entities."""

from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
from arm_collision import CollisionGate
from arm_planning import CuroboPlanner


def main(path: Path):
    data = json.loads(path.read_text())
    planner = CuroboPlanner(data["manifest"], data["profile"], path.parent)
    stage = "snapshot"
    backend_success = False
    try:
        captured = data["captured"]
        captured["geometries"] = {int(k): v for k, v in captured["geometries"].items()}
        for geom in captured["geometries"].values():
            for field in ("vertices", "faces", "pose"):
                if field in geom:
                    geom[field] = np.array(geom[field], dtype=int if field == "faces" else float)
            if geom.get("payload"):
                # Enclose loaded geometry in tool frame; independently check full geometry below.
                points = geom["vertices"] @ geom["pose"][:3, :3].T + geom["pose"][:3, 3]
                center = (points.max(0) + points.min(0)) / 2
                radius = float(np.linalg.norm(points - center, axis=1).max()) + 0.003
                planner.profile["collision_spheres"].setdefault(geom["link"], []).append(
                    {"center": center.tolist(), "radius": radius}
                )
        gate = CollisionGate(planner.root, planner.profile, captured)
        coverage = gate.coverage()
        (path.parent / "coverage.json").write_text(json.dumps(coverage, indent=2, allow_nan=False))
        stage = "initial_collision_check"
        gate.check_state(data["state"])
        request = data["request"]
        stage = "curobo_planning"
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
        backend_success = True
        stage = "path_collision_check"
        validation = gate.check(trajectory)
        result = {
            "ok": True,
            "backend_success": True,
            "timing": {
                "requested_speed": request.get("speed", 1.0),
                "retimed_dt_s": trajectory.dt,
                "effective_speed_vs_curobo_samples": 0.025 / trajectory.dt,
                "command_derivative_maxima": [values.tolist() for values in trajectory.derivative_maxima()],
            },
            "collision_validation": validation,
            "trajectory": {field.name: getattr(trajectory, field.name) for field in fields(trajectory)},
        }
    except Exception as exc:
        result = {
            "ok": False,
            "backend_success": backend_success,
            "failure_stage": stage,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    finally:
        planner.close()
    (path.parent / "result.json").write_text(
        json.dumps(
            result, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else dict(v), indent=2, allow_nan=False
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
