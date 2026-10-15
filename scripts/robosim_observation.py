"""RoboSim-side read-only observation extension; imported only by the launcher.

This module intentionally does not import jiuwensymbiosis: RoboSim and the agent
run in separate environments. Articulation access uses RoboSim's main-thread queue.
"""

from __future__ import annotations

import asyncio
import math
from typing import Literal


def install_observation_route(app) -> None:
    """Expose measured arm state without changing the RoboSim checkout."""
    from fastapi import HTTPException
    from pxr import Usd, UsdGeom, UsdPhysics
    from sim_service import sim_world
    from sim_service.robots.manager import robot_manager

    @app.get("/symbiosis/observation")
    async def observation(side: Literal["left", "right"] = "left") -> dict:
        """Read selected arm joints, TCP and base frame from the live USD stage."""
        try:
            robot = robot_manager.current()
            state = await robot.get_state()
            spec = robot_manager.get_spec(state.robot_type)
            resource = next(r for r in spec.execution_resources if r.resource_id == f"{side}_arm")
            joints = await robot.get_execution_resource_joint_positions(resource.resource_id)
            tcp = await robot.get_execution_resource_pose(resource.resource_id)

            def read_geometry() -> dict:
                """Read base transform and authored joint limits on the Kit thread."""
                world = sim_world.get_world()
                root = f"/World/Robots/{state.robot_type}"
                base_path = f"{root}/{spec.default_config['parking_base_rigid_body_path']}"
                base = world.stage.GetPrimAtPath(base_path)
                if not base.IsValid():
                    raise RuntimeError(f"Base prim unavailable: {base_path}")
                matrix = UsdGeom.Xformable(base).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                limits = {}
                for prim in Usd.PrimRange(world.stage.GetPrimAtPath(root)):
                    if prim.GetName() in resource.joint_names and prim.IsA(UsdPhysics.RevoluteJoint):
                        joint = UsdPhysics.RevoluteJoint(prim)
                        low, high = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
                        if low is None or high is None:
                            continue
                        if math.isfinite(low) and math.isfinite(high):
                            limits[prim.GetName()] = [math.radians(low), math.radians(high)]
                # Gf uses row vectors; the agent uses column-vector homogeneous transforms.
                return {
                    "world_from_base": [[float(matrix[j][i]) for j in range(4)] for i in range(4)],
                    "joint_limits": limits,
                    "simulation_time_s": float(world.current_time),
                    "base_frame": base_path,
                }

            geometry = await asyncio.to_thread(sim_world.run_on_main_thread, read_geometry)
            return {
                "bridge_version": 1,
                "robot_type": state.robot_type,
                "resource_id": resource.resource_id,
                "joint_positions": dict(zip(resource.joint_names, joints, strict=True)),
                "tcp_world": tcp.model_dump(mode="json"),
                "reported_state": state.model_dump(mode="json"),
                **geometry,
            }
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
