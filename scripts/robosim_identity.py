"""Read-only service/robot identity bridge, without importing the Agent package."""

from __future__ import annotations

import uuid
from typing import Any


class RuntimeIdentity:
    """Identify a service incarnation and detect replacement of a live robot object."""

    def __init__(self) -> None:
        self.service_instance_id = uuid.uuid4().hex
        self._robot: Any = None
        self._robot_id: str | None = None

    def snapshot(self, robot: Any) -> dict:
        """Return robot identity independently of its model name or joint state."""
        if robot is not self._robot:
            self._robot = robot
            self._robot_id = uuid.uuid4().hex if robot is not None else None
        return {
            "bridge_version": 1,
            "service_instance_id": self.service_instance_id,
            "robot_instance_id": self._robot_id,
        }


def install_identity_route(app: Any) -> None:
    """Expose identity even before a robot has been loaded; no scene writes."""
    from sim_service.robots.action_errors import RobotNotLoadedError
    from sim_service.robots.manager import robot_manager

    identity = RuntimeIdentity()

    @app.get("/symbiosis/workspace")
    async def workspace_identity() -> dict:
        try:
            robot = robot_manager.current()
        except RobotNotLoadedError:
            robot = None
        return identity.snapshot(robot)
