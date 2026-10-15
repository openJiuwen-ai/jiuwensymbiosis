"""Shared ActionSpec bindings for the RoboSim MOZ1 arm."""

from __future__ import annotations

from typing import Literal

from jiuwensymbiosis.api.actions import (
    CLOSE_GRIPPER,
    GET_JOINT_POSITIONS,
    GET_POSE,
    GOTO_POSE,
    GOTO_XYZR,
    MOVE_JOINT,
    OPEN_GRIPPER,
    implements,
)
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.simulators.robosim.env import RobosimEnv


class RobosimApi(BaseRobotApi):
    """Reuse the action vocabulary; transport and coordinate conversion live in the driver."""

    env: RobosimEnv

    @implements(GET_JOINT_POSITIONS)
    def get_joint_positions(self) -> dict[str, float]:
        """Read physical joint positions in radians."""
        return self.env.get_joint_positions()

    @implements(MOVE_JOINT)
    def move_joint(self, targets: dict[str, float]) -> dict:
        """Command selected named joints and wait for measured convergence."""
        measured = self.env.move_joint(targets)
        return {"ok": True, "joint_positions": measured}

    @implements(GET_POSE)
    def get_pose(self) -> dict[str, float]:
        """Read TCP in base-frame mm/deg."""
        return self.env.get_tcp_pose()

    @implements(GOTO_POSE)
    def goto_pose(self, pose: dict[str, float]) -> dict:
        """Plan and execute a full TCP target through RoboSim."""
        measured = self.env.move_to_tcp(pose)
        return {"ok": True, "pose": measured}

    @implements(GOTO_XYZR)
    def goto_xyzr(
        self,
        x: float,
        y: float,
        z: float,
        r: float | None = None,
        orientation_policy: Literal["preserve"] | None = None,
    ) -> dict:
        """Translate while retaining measured tilt and optionally setting yaw."""
        if orientation_policy not in {None, "preserve"}:
            raise ValueError("Only preserve orientation is supported by this backend")
        pose = self.get_pose()
        pose.update(x=x, y=y, z=z)
        if r is not None:
            pose["rz"] = r
        return dict(self.goto_pose(pose))

    @implements(OPEN_GRIPPER)
    def open_gripper(self, width_mm: float | None = None) -> dict:
        """Open the selected gripper; the existing service has no width control."""
        self.env.open_gripper()
        return {"ok": True}

    @implements(CLOSE_GRIPPER)
    def close_gripper(self, force_n: float | None = None) -> dict:
        """Fail closed unless RoboSim has a grasp binding for this arm."""
        state = self.env.close_gripper()
        return {"ok": True, "holding": state["holding"]}
