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
from jiuwensymbiosis.api.defaults import pose_to_dict
from jiuwensymbiosis.errors import GraspNotConfirmedError
from jiuwensymbiosis.simulators.robosim.env import RobosimEnv


class RobosimApi(BaseRobotApi):
    """Reuse the action vocabulary; transport and coordinate conversion live in the driver."""

    env: RobosimEnv

    @implements(GET_JOINT_POSITIONS)
    def get_joint_positions(self) -> dict[str, float]:
        """Read physical joint positions in radians."""
        return self.env.low_level.get_joint_positions()

    @implements(MOVE_JOINT)
    def move_joint(self, targets: dict[str, float]) -> dict:
        """Command selected named joints and wait for measured convergence."""
        self.env.low_level.move_joints_blocking(targets)
        return {"ok": True, "joint_positions": self.get_joint_positions()}

    @implements(GET_POSE)
    def get_pose(self) -> dict[str, float]:
        """Read TCP in base-frame mm/deg."""
        pose = self.env.low_level.get_pose()
        return pose if isinstance(pose, dict) else pose_to_dict(pose)

    @implements(GOTO_POSE)
    def goto_pose(self, pose: dict[str, float]) -> dict:
        """Plan and execute a full TCP target through RoboSim."""
        self.env.low_level.move_to_pose_blocking(pose)
        return {"ok": True, "pose": self.get_pose()}

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
            raise ValueError("Only preserve orientation is supported by this adapter")
        pose = self.get_pose()
        pose.update(x=x, y=y, z=z)
        if r is not None:
            pose["rz"] = r
        return dict(self.goto_pose(pose))

    @implements(OPEN_GRIPPER)
    def open_gripper(self, width_mm: float | None = None) -> dict:
        """Open the selected gripper; the existing service has no width control."""
        self.env.low_level.set_gripper(False)
        return {"ok": True}

    @implements(CLOSE_GRIPPER)
    def close_gripper(self, force_n: float | None = None) -> dict:
        """Fail closed unless RoboSim has a grasp binding for this arm."""
        state = self.env.low_level.set_gripper(True)
        if not state or not state.get("holding") or state.get("holding_arm_id") != self.env.cfg.side:
            raise GraspNotConfirmedError("Gripper closed, but no payload binding confirms a grasp")
        return {"ok": True, "holding": state["holding"]}
