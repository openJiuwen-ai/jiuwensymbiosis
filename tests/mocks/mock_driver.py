# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Mock PiperLowLevel — satisfies the RobotDriver / JointDriver protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MockPiperDriver:
    """In-memory 6-DOF driver that tracks pose, joints, and gripper state."""

    _home_pose: Any = None
    _pose: Any = None
    _connected: bool = False
    _suction: bool = False
    _gripper_closed: bool = False
    _call_log: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self._home_pose is None:
            self._home_pose = type("Pose", (), {"x": 200.0, "y": 0.0, "z": 400.0, "rx": 0.0, "ry": 90.0, "rz": 0.0})()
        if self._pose is None:
            self._pose = type(
                "Pose",
                (),
                {
                    "x": self._home_pose.x,
                    "y": self._home_pose.y,
                    "z": self._home_pose.z,
                    "rx": self._home_pose.rx,
                    "ry": self._home_pose.ry,
                    "rz": self._home_pose.rz,
                },
            )()

    @property
    def home_pose(self) -> Any:
        return self._home_pose

    @property
    def z_min_safe(self) -> float:
        return 50.0

    @property
    def flange_z_min_safe(self) -> float:
        return 185.8  # 50.0 + 135.8 (tool_offset)

    @property
    def tool_offset_mm(self) -> float:
        return 135.8

    def close(self) -> None:
        self._connected = False

    def home(self) -> None:
        self._call_log.append("home")
        p = self._home_pose
        self._pose = type(
            "Pose",
            (),
            {
                "x": p.x,
                "y": p.y,
                "z": p.z,
                "rx": p.rx,
                "ry": p.ry,
                "rz": p.rz,
            },
        )()

    def get_pose(self) -> Any:
        return self._pose

    def move_to_pose_blocking(self, *args, **kwargs) -> None:
        self._call_log.append(f"move_to_pose_blocking({args}, {kwargs})")

    def get_angles(self) -> Any:
        return type("Joints", (), {"j": [0.0] * 6})()

    def move_joint_blocking(self, q: list[float], *, timeout_s: float = 30.0) -> None:
        self._call_log.append(f"move_joint_blocking({q})")
