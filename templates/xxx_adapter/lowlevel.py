# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""XxxDriver — low-level hardware communication.

Replace this file with your hardware-specific driver (serial, CAN, socket, etc.).
A plain Python class satisfying the ``RobotDriver`` Protocol
(adapters/_common/protocol.py) — add the sibling protocols for the capabilities
you declare. The Env verbs delegate here:

    connect() / disconnect()            — idempotent lifecycle (called by Env)
    get_pose() → x,y,z,rx,ry,rz         — current pose       (env.get_flange_pose)
    home()                              — blocking home move  (env.home)
    move_to_pose_blocking(pose)         — blocking move       (env.move_to_flange)
    move_joint_blocking(q)              — [motion.joint]      (env.move_joint)
    set_gripper(on) / set_suction(on)   — [grasp.*]           (env.set_end_effector)
    grab_frames() → (rgb, depth) | None — [vision.*] camera frames
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional


class XxxDriver:
    """Hardware communication driver — replace with real hardware code.

    This mock implementation tracks pose in memory so you can verify
    the adapter integration before connecting real hardware.
    """

    def __init__(self) -> None:
        # [必填] 内部位姿状态 (替换为真实硬件状态读取)
        self._pose: dict[str, float] = {
            "x": 200.0, "y": 0.0, "z": 250.0,
            "rx": 0.0, "ry": 90.0, "rz": 0.0,
        }
        # [必填] home 位姿对象 (get_pose 返回同类型)
        self.home_pose = SimpleNamespace(
            x=200.0, y=0.0, z=250.0, rx=0.0, ry=90.0, rz=0.0,
        )
        # [必填-仅 motion.*] 工具末端偏移 mm (flange → tip)
        self.tool_offset_mm: float = 0.0

        self._connected: bool = False

    # ============================== 生命周期 [必填] ==============================

    def connect(self) -> None:
        """Open hardware connection. Must be idempotent."""
        # TODO: Replace with real hardware connection
        self._connected = True

    def disconnect(self) -> None:
        """Release hardware. Must be idempotent and safe at any state."""
        # TODO: Replace with real hardware shutdown
        self._connected = False

    # ============================== 运动 [选填-仅 motion.*] ==============================

    def get_pose(self) -> Any:
        """Return current tip pose. Suggested type: SimpleNamespace with x,y,z,rx,ry,rz.

        For SCARA (4-DOF), rx/ry may be fixed at 0 or omitted.
        """
        # TODO: Read real pose from hardware
        p = self._pose
        return SimpleNamespace(
            x=p["x"], y=p["y"], z=p["z"],
            rx=p["rx"], ry=p["ry"], rz=p["rz"],
        )

    def home(self) -> None:
        """Execute homing sequence. Blocking."""
        # TODO: Send home command to hardware
        hp = self.home_pose
        self._pose = {"x": hp.x, "y": hp.y, "z": hp.z,
                      "rx": hp.rx, "ry": hp.ry, "rz": hp.rz}

    def move_to_pose_blocking(self, pose: Any) -> None:
        """Blocking Cartesian move to <pose>. pose has x,y,z,rx,ry,rz attributes."""
        # TODO: Send motion command to hardware, wait for completion
        self._pose["x"] = float(pose.x)
        self._pose["y"] = float(pose.y)
        self._pose["z"] = float(pose.z)
        self._pose["rx"] = float(getattr(pose, "rx", 0.0))
        self._pose["ry"] = float(getattr(pose, "ry", 90.0))
        self._pose["rz"] = float(getattr(pose, "rz", 0.0))

    def move_joint_blocking(self, q: list[float]) -> None:
        """Blocking joint-space move. [选填-仅 motion.joint]"""
        # TODO: Send joint command to hardware
        pass

    # ============================== 末端执行器 [选填-仅 grasp.*] ==============================

    def set_gripper(self, on: bool) -> None:
        """Set gripper/suction state.
        on=True  → close/grip/activate
        on=False → open/release/deactivate
        """
        # TODO: Send gripper command to hardware
        pass

    # ============================== 传感器 [选填-仅 vision.*] ==============================

    def grab_frames(self) -> Optional[tuple]:
        """Grab one RGB + depth frame pair.
        Returns (rgb: HxWx3 uint8 ndarray, depth: HxW float32 ndarray) or None.
        """
        # TODO: Capture frames from camera
        return None
