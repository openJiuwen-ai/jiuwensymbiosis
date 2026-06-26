# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Capability mixins.

Each mixin declares one capability string and the methods that make up that
capability. A concrete API class inherits from the mixins it supports.

The methods are decorated with ``@robot_tool`` already, so subclasses need
NOT decorate the overrides — the metadata propagates with the function.

Default behaviour
-----------------
Motion / joint / grasp methods, plus ``get_image``, ship **working default
implementations that delegate to the Env contract verbs** (``self.env.home`` /
``move_to_flange`` / ``move_joint`` / ``set_end_effector`` / ``get_flange_pose``
/ ``home_pose`` / ``tool_offset_mm`` / ``grab_rgb``). A robot whose body matches
the common case (top-down tip, tip == flange, two-state end effector) therefore
needs to write *no* api code for these — composing the mixins is enough. Override
a method only when the body has special geometry (e.g. a tilted tool, a tool offset).

The *high-level vision* methods (``get_grasp_info_simple`` / ``pixel_to_base_xyz``
/ ``analyze_scene``) cannot have a generic default — they depend on the adapter's
detector client and hand-eye calibration — so they stay abstract and raise
``NotImplementedError`` until the adapter provides them.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from jiuwensymbiosis.api.decorators import robot_tool

if TYPE_CHECKING:
    # Mixins are composed into BaseRobotApi subclasses, which set `self.env`.
    # Declared here only for type checking; runtime attribute is provided by
    # the composing class (see BaseRobotApi.__init__).
    from jiuwensymbiosis.env.base import BaseRobotEnv


def _pose_to_dict(pose: Any) -> dict:
    """Best-effort vendor-pose object → dict.

    Tolerates both the SCARA convention (``.r``) and the 6-DoF convention
    (``.rx/.ry/.rz``); only the fields the pose actually exposes are emitted.
    """
    out: dict[str, float] = {}
    for key in ("x", "y", "z", "rx", "ry", "rz", "r"):
        val = getattr(pose, key, None)
        if val is not None:
            out[key] = float(val)
    return out


# =============================================================================
# Motion
# =============================================================================
class MotionMixin:
    """Cartesian motion capability mixin."""

    env: BaseRobotEnv  # provided by the composing BaseRobotApi subclass
    capability = "motion.cartesian"

    @robot_tool(desc="Return to the home pose. Always safe.", tags=["motion"])
    def home(self) -> None:
        """Return to the home pose (delegates to the Env verb)."""
        self.env.home()

    @robot_tool(desc="Get current end-effector pose in mm/deg, base frame.")
    def get_pose(self) -> dict:
        """Current pose. Default reports the flange pose (assumes tip == flange;
        override when a tool offset applies).
        """
        return _pose_to_dict(self.env.get_flange_pose())

    @robot_tool(desc="Get the home pose constants (read-only) for this robot.")
    def get_home_pose(self) -> dict:
        """Home pose constants, read from the env."""
        return _pose_to_dict(self.env.home_pose)

    @robot_tool(
        desc="Move the end-effector tip to absolute (x, y, z[, r]) in mm/deg, base frame. "
        "If r is omitted, current r is preserved.",
        tags=["motion"],
    )
    def goto_xyzr(self, x: float, y: float, z: float, r: float | None = None) -> None:
        """Move the tip to an absolute Cartesian pose. Default is a top-down move
        (tip == flange, rx=180, ry=0); override for a tool offset or a tilted tool.
        """
        if r is None:
            cur = self.env.get_flange_pose()
            r = getattr(cur, "rz", getattr(cur, "r", 0.0))
        self.env.move_to_flange(SimpleNamespace(x=float(x), y=float(y), z=float(z), rx=180.0, ry=0.0, rz=float(r)))


class JointMotionMixin:
    """Joint-space motion capability mixin."""

    env: BaseRobotEnv  # provided by the composing BaseRobotApi subclass
    capability = "motion.joint"

    @robot_tool(desc="Move to a joint configuration q (rad or deg per robot convention).", tags=["motion"])
    def move_joint(self, q: list[float]) -> None:
        """Move to a joint configuration (delegates to the Env verb)."""
        self.env.move_joint(q)


# =============================================================================
# Grasp
# =============================================================================
class SuctionMixin:
    """Suction grasp capability mixin."""

    env: BaseRobotEnv  # provided by the composing BaseRobotApi subclass
    capability = "grasp.suction"

    @robot_tool(desc="Turn suction ON. Should be called only after the tip is on/near the target.", tags=["grasp"])
    def activate_suction(self) -> dict:
        """Turn suction ON (delegates to the Env end-effector verb)."""
        self.env.set_end_effector(True)
        return {"ok": True, "state": "on"}

    @robot_tool(desc="Turn suction OFF — releases whatever is held.", tags=["grasp"])
    def deactivate_suction(self) -> dict:
        """Turn suction OFF (delegates to the Env end-effector verb)."""
        self.env.set_end_effector(False)
        return {"ok": True, "state": "off"}


class ParallelGripperMixin:
    """Parallel gripper capability mixin."""

    env: BaseRobotEnv  # provided by the composing BaseRobotApi subclass
    capability = "grasp.parallel"

    @robot_tool(desc="Open the parallel gripper to width_mm.", tags=["grasp"])
    def open_gripper(self, width_mm: float = 80.0) -> dict:
        """Open the gripper (delegates to the Env end-effector verb). ``width_mm``
        is accepted for API parity; bodies with width control override this.
        """
        self.env.set_end_effector(False)
        return {"ok": True, "state": "open"}

    @robot_tool(desc="Close the parallel gripper, optionally with a target force in N.", tags=["grasp"])
    def close_gripper(self, force_n: float | None = None) -> dict:
        """Close the gripper (delegates to the Env end-effector verb). ``force_n``
        is accepted for API parity; bodies with force control override this.
        """
        self.env.set_end_effector(True)
        return {"ok": True, "state": "closed"}


# =============================================================================
# Vision
# =============================================================================
class VisionMixin:
    """Vision and object detection capability mixin.

    ``get_image`` has a working default (raw frame grab); the higher-level
    detection methods need the adapter's detector + calibration and stay abstract.
    """

    env: BaseRobotEnv  # provided by the composing BaseRobotApi subclass

    capability = "vision.detection"

    @robot_tool(
        desc="One-shot: detect `object_name` in the live frame, project "
        "to base XYZ via depth+calibration. Returns "
        '{"ok": bool, "object": str, "position": [x,y,z]_mm, "grasp_z": float, '
        '"grasp_position": [x,y,z]_mm, "place_z": float, "place_position": [x,y,z]_mm, '
        '"score": float, "pixel_uv": [u,v], "depth_m": float}.',
    )
    def get_grasp_info_simple(self, object_name: str) -> dict:
        """Detect an object and return its 3D grasp/place geometry.

        No generic default: requires the adapter's detector client and hand-eye
        calibration. Adapters must implement this.
        """
        raise NotImplementedError

    @robot_tool(desc="Convert a pixel (u,v) at known depth to base-frame XYZ in mm.")
    def pixel_to_base_xyz(self, u: float, v: float, depth_m: float) -> dict:
        """Reproject a pixel to base-frame XYZ.

        No generic default: requires the adapter's hand-eye calibration.
        """
        raise NotImplementedError

    @robot_tool(desc="Grab the latest RGB frame as numpy HxWx3 (rarely needed by the agent itself).")
    def get_image(self) -> Any:
        """Latest RGB frame, or None if no camera (delegates to the env)."""
        return self.env.grab_rgb()

    @robot_tool(desc="Higher-level scene analysis with prompt grounded on object_name.")
    def analyze_scene(self, object_name: str | None = None) -> dict:
        """Scene analysis grounded on ``object_name``.

        No generic default: requires the adapter's detector client.
        """
        raise NotImplementedError
