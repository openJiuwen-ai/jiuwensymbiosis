# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Capability mixins.

Each mixin declares one capability string and the *abstract* methods that
make up that capability. A concrete API class inherits from the mixins it
supports and provides implementations.

The methods are decorated with ``@robot_tool`` already, so subclasses need
NOT decorate the overrides — the metadata propagates with the function.

A mixin's methods raise ``NotImplementedError`` by default. The framework
intentionally does *not* enforce 100% override at class-definition time; if
your robot only supports a subset of a capability (e.g. has motion but
no R-axis), override the supported methods and have the others raise.
"""

from __future__ import annotations

from typing import Any, Optional

from jiuwensymbiosis.api.decorators import robot_tool


# =============================================================================
# Motion
# =============================================================================
class MotionMixin:
    """Cartesian motion capability mixin."""

    capability = "motion.cartesian"

    @robot_tool(desc="Return to the home pose. Always safe.", tags=["motion"])
    def home(self) -> None:
        """Return to the home pose."""
        raise NotImplementedError

    @robot_tool(desc="Get current end-effector pose in mm/deg, base frame.")
    def get_pose(self) -> dict:
        """Get current end-effector pose."""
        raise NotImplementedError

    @robot_tool(desc="Get the home pose constants (read-only) for this robot.")
    def get_home_pose(self) -> dict:
        """Get the home pose constants."""
        raise NotImplementedError

    @robot_tool(
        desc="Move the end-effector tip to absolute (x, y, z[, r]) in mm/deg, base frame. "
        "If r is omitted, current r is preserved.",
        tags=["motion"],
    )
    def goto_xyzr(self, x: float, y: float, z: float, r: Optional[float] = None) -> None:
        """Move the end-effector tip to an absolute Cartesian pose."""
        raise NotImplementedError


class JointMotionMixin:
    """Joint-space motion capability mixin."""

    capability = "motion.joint"

    @robot_tool(desc="Move to a joint configuration q (rad or deg per robot convention).", tags=["motion"])
    def move_joint(self, q: list[float]) -> None:
        """Move to a joint configuration."""
        raise NotImplementedError


# =============================================================================
# Grasp
# =============================================================================
class SuctionMixin:
    """Suction grasp capability mixin."""

    capability = "grasp.suction"

    @robot_tool(desc="Turn suction ON. Should be called only after the tip is on/near the target.", tags=["grasp"])
    def activate_suction(self) -> dict:
        """Turn suction ON."""
        raise NotImplementedError

    @robot_tool(desc="Turn suction OFF — releases whatever is held.", tags=["grasp"])
    def deactivate_suction(self) -> dict:
        """Turn suction OFF."""
        raise NotImplementedError


class ParallelGripperMixin:
    """Parallel gripper capability mixin."""

    capability = "grasp.parallel"

    @robot_tool(desc="Open the parallel gripper to width_mm.", tags=["grasp"])
    def open_gripper(self, width_mm: float = 80.0) -> dict:
        """Open the parallel gripper."""
        raise NotImplementedError

    @robot_tool(desc="Close the parallel gripper, optionally with a target force in N.", tags=["grasp"])
    def close_gripper(self, force_n: Optional[float] = None) -> dict:
        """Close the parallel gripper."""
        raise NotImplementedError


# =============================================================================
# Vision
# =============================================================================
class VisionMixin:
    """Vision and object detection capability mixin."""

    capability = "vision.detection"

    @robot_tool(
        desc="One-shot: detect `object_name` in the live frame, project "
        "to base XYZ via depth+calibration. Returns "
        '{"ok": bool, "object": str, "position": [x,y,z]_mm, "grasp_z": float, '
        '"grasp_position": [x,y,z]_mm, "place_z": float, "place_position": [x,y,z]_mm, '
        '"score": float, "pixel_uv": [u,v], "depth_m": float}.',
    )
    def get_grasp_info_simple(self, object_name: str) -> dict:
        """Detect an object and return its 3D position."""
        raise NotImplementedError

    @robot_tool(desc="Convert a pixel (u,v) at known depth to base-frame XYZ in mm.")
    def pixel_to_base_xyz(self, u: float, v: float, depth_m: float) -> dict:
        """Convert a pixel to base-frame XYZ."""
        raise NotImplementedError

    @robot_tool(desc="Grab the latest RGB frame as numpy HxWx3 (rarely needed by the agent itself).")
    def get_image(self) -> Any:
        """Grab the latest RGB frame."""
        raise NotImplementedError

    @robot_tool(desc="Higher-level scene analysis with prompt grounded on object_name.")
    def analyze_scene(self, object_name: Optional[str] = None) -> dict:
        """Run scene analysis for the given object."""
        raise NotImplementedError
