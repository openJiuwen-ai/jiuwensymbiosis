# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""``RobotDriver`` Protocol — the contract a new vendor implements.

This is the smallest surface a per-vendor ``XxxLowLevel`` must expose for the
cross-vendor scaffolding (Env wrappers, ``_common.skills``, ``_common.vision``)
to bind onto it.

Structural typing (``typing.Protocol``, not a base class) is intentional:

* Adapters compose differently — some hold a single bespoke driver, others
  wire together independent submodules — so an abstract base class would
  force inheritance and lose composability.
* Pose / joint dataclasses are vendor-specific (4-DoF vs 6-DoF).
  A Protocol expresses "has these methods" without enforcing identical
  dataclass shapes.
* Most properties (camera, suction) are *optional* capabilities. Forcing
  every driver to implement them as ``raise NotImplementedError`` stubs
  bloats adapters. Instead, ``Env.capabilities`` advertises what's available
  and the consumer checks before calling.

Implementer contract:

  1. Construct: open SDK sockets, enable robot, snapshot init pose, load
     calibration, optionally start a camera.
  2. ``get_pose`` and ``home_pose`` return your own vendor Pose dataclass
     — e.g. ``4-DoF (x, y, z, r)`` or ``6-DoF (x, y, z, rx, ry, rz)``.
     The ``XxxEnv.get_observation()`` is what flattens to ``RobotObservation.pose``.
  3. ``move_to_pose_blocking`` speaks FLANGE frame. The api layer's
     ``goto_xyzr`` is responsible for tip↔flange conversion (so the same
     ``_common.skills`` works for any tool-offset).
  4. ``close()`` must be idempotent — it's called from ``Env.disconnect``
     which itself may be invoked twice on error paths.

Optional sibling protocol (``JointDriver``) covers joint-space access for
adapters that support it.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RobotDriver(Protocol):
    """The minimum surface a per-vendor low-level driver exposes.

    Vendor Pose dataclasses are returned by ``get_pose`` / ``home_pose``.
    ``move_to_pose_blocking`` is intentionally typed ``*args, **kwargs``
    because the positional shape differs by DoF (4 or 6).
    Each call site uses the vendor-appropriate signature.
    """

    @property
    def home_pose(self) -> Any:
        """Vendor Pose dataclass for the snapshotted init/home pose."""

    # Safety bounds.
    @property
    def z_min_safe(self) -> float:
        """Tip-frame Z floor in mm (flange floor = this + ``tool_offset_mm``)."""

    @property
    def flange_z_min_safe(self) -> float:
        """Flange-frame Z floor in mm, enforced by ``move_to_pose_blocking``."""

    @property
    def tool_offset_mm(self) -> float:
        """Tool-tip offset from the flange along Z (mm), for tip↔flange conversion."""

    def close(self) -> None:
        """Release SDK resources / disable the robot. Must be idempotent."""

    def home(self) -> None:
        """Move the robot to its home pose (blocking)."""

    def get_pose(self) -> Any:
        """Return the current pose as the vendor's Pose dataclass."""

    def move_to_pose_blocking(self, *args: Any, **kwargs: Any) -> None:
        """Move to a FLANGE-frame target pose, blocking until motion completes.

        Typed ``*args, **kwargs`` because the positional shape is vendor-
        specific (4-DoF vs 6-DoF).
        """


@runtime_checkable
class JointDriver(Protocol):
    """Optional joint-space surface. Implementations may pick a subset."""

    def get_angles(self) -> Any:
        """Return current joint angles as the vendor's JointAngles dataclass."""

    def move_joint_blocking(
        self, q: list[float], *, timeout_s: float = 30.0,
    ) -> None:
        """Move to joint configuration ``q``, blocking until reached or ``timeout_s`` elapses."""


@runtime_checkable
class CameraDriver(Protocol):
    """Optional camera surface — typically delegates to ``_common.RealSenseCamera``."""

    @property
    def intrinsics(self) -> Optional[np.ndarray]:
        """3x3 camera intrinsics ``K``; ``None`` until the camera has started."""

    def grab_frames(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Grab one aligned ``(rgb_uint8, depth_m_float32)`` pair, or ``None`` if unavailable."""


@runtime_checkable
class SuctionDriver(Protocol):
    """Optional suction-gripper IO surface."""

    @property
    def suction_state(self) -> bool:
        """Last commanded suction state (True = on)."""

    @property
    def suction_di_last(self) -> Optional[int]:
        """Last suction digital-input reading, or ``None`` if unread/unsupported."""

    def set_suction(self, on: bool) -> None:
        """Turn the suction gripper on or off."""


@runtime_checkable
class GripperDriver(Protocol):
    """Optional parallel-gripper IO surface (sibling of ``SuctionDriver``)."""

    def set_gripper(self, on: bool) -> None:
        """Close (True) or open (False) the parallel gripper."""

    @property
    def gripper_state(self) -> Any:
        """Last commanded gripper state (implementation-defined; e.g. bool closed)."""


@runtime_checkable
class VisionDriver(Protocol):
    """Optional hand-eye calibration surface for eye-in-hand back-projection."""

    @property
    def tf_flange_cam(self) -> Optional[np.ndarray]:
        """4x4 flange→camera extrinsic transform, or None if uncalibrated."""

    @property
    def calibration(self) -> Optional[dict]:
        """Loaded hand-eye calibration payload, or None."""


# ---------------------------------------------------------------------------
# Composite driver types for adapters whose driver implements multiple protocols.
# A multi-protocol ``Protocol`` subclass gives true static type checking (mypy /
# pyright verify every member) plus a ``runtime_checkable`` ``isinstance`` probe
# for capability gating — replacing the former type alias which only documented
# the expected surface without enforcing it.
# ---------------------------------------------------------------------------


@runtime_checkable
class PiperFullDriver(RobotDriver, JointDriver, CameraDriver, GripperDriver, VisionDriver, Protocol):
    """Composite driver surface — union of all five vendor protocols.

    ``PiperLowLevel`` implements all five; ``PiperApi._ll()`` returns this
    type so vision reads (``tf_flange_cam`` / ``calibration`` / ``intrinsics``
    / ``grab_frames``) plus motion / gripper / camera reads are statically
    verified by mypy / pyright. ``isinstance(driver, PiperFullDriver)`` gives
    a runtime capability check for adapters that want to gate on the full
    surface.
    """

    pass
