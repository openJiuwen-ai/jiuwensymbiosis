# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""``JointTransport`` — the vendor seam for a joint-level SDK.

Some arms expose a real Cartesian interface in firmware (their driver wraps a
``move_linear(pose)`` and satisfies ``RobotDriver`` directly). Others only take
joint-motor targets plus an observation stream; the Cartesian layer (FK/IK,
waypoints, arrival polling, safety) has to live above the SDK. ``JointTransport``
is that lower seam: the *only* part an adapter for a joint-level arm must hand-
write. Everything above it is provided once by ``KinematicArmDriver``.

Design mirrors ``perception.vision.default_get_grasp_info_simple`` — the
framework owns the pipeline; the adapter injects the single vendor-specific
piece. Structural typing (``Protocol``) keeps the SDK object free of any
framework base class.

Contract:
  * ``open`` / ``close`` are the SDK connection lifecycle; ``close`` is idempotent.
  * ``read_arm_joints`` / ``send_arm_joints`` carry the *arm* joints only, in the
    SDK's native unit, in a fixed order the adapter and its ``KinematicSpec``
    agree on. The end effector (if any) is a separate channel and never enters
    this vector, so FK/IK and joint-limit checks see arm joints alone.
  * ``send_arm_joints`` returns the target the SDK actually accepted (some SDKs
    clamp per-command deltas); arrival is still judged from ``read_arm_joints``.
  * ``read_effector`` / ``send_effector`` are one effector-neutral two-state
    channel — a parallel gripper and a suction cup both ride it (engaged =
    holding), so the same transport serves either grasp capability.

A transport that also carries a camera / hand-eye calibration may additionally
expose ``grab_frames`` / ``intrinsics`` / ``tf_flange_cam`` / ``calibration``;
``KinematicArmDriver`` forwards those (duck-typed) so vision composes when present.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class JointTransport(Protocol):
    """Minimal joint-level SDK surface a ``KinematicArmDriver`` drives."""

    def open(self) -> None:
        """Open the SDK connection (serial/CAN/socket). Blocking."""

    def close(self) -> None:
        """Release the SDK connection. Must be idempotent."""

    def precheck(self) -> None:
        """Validate SDK-specific preconditions (calibration file, feature names).

        Called after ``open`` and before the driver is marked connected; raise to
        abort the connection.
        """

    def read_arm_joints(self) -> list[float]:
        """Current arm-joint values in native unit, fixed order (no gripper)."""

    def send_arm_joints(self, q: list[float]) -> list[float]:
        """Command arm-joint targets (no gripper); return the accepted targets."""

    def read_effector(self) -> float:
        """Current end-effector value in native unit (e.g. mm width, 0..100 percent, on/off)."""

    def send_effector(self, pos: float) -> float:
        """Command an end-effector value (engaged or released); return the accepted value."""
