# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Driver Protocols — the contract a new vendor implements, split by capability.

These are the surfaces a per-vendor ``XxxLowLevel`` exposes so the cross-vendor
scaffolding (Env wrappers, ``perception.vision``) can bind onto it. There is one
protocol per capability rather than one god-protocol: a mobile dual-arm body has
no flange pose, and a bench arm has no wheels, so a single contract would force
one of them into ``NotImplementedError`` stubs. ``RobotDriver`` therefore holds
only what every driver has — ``close()`` — and each capability adds its own
sibling protocol, mirroring ``CAPABILITY_DRIVER_MEMBERS`` in
``adapters/_common/capability_spec.py`` row for row.

Structural typing (``typing.Protocol``, not a base class) is intentional:

* Adapters compose differently — some hold a single bespoke driver, others
  wire together independent submodules — so an abstract base class would
  force inheritance and lose composability.
* Pose / joint dataclasses are vendor-specific (4-DoF vs 6-DoF).
  A Protocol expresses "has these methods" without enforcing identical
  dataclass shapes.
* Capabilities are *optional*. ``Env.capabilities`` advertises what's
  available and the consumer checks before calling.

Implementer contract:

  1. Construct: open SDK sockets, enable robot, snapshot init pose, load
     calibration, optionally start a camera.
  2. ``get_pose`` and ``home_pose`` return your own vendor Pose dataclass
     — e.g. ``4-DoF (x, y, z, r)`` or ``6-DoF (x, y, z, rx, ry, rz)``.
     The ``XxxEnv.get_observation()`` is what flattens to ``RobotObservation.pose``.
  3. ``move_to_pose_blocking`` speaks FLANGE frame. The api layer's
     ``goto_xyzr`` is responsible for tip↔flange conversion (so the shared
     motion tools work for any tool-offset).
  4. ``close()`` must be idempotent — it's called from ``Env.disconnect``
     which itself may be invoked twice on error paths.

Mobile-body protocols name the verb the way ``BaseRobotEnv`` does, because an
Env without its own implementation forwards to the driver under the same name.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RobotDriver(Protocol):
    """What every driver has, whatever it drives: a releasable connection."""

    def close(self) -> None:
        """Release SDK resources / disable the robot. Must be idempotent."""


@runtime_checkable
class CartesianDriver(RobotDriver, Protocol):
    """Flange-frame Cartesian motion — ``motion.cartesian``.

    Vendor Pose dataclasses are returned by ``get_pose`` / ``home_pose``.
    ``move_to_pose_blocking`` takes the structured ``pose`` object first
    (the vendor Pose dataclass, 4- or 6-DoF), with vendor extensions in
    ``*args``/``**kwargs`` after it.
    """

    @property
    def home_pose(self) -> Any:
        """Vendor Pose dataclass for the snapshotted init/home pose."""

    # Safety bounds.
    @property
    def z_min_safe(self) -> float:
        """Target TIP Z floor in mm; not a whole-robot or trajectory collision check.

        Flange-to-tip geometry is adapter-specific.
        """

    @property
    def flange_z_min_safe(self) -> float:
        """Conservative scalar flange Z reference in mm for the adapter's tool.

        A driver may accept a lower flange target when checking its complete
        target pose proves that TIP Z still satisfies ``z_min_safe``.
        """

    @property
    def tool_offset_mm(self) -> float:
        """Tool-tip offset in mm; the adapter declares the axis/frame convention."""

    def home(self) -> None:
        """Move the robot to its home pose (blocking)."""

    def get_pose(self) -> Any:
        """Return the current pose as the vendor's Pose dataclass."""

    def move_to_pose_blocking(self, pose: Any, *args: Any, **kwargs: Any) -> None:
        """Move to a FLANGE-frame target pose, blocking until motion completes.

        ``pose`` is the structured vendor Pose object (``x,y,z,rx,ry,rz`` for
        6-DoF, ``x,y,z,r`` for 4-DoF SCARA). Making it a named positional
        parameter — rather than burying it in ``*args`` — turns a forgotten
        pose into a static error instead of a runtime crash. Vendor extensions
        (``sync_timeout_s``, ``joint=True``, ...) ride in ``*args``/``**kwargs``
        after it.
        """


@runtime_checkable
class JointDriver(Protocol):
    """Optional INDEXED joint-space surface — the whole joint vector, in chain order.

    The sibling ``NamedJointDriver`` is the same capability (``motion.joint``) in the other
    encoding; see it for when a body needs that one instead. A single-chain arm implements
    this slice, and units are the body's own convention (the one ``move_joint`` states), so a
    consumer reads them off the body rather than assuming.
    """

    def get_angles(self) -> Any:
        """Return current joint angles as the vendor's JointAngles dataclass."""

    def move_joint_blocking(
        self,
        q: list[float],
        *,
        timeout_s: float = 30.0,
    ) -> None:
        """Move to joint configuration ``q``, blocking until reached or ``timeout_s`` elapses."""


@runtime_checkable
class NamedJointDriver(Protocol):
    """Optional NAMED joint-space surface — the same ``motion.joint`` capability as
    ``JointDriver``, in the encoding a multi-limb body needs.

    Named subsumes indexed (``move_joints_blocking(dict(zip(names, q)))`` is
    ``move_joint_blocking(q)``), so this is deliberately NOT a merge with ``JointDriver``: a
    single contract holding both encodings would leave a bench arm — which has no named-joint
    interface — failing its own composite, the same "forces NotImplementedError stubs" reason
    this module splits by capability at all. A body implements the encoding its hardware
    speaks; cruzr happens to speak both and implements both slices.

    Two things the indexed form cannot express, which is why this slice exists:

    * **A partial command.** A dict commands a subset and holds the rest; a list must state
      every joint. "Raise the right shoulder, hold everything else" is only sayable as a dict.
    * **A body with more than one kinematic chain.** A list needs an agreed index order. For
      one arm that is the chain; for two arms plus a waist and a lifter, index 7 means nothing.
    """

    def get_joint_positions(self) -> dict[str, float]:
        """Return the latest known joint positions keyed by joint name."""

    def move_joints_blocking(
        self,
        targets: dict[str, float],
        *,
        timeout_s: float = 30.0,
    ) -> Any:
        """Move the NAMED joints in ``targets`` to their absolute positions, holding the rest.

        Blocks until reached or ``timeout_s`` elapses. Joints absent from ``targets`` are held,
        which is what makes this the form a multi-limb body can use.
        """


@runtime_checkable
class ServoDriver(Protocol):
    """Optional non-blocking streaming-motion surface for the real-time servo loop.

    Unlike ``move_to_pose_blocking`` (which polls to completion), ``servo_to_pose``
    fires a FLANGE-frame pose command and returns immediately, so a ``control_hz``
    loop can stream small slew-limited steps toward a moving target. ``pose`` is a
    mapping with ``x/y/z`` (mm) and optional ``rx/ry/rz``/``r`` (deg).
    """

    def servo_to_pose(self, pose: Any) -> bool | None:
        """Issue a non-blocking FLANGE-frame pose command.

        ``False`` explicitly means the low-level controller did not advance its
        plan (for example, a rate-gate skip or tracking catch-up hold). ``True``
        or legacy ``None`` means the command was accepted.
        """


@runtime_checkable
class BaseDriver(Protocol):
    """Planar mobile-base motion — ``motion.base`` / ``motion.goal``.

    Both verbs are blocking and return ``{ok, reason, ...}``. Distances are
    METRES here (detections are millimetres) — the framework's convention.
    """

    def navigate_relative(self, dx_m: float, dy_m: float = 0.0, dyaw_rad: float = 0.0) -> dict:
        """Turn by ``dyaw_rad`` then translate (dx forward, dy left), REP-103."""

    def navigate_arc(self, radius_m: float, dyaw_rad: float) -> dict:
        """Drive ONE constant-curvature arc, turning while advancing."""


@runtime_checkable
class ContinuousBaseDriver(Protocol):
    """Non-blocking streaming base motion — ``motion.base_servo``.

    The base keeps rolling while the caller senses, so a moving target can be
    steered toward mid-drive. Pair every ``start`` with a ``stop``: an abandoned
    handle leaves the wheels turning.
    """

    def start_base_drive(self, **kwargs: Any) -> Any:
        """Start a forward drive and return an opaque handle."""

    def base_drive_running(self, handle: Any) -> bool:
        """Whether the drive behind ``handle`` is still moving."""

    def steer_base_drive(self, handle: Any, bearing_rad: float) -> None:
        """Aim a running drive at ``bearing_rad`` (+ = left of heading)."""

    def hold_base_drive(self, handle: Any) -> None:
        """Pause the wheels without ending the drive (target lost → do not creep blind)."""

    def stop_base_drive(self, handle: Any) -> dict:
        """Stop the drive and reap its result. Idempotent."""


@runtime_checkable
class LifterDriver(Protocol):
    """Vertical torso/lifter position control — ``motion.lift``."""

    def set_lifter(self, q_lifter: dict[str, float]) -> Any:
        """Command the lifter joints to absolute positions (rad per joint name)."""


@runtime_checkable
class WaistDriver(Protocol):
    """Torso yaw rotation — ``motion.waist``."""

    def turn_waist(self, delta_rad: float) -> Any:
        """Rotate the torso waist by ``delta_rad`` (+ = left)."""


@runtime_checkable
class DualArmDriver(Protocol):
    """Two arms driven in coordination — ``motion.dual_arm``.

    A TOPOLOGY slice, not an end-effector one: what the arms carry (paddles, grippers, a
    hand) is the separate ``grasp.*`` axis, and only that part has no cross-vendor default.
    The coordination around it does — two-arm IK, the ready/descend/contact sequence and
    the force confirmation are the same whatever is on the end — so ``dual_arm_grasp`` /
    ``dual_arm_place`` reach a shared implementation and take the contact plan from a body
    hook. Only ``home`` is contractual here.
    """

    def home(self) -> None:
        """Return both arms to their home configuration (blocking)."""


@runtime_checkable
class CameraDriver(Protocol):
    """Optional camera surface — typically delegates to ``_common.RealSenseCamera``."""

    @property
    def intrinsics(self) -> np.ndarray | None:
        """3x3 camera intrinsics ``K``; ``None`` until the camera has started."""

    def grab_frames(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Grab one aligned ``(rgb_uint8, depth_m_float32)`` pair, or ``None`` if unavailable."""


@runtime_checkable
class SuctionDriver(Protocol):
    """Optional suction-gripper IO surface."""

    @property
    def suction_state(self) -> bool:
        """Last commanded suction state (True = on)."""

    @property
    def suction_di_last(self) -> int | None:
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


class HandGuidingRecoveryError(RuntimeError):
    """A hand-guiding context could not return the robot to a controllable state.

    Distinct from an ordinary failure: the arm may still be unpowered and needs
    a human hand on it before anything else is attempted.
    """


@runtime_checkable
class HandGuidingDriver(Protocol):
    """Optional hand-guiding surface — release torque so a human can pose the robot."""

    def hand_guiding(self, *, include_end_effector: bool = False) -> AbstractContextManager[None]:
        """Release torque on entry; restore a controllable state on exit.

        ``include_end_effector=False`` (default) releases the arm only and leaves
        the end effector powered — hand-eye calibration teaching relies on this to
        keep the board clamped. ``True`` releases the end effector as well, so
        whatever it holds will drop. Drivers whose end effector cannot be released
        (e.g. suction) treat the flag as a no-op.

        Exit must resynchronise the motion targets with where the human actually
        left the robot before re-energising, otherwise the servos snap back to the
        pre-release goal. Raise :class:`HandGuidingRecoveryError` when that fails.
        """


@runtime_checkable
class VisionDriver(Protocol):
    """Optional hand-eye calibration surface for eye-in-hand back-projection."""

    @property
    def tf_flange_cam(self) -> np.ndarray | None:
        """4x4 flange→camera extrinsic transform, or None if uncalibrated."""

    @property
    def calibration(self) -> dict | None:
        """Loaded hand-eye calibration payload, or None."""


# ---------------------------------------------------------------------------
# Composite driver types for adapters whose driver implements multiple protocols.
# A multi-protocol ``Protocol`` subclass gives true static type checking (mypy /
# pyright verify every member) plus a ``runtime_checkable`` ``isinstance`` probe
# for capability gating — replacing the former type alias which only documented
# the expected surface without enforcing it.
# ---------------------------------------------------------------------------


@runtime_checkable
class PiperFullDriver(CartesianDriver, JointDriver, CameraDriver, GripperDriver, VisionDriver, Protocol):
    """Composite driver surface — union of all five vendor protocols.

    ``PiperLowLevel`` implements all five; ``PiperApi._ll()`` returns this
    type so vision reads (``tf_flange_cam`` / ``calibration`` / ``intrinsics``
    / ``grab_frames``) plus motion / gripper / camera reads are statically
    verified by mypy / pyright. ``isinstance(driver, PiperFullDriver)`` gives
    a runtime capability check for adapters that want to gate on the full
    surface.
    """

    pass
