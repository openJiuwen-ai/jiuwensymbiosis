# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Hardware-protocol abstraction (Layer 2).

`BaseRobotEnv` is the minimal contract every robot body must satisfy.
The framework relies on:
- ``capabilities``: a closed-vocabulary set advertising what the env supports;
  the rails and tool builder gate themselves by these strings.
- ``connect``/``disconnect``: lifecycle. Idempotent.
- ``get_observation``: returns a ``RobotObservation`` with whatever fields
  the env can populate; downstream code checks for ``None`` rather than
  asking ``hasattr``.

Hardware emergency stop must remain a hardware-layer concern; the rails
in this framework do *not* take over physical safety.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from jiuwensymbiosis.env.protocol import CartesianDriver, RobotDriver

logger = logging.getLogger(__name__)

# Closed vocabulary for capability strings.
# Adding a new capability:
#   1. Append it here.
#   2. (Optionally) add a Mixin in api/components.py that declares it.
#   3. (Optionally) write a Rail that activates only when this string is present.
KNOWN_CAPABILITIES: frozenset[str] = frozenset(
    {
        "motion.cartesian",  # XYZ(R) end-effector commands in base frame
        "motion.joint",  # joint-space commands
        "motion.servo",  # non-blocking streaming pose commands (real-time servo loop)
        "grasp.suction",  # suction on/off
        "grasp.parallel",  # parallel gripper open/close
        "vision.camera",  # raw image stream available
        "vision.depth",  # depth stream available
        "vision.detection",  # high-level object detection
        "vision.eye_to_hand",  # camera is fixed in the robot base/world frame
        # The body can turn/move whatever carries a camera (head, waist or the base
        # itself), so it can look around for a target instead of only seeing what
        # happens to be in front of it. Says nothing about the camera: RGB or RGBD,
        # wide or narrow — searching only ever reports a BEARING.
        "vision.search",
        "sorting.command",  # opaque sorting protocol (no Cartesian motion)
        "speech.tts",  # text-to-speech available
        "motion.base",  # planar mobile-base relative motion (differential; no strafe)
        "motion.base_servo",  # non-blocking streaming base drive (steer-while-moving)
        "motion.lift",  # vertical torso/lifter position control
        "motion.waist",  # torso yaw (waist) rotation
        "motion.goal",  # autonomous drive to a goal/grasp-band via a nav stack
        "grasp.dual_arm",  # coordinated two-arm (paddle) grasp/release
        "planning.reachability",  # URDF-based reachability / workspace prior for planning
    }
)


@dataclass
class RobotObservation:
    """Snapshot of robot+env state at one instant.

    All fields are optional — the env populates whatever it can. Consumers
    check for None.

    Attributes:
        pose: Cartesian pose dict, schema is robot-specific but conventional
            keys are {"x","y","z","r"} for SCARA and {"x","y","z","rx","ry",
            "rz"} (Euler, deg) for 6-DOF.
        joints: Joint angles in rad or deg (per-robot convention).
        rgb: HxWx3 uint8 image, base-of-robot camera or wrist camera.
        depth: HxW float32 depth in meters, aligned to ``rgb`` if both present.
        extra: Anything else (gripper width, force/torque, status flags).
    """

    pose: dict | None = None
    joints: list[float] | None = None
    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


class BaseRobotEnv(ABC):
    """Robot hardware protocol — minimal common surface."""

    capabilities: frozenset[str] = frozenset()
    name: str = "robot"

    # --- class hooks ---

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate subclass capabilities against KNOWN_CAPABILITIES."""
        super().__init_subclass__(**kwargs)
        unknown = set(cls.capabilities) - KNOWN_CAPABILITIES
        if unknown:
            raise ValueError(
                f"{cls.__name__} declares unknown capabilities: {sorted(unknown)}. "
                f"Add them to KNOWN_CAPABILITIES in jiuwensymbiosis/env/base.py first."
            )

    # --- context manager ---

    def __enter__(self):
        """Enter context: connect the env."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        """Exit context: disconnect the env."""
        try:
            self.disconnect()
        except Exception as e:
            if exc_type is not None:
                logger.debug("disconnect() failed during exception unwind: %s", e)
            else:
                logger.warning("disconnect() failed: %s", e)

    # --- lifecycle ---

    @abstractmethod
    def connect(self) -> None:
        """Open hardware connection. Must be idempotent."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release hardware. Must be idempotent and safe at any state."""

    @abstractmethod
    def get_observation(self) -> RobotObservation:
        """Best-effort snapshot. Should not raise on transient sensor gaps."""

    def reset(self) -> None:
        """Optional: bring the robot back to a safe pose. Default: no-op."""
        return None

    def emergency_stop(self) -> None:
        """Optional software-level halt. Default: no-op. Hardware E-stop must
        be wired physically — do not rely on this.
        """
        return None

    # --- helpers ---

    def has(self, capability: str) -> bool:
        """Check whether the env supports a given capability string."""
        return capability in self.capabilities

    # --- optional hardware contract (default None; adapters set or override) ---
    # Assign in connect() (e.g. ``self.low_level = XxxDriver()``) or override as a
    # read-only @property (re-declare the @property *and* a setter that raises
    # AttributeError — mypy forbids a read-only property overriding a read-write
    # one). ``z_min_safe`` / ``workspace_bounds`` are the safety envelope
    # SafetyRail reads. ``home_pose`` / ``tool_offset_mm`` are robot body constants
    # the api layer needs for coordinate math.
    #
    # ``low_level`` is a **controlled penetration point**: motion / end-effector /
    # safety-boundary access MUST go through Env methods (``home()``,
    # ``get_flange_pose()`` etc). Vision calibration data (``tf_flange_cam``,
    # ``calibration``, ``intrinsics``, ``grab_frames``) and vendor-specific
    # operations may access ``low_level`` directly — but the access is type-
    # constrained by the ``RobotDriver`` (and sibling) Protocol(s).
    _low_level: RobotDriver | None = None
    _z_min_safe: float | None = None
    _workspace_bounds: tuple[float, float, float, float] | None = None
    # Joint soft limits: {J1: (low, high), ...}. Unit MUST match the env's
    # ``move_joint(q)`` convention (deg for Piper; rad for a ROS-style adapter);
    # key order = q index order. ``None`` → SafetyRail skips the range check
    # (only q-presence / type / finite checks run).
    _joint_limits: dict[str, tuple[float, float]] | None = None
    # Mobile-base / torso envelope, same ``None`` → "SafetyRail skips the range check"
    # contract as ``joint_limits``. Relative verbs (``navigate_relative`` / ``rotate_base`` /
    # ``drive_arc`` / ``turn_waist``) get a PER-COMMAND cap — there is no absolute frame to
    # bound them in; the absolute one (``set_lifter``) gets a real range.
    # ``_base_step_limits`` = (max |translation| per command in m, max |turn| in rad).
    _base_step_limits: tuple[float, float] | None = None
    # {lifter_joint: (low, high)}; unit MUST match the env's ``set_lifter`` convention.
    _lift_limits: dict[str, tuple[float, float]] | None = None
    _waist_step_limit_rad: float | None = None
    _home_pose: Any = None
    _tool_offset_mm: float = 0.0
    # URDF model + arm kinematic chains for planning-time reachability (capability
    # ``planning.reachability``). ``None`` → the body has no URDF model → the reachability
    # mixin degrades to a no-op (piper and other URDF-less bodies are unaffected).
    _urdf_path: str | None = None
    _arm_chains: dict[str, tuple[str, str]] | None = None

    @property
    def low_level(self) -> RobotDriver | None:
        return self._low_level

    @low_level.setter
    def low_level(self, value: RobotDriver | None) -> None:
        self._low_level = value

    @property
    def z_min_safe(self) -> float | None:
        return self._z_min_safe

    @z_min_safe.setter
    def z_min_safe(self, value: float | None) -> None:
        self._z_min_safe = value

    @property
    def workspace_bounds(self) -> tuple[float, float, float, float] | None:
        return self._workspace_bounds

    @workspace_bounds.setter
    def workspace_bounds(self, value: tuple[float, float, float, float] | None) -> None:
        self._workspace_bounds = value

    @property
    def joint_limits(self) -> dict[str, tuple[float, float]] | None:
        return self._joint_limits

    @joint_limits.setter
    def joint_limits(self, value: dict[str, tuple[float, float]] | None) -> None:
        self._joint_limits = value

    @property
    def base_step_limits(self) -> tuple[float, float] | None:
        """Per-command mobile-base cap: (max |translation| m, max |turn| rad). None = no cap."""
        return self._base_step_limits

    @base_step_limits.setter
    def base_step_limits(self, value: tuple[float, float] | None) -> None:
        self._base_step_limits = value

    @property
    def lift_limits(self) -> dict[str, tuple[float, float]] | None:
        """Lifter joint soft limits, ``set_lifter``'s unit and key names. None = no range check."""
        return self._lift_limits

    @lift_limits.setter
    def lift_limits(self, value: dict[str, tuple[float, float]] | None) -> None:
        self._lift_limits = value

    @property
    def waist_step_limit_rad(self) -> float | None:
        """Max |delta_rad| a single ``turn_waist`` command may request. None = no cap."""
        return self._waist_step_limit_rad

    @waist_step_limit_rad.setter
    def waist_step_limit_rad(self, value: float | None) -> None:
        self._waist_step_limit_rad = value

    # Robot body constants. Adapters override as @property or set in connect().
    @property
    def home_pose(self) -> Any:
        return self._home_pose

    @home_pose.setter
    def home_pose(self, value: Any) -> None:
        self._home_pose = value

    @property
    def tool_offset_mm(self) -> float:
        return self._tool_offset_mm

    @tool_offset_mm.setter
    def tool_offset_mm(self, value: float) -> None:
        self._tool_offset_mm = value

    # URDF-based reachability contract (planning.reachability). Adapters that ship a URDF set these
    # (in connect() or as a read-only @property); the base default None means "no URDF → skip".
    @property
    def urdf_path(self) -> str | None:
        return self._urdf_path

    @urdf_path.setter
    def urdf_path(self, value: str | None) -> None:
        self._urdf_path = value

    @property
    def arm_chains(self) -> dict[str, tuple[str, str]] | None:
        """Arm kinematic chains for reachability: name → (root_link, leaf_link). None = no URDF model."""
        return self._arm_chains

    @arm_chains.setter
    def arm_chains(self, value: dict[str, tuple[str, str]] | None) -> None:
        self._arm_chains = value

    # --- motion / end-effector verbs (default: delegate to low_level) ---

    def _require_driver(self) -> RobotDriver:
        """Return ``low_level`` or raise if the env is not connected."""
        ll = self.low_level
        if ll is None:
            raise RuntimeError(f"{self.name}: env not connected (no low_level driver).")
        return ll

    def _require_cartesian(self) -> CartesianDriver:
        """Return the driver typed as its Cartesian surface (``motion.cartesian``-gated)."""
        return cast(CartesianDriver, self._require_driver())

    def home(self) -> None:
        """Move to the home pose (blocking)."""
        self._require_cartesian().home()

    def get_flange_pose(self) -> Any:
        """Return the current flange-frame pose (vendor Pose object)."""
        return self._require_cartesian().get_pose()

    def move_to_flange(self, pose: Any) -> None:
        """Move to a FLANGE-frame target pose (blocking)."""
        self._require_cartesian().move_to_pose_blocking(pose)

    def move_joint(self, q: list[float]) -> None:
        """Move to a joint-space configuration (blocking)."""
        # JointDriver sibling protocol; motion.joint-capability-gated
        self._require_driver().move_joint_blocking(q)  # type: ignore[attr-defined]

    def servo_to_flange(self, pose: Any) -> bool | None:
        """Issue a NON-BLOCKING FLANGE-frame pose command (returns immediately).

        This is the streaming-motion primitive the real-time servo loop drives
        at ``control_hz``: it commands a small step toward a target and returns
        without waiting for the arm to settle (unlike ``move_to_flange``, which
        polls to completion). ``pose`` is a mapping with keys ``x/y/z`` (mm) and
        optional ``rx/ry/rz`` or ``r`` (deg), base frame.

        Default delegates to the driver's ``servo_to_pose`` when present; envs
        that declare ``motion.servo`` must provide it (override or driver). Raises
        otherwise.
        """
        driver = self._require_driver()
        servo = getattr(driver, "servo_to_pose", None)
        if servo is None:
            raise NotImplementedError(f"{self.name}: driver has no servo_to_pose (declare/implement 'motion.servo').")
        return cast(bool | None, servo(pose))

    def set_end_effector(self, engaged: bool) -> None:
        """Engage (True) / release (False) the end effector.

        Dispatches to the driver's ``set_gripper`` or ``set_suction`` based on
        the env's declared capabilities (``grasp.parallel`` vs ``grasp.suction``).
        """
        driver = self._require_driver()
        if "grasp.parallel" in self.capabilities:
            # GripperDriver sibling protocol; grasp.parallel-capability-gated
            driver.set_gripper(engaged)  # type: ignore[attr-defined]
        elif "grasp.suction" in self.capabilities:
            # SuctionDriver sibling protocol; grasp.suction-capability-gated
            driver.set_suction(engaged)  # type: ignore[attr-defined]
        else:
            raise NotImplementedError(
                f"{self.name}: no grasp capability declared (need 'grasp.parallel' or 'grasp.suction')"
            )

    # --- mobile-base / torso verbs (default: raise; implement when the capability is declared) ---

    def navigate_relative(self, dx_m: float, dy_m: float = 0.0, dyaw_rad: float = 0.0) -> dict:
        """Turn by ``dyaw_rad``, then translate ``dx_m`` forward / ``dy_m`` left (REP-103).

        Blocking; returns ``{ok, reason, ...}``. Metres here, millimetres in detections —
        the framework's convention. Envs declaring ``motion.base`` implement it.
        """
        raise NotImplementedError(f"{self.name}: navigate_relative not implemented (declare/implement 'motion.base').")

    def navigate_arc(self, radius_m: float, dyaw_rad: float) -> dict:
        """Drive ONE constant-curvature arc: ``radius_m`` radius, signed ``dyaw_rad`` (+ = left).

        Turning while advancing, so the base lands off its original heading line — the
        primitive an approach planner needs to reach a target's face normal. Blocking;
        returns ``{ok, reason, ...}``. Envs declaring ``motion.base`` may implement it.
        """
        raise NotImplementedError(f"{self.name}: navigate_arc not implemented (declare/implement 'motion.base').")

    def start_base_drive(self, **kwargs: Any) -> Any:
        """Start a NON-BLOCKING forward drive and return an opaque handle.

        The streaming counterpart of ``navigate_relative``: the wheels keep rolling while
        the caller senses, so a moving target can be steered toward mid-drive (there is no
        stop-look-go dead time). Required by ``motion.base_servo``; pair every start with
        ``stop_base_drive`` — an abandoned handle leaves the base rolling.
        """
        raise NotImplementedError(
            f"{self.name}: start_base_drive not implemented (declare/implement 'motion.base_servo')."
        )

    def base_drive_running(self, handle: Any) -> bool:
        """Whether the drive behind ``handle`` is still moving (False once it self-stopped)."""
        raise NotImplementedError(
            f"{self.name}: base_drive_running not implemented (declare/implement 'motion.base_servo')."
        )

    def steer_base_drive(self, handle: Any, bearing_rad: float) -> None:
        """Aim a running drive at ``bearing_rad`` (+ = left of the body's heading). Best-effort."""
        raise NotImplementedError(
            f"{self.name}: steer_base_drive not implemented (declare/implement 'motion.base_servo')."
        )

    def hold_base_drive(self, handle: Any) -> None:
        """Pause the wheels of a running drive WITHOUT ending it (target lost → do not creep blind).

        The next ``steer_base_drive`` resumes it, so a perception dropout costs latency, not a
        restart. Best-effort.
        """
        raise NotImplementedError(
            f"{self.name}: hold_base_drive not implemented (declare/implement 'motion.base_servo')."
        )

    def stop_base_drive(self, handle: Any) -> dict:
        """Stop the drive behind ``handle`` and reap its result ``{ok, reason, ...}``. Idempotent."""
        raise NotImplementedError(
            f"{self.name}: stop_base_drive not implemented (declare/implement 'motion.base_servo')."
        )

    def set_lifter(self, q_lifter: dict[str, float]) -> Any:
        """Command the torso lifter joints to absolute positions (rad, keyed by joint name).

        Envs declaring ``motion.lift`` implement it.
        """
        raise NotImplementedError(f"{self.name}: set_lifter not implemented (declare/implement 'motion.lift').")

    def turn_waist(self, delta_rad: float) -> Any:
        """Rotate the torso waist by ``delta_rad`` (+ = left).

        A waist yaw leaves the base frame fixed, so base-frame detections stay valid.
        Envs declaring ``motion.waist`` implement it.
        """
        raise NotImplementedError(f"{self.name}: turn_waist not implemented (declare/implement 'motion.waist').")

    # --- sensor convenience ---

    def grab_rgb(self) -> np.ndarray | None:
        """Single-frame RGB grab for vision tools.

        Default delegates to ``get_observation().rgb``; override in adapters
        that can fetch RGB more cheaply than a full observation snapshot.
        """
        return self.get_observation().rgb

    @property
    def cameras(self) -> tuple[str | None, ...]:
        """Cameras this body can perceive with, best-first. Default: the one unnamed camera.

        Perception looks through ALL of them and lets the answers decide what to do next:
        a frame that carries depth yields a face normal (so the base can square up to the
        target), a frame without one yields only a bearing (so the base can close in until
        something with depth acquires it). That is a property of the FRAME, readable from
        ``CameraFrame.depth_m``, not a class of hardware — so this list deliberately says
        nothing about what kind each camera is, and a body with a single RGBD camera is a
        perfectly ordinary case rather than a missing sensor.
        """
        return (None,)

    def grab_calibrated_frame(self, camera: str | None = None) -> Any:
        """Grab rgb + depth + intrinsics + base←camera extrinsics as one ``CameraFrame``, or None.

        3-D perception needs all four from the SAME instant — a pixel projected with a stale
        extrinsic lands somewhere the object never was — which is why this is one verb rather
        than four getters. ``camera`` names one of ``cameras`` (None = default).
        Envs declaring ``vision.depth`` implement it.
        """
        raise NotImplementedError(
            f"{self.name}: grab_calibrated_frame not implemented (declare/implement 'vision.depth')."
        )
