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
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    # Type-only: keeps the core ``env`` package free of an ``adapters`` runtime import.
    from jiuwensymbiosis.adapters._common.protocol import RobotDriver

logger = logging.getLogger(__name__)

# Closed vocabulary for capability strings.
# Adding a new capability:
#   1. Append it here.
#   2. (Optionally) add a Mixin in api/mixins.py that declares it.
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
        "sorting.command",  # opaque sorting protocol (no Cartesian motion)
        "speech.tts",  # text-to-speech available
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
    _home_pose: Any = None
    _tool_offset_mm: float = 0.0

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

    # --- motion / end-effector verbs (default: delegate to low_level) ---

    def _require_driver(self) -> RobotDriver:
        """Return ``low_level`` or raise if the env is not connected."""
        ll = self.low_level
        if ll is None:
            raise RuntimeError(f"{self.name}: env not connected (no low_level driver).")
        return ll

    def home(self) -> None:
        """Move to the home pose (blocking)."""
        self._require_driver().home()

    def get_flange_pose(self) -> Any:
        """Return the current flange-frame pose (vendor Pose object)."""
        return self._require_driver().get_pose()

    def move_to_flange(self, pose: Any) -> None:
        """Move to a FLANGE-frame target pose (blocking)."""
        self._require_driver().move_to_pose_blocking(pose)

    def move_joint(self, q: list[float]) -> None:
        """Move to a joint-space configuration (blocking)."""
        # JointDriver sibling protocol; motion.joint-capability-gated
        self._require_driver().move_joint_blocking(q)  # type: ignore[attr-defined]

    def servo_to_flange(self, pose: Any) -> None:
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
        servo(pose)

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

    # --- sensor convenience ---

    def grab_rgb(self) -> np.ndarray | None:
        """Single-frame RGB grab for vision tools.

        Default delegates to ``get_observation().rgb``; override in adapters
        that can fetch RGB more cheaply than a full observation snapshot.
        """
        return self.get_observation().rgb
