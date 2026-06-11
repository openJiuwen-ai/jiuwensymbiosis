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
from typing import Any, Optional

import numpy as np

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
            keys are {"x","y","z","r"} for SCARA and {"x","y","z","qx","qy",
            "qz","qw"} for 6-DOF.
        joints: Joint angles in rad or deg (per-robot convention).
        rgb: HxWx3 uint8 image, base-of-robot camera or wrist camera.
        depth: HxW float32 depth in meters, aligned to ``rgb`` if both present.
        extra: Anything else (gripper width, force/torque, status flags).
    """

    pose: Optional[dict] = None
    joints: Optional[list[float]] = None
    rgb: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
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
