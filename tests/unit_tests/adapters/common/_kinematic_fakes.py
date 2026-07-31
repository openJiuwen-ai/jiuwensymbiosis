# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Generic fakes for the joint-level motion core — no vendor/SDK anywhere.

``GantryBackend`` is an analytic stand-in for a URDF FK/IK solver: its three
joints ARE the XYZ position (mm), orientation is identity, and ``max_reach_mm``
bounds IK so far targets leave a residual (exercising the reject path).
``FakeJointTransport`` is an in-memory joint SDK that either follows commands
perfectly or stays stuck (for the stall path).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


class GantryBackend:
    """Analytic 3-axis backend: joints are XYZ in mm, identity orientation."""

    def __init__(self, max_reach_mm: float = 1.0e9) -> None:
        self.max_reach_mm = float(max_reach_mm)

    def forward_kinematics(self, q: list[float]) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, 3] = np.array([q[0], q[1], q[2]], dtype=np.float64) / 1000.0
        return matrix

    def inverse_kinematics(self, current: list[float], target: np.ndarray, *, orientation_weight: float) -> list[float]:
        desired = np.asarray(target, dtype=np.float64)[:3, 3] * 1000.0
        norm = float(np.linalg.norm(desired))
        if norm > self.max_reach_mm and norm > 0.0:
            desired = desired * (self.max_reach_mm / norm)
        return [float(desired[0]), float(desired[1]), float(desired[2])]


class FakeJointTransport:
    """In-memory joint-level SDK seam for driver tests.

    Pass ``frames`` to also act as a camera transport (the driver forwards the
    VisionDriver surface); leave it None for a motion-only arm.
    """

    def __init__(
        self,
        initial: list[float],
        *,
        effector: float = 0.0,
        follows: bool = True,
        fail_precheck: bool = False,
        frames: object | None = None,
        clip_fn: Callable[[list[float], list[float]], list[float]] | None = None,
        drift_fn: Callable[[list[float], list[float]], list[float]] | None = None,
        steady_offset: list[float] | None = None,
        effector_clip_fn: Callable[[float, float], float] | None = None,
    ) -> None:
        self._joints = [float(v) for v in initial]
        self._effector = float(effector)
        self._follows = follows
        self._fail_precheck = fail_precheck
        # clip_fn(requested, current) -> accepted: simulates an SDK that clamps
        # per-command deltas (e.g. LeRobot max_relative_target). drift_fn(current,
        # requested) -> new: a gravity-loaded servo moving the wrong way (error
        # grows each resend). steady_offset: the arm parks at command + offset
        # after each send (PD steady-state droop, no firmware I term).
        self._clip_fn = clip_fn
        self._drift_fn = drift_fn
        self._steady_offset = [float(v) for v in steady_offset] if steady_offset is not None else None
        self._effector_clip_fn = effector_clip_fn
        self.opened = False
        self.close_count = 0
        self.sent_arm: list[list[float]] = []
        self.sent_effector: list[float] = []
        if frames is not None:
            self.frames = frames  # presence enables the driver's grab_frames forwarding

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.close_count += 1
        self.opened = False

    def precheck(self) -> None:
        if self._fail_precheck:
            raise RuntimeError("fake precheck failed")

    def read_arm_joints(self) -> list[float]:
        return list(self._joints)

    def send_arm_joints(self, q: list[float]) -> list[float]:
        requested = [float(v) for v in q]
        self.sent_arm.append(list(requested))  # records the COMMANDED value
        if self._drift_fn is not None:
            self._joints = [float(v) for v in self._drift_fn(list(self._joints), requested)]
            return list(self._joints)
        if not self._follows:
            return list(self._joints)
        accepted = requested
        if self._clip_fn is not None:
            accepted = [float(v) for v in self._clip_fn(requested, list(self._joints))]
        if self._steady_offset is not None:
            self._joints = [a + off for a, off in zip(accepted, self._steady_offset, strict=True)]
        else:
            self._joints = list(accepted)
        return list(self._joints)

    def read_effector(self) -> float:
        return self._effector

    def send_effector(self, pos: float) -> float:
        self.sent_effector.append(float(pos))  # records the COMMANDED value
        accepted = float(pos)
        if self._effector_clip_fn is not None:
            accepted = float(self._effector_clip_fn(float(pos), self._effector))
        if self._follows:
            self._effector = accepted
        return self._effector

    def grab_frames(self):
        return getattr(self, "frames", None)


def advancing_clock(dt: float = 1.0):
    """A monotonic clock that advances ``dt`` seconds on every call."""
    state = {"t": 0.0}

    def clock() -> float:
        state["t"] += dt
        return state["t"]

    return clock
