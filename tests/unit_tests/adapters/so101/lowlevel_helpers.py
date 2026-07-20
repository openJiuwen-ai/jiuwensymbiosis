# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Test doubles for the SO-101 driver that do NOT import LeRobot.

These fakes exercise the driver's control flow (interpolation, settle loop,
set_gripper, connect sequence, reachability rejection) deterministically.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from jiuwensymbiosis.adapters.so101.geometry import (
    matrix_m_to_pose_mm_deg,
    pose_mm_deg_to_matrix_m,
)


class FakeKinematics:
    """Stand-in for ``lerobot.model.kinematics.RobotKinematics``.

    FK/IK operate purely on the :mod:`geometry` converters so tests need no
    LeRobot install. FK maps joint config to a pose whose translation is the
    joint vector scaled; IK inverts by solving a trivial target.
    """

    def __init__(
        self, urdf_path: str, target_frame_name: str = "gripper_frame_link", joint_names: list[str] | None = None
    ) -> None:
        self.urdf_path = urdf_path
        self.target_frame_name = target_frame_name
        self.joint_names = joint_names
        self.fk_calls = 0
        self.ik_calls = 0

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        self.fk_calls += 1
        q = np.asarray(joint_pos_deg, dtype=float).ravel()
        if q.size != 5:
            raise ValueError(f"FakeKinematics expects 5 joints, got {q.size}.")
        # Translation in mm = joint values * 10 (mm), so tests can predict it.
        pose = np.zeros(6)
        pose[0] = q[0] * 10.0
        pose[1] = q[1] * 10.0
        pose[2] = q[2] * 10.0
        return pose_mm_deg_to_matrix_m(_pose_from_array(pose))

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ) -> np.ndarray:
        self.ik_calls += 1
        # Trivial IK: recover the joint values that would produce the desired pose
        # under our FK convention (mm = joint * 10). current_joint_pos seeds it.
        target_pose = matrix_m_to_pose_mm_deg(np.asarray(desired_ee_pose, dtype=float))
        return np.array(
            [target_pose.x / 10.0, target_pose.y / 10.0, target_pose.z / 10.0, 0.0, 0.0],
            dtype=float,
        )


def _pose_from_array(arr: np.ndarray):
    from jiuwensymbiosis.adapters.so101.geometry import So101Pose

    return So101Pose(
        x=float(arr[0]),
        y=float(arr[1]),
        z=float(arr[2]),
        rx=float(arr[3]),
        ry=float(arr[4]),
        rz=float(arr[5]),
    )


class FakeFollower:
    """Stand-in for ``lerobot...SOFollower``.

    Records sent actions and serves a configurable observation. The arm
    position can be set to ``track`` the last sent target so the settle loop
    converges in tests.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.calibration_fpath = "/tmp/fake_so101_calib.json"
        self.is_calibrated = True
        self.action_features = {
            "shoulder_pan.pos": None,
            "shoulder_lift.pos": None,
            "elbow_flex.pos": None,
            "wrist_flex.pos": None,
            "wrist_roll.pos": None,
            "gripper.pos": None,
        }
        self.sent_actions: list[dict[str, float]] = []
        self.connected = False
        self._arm: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
        self._gripper: float = 50.0
        self.track = True  # if True, get_observation reflects the last sent target
        # Optional clip hook: maps the requested action to the *actual* action
        # returned to the driver (LeRobot may clip via max_relative_target).
        # When None, send_action echoes the requested action unchanged.
        self.clip_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None
        # Optional drift hook: maps (arm_after_track, target_arm) -> new_arm,
        # applied after tracking the sent target. Simulates a gravity-loaded
        # servo that drifts away from the target (e.g. elbow_flex under load).
        # When None, no drift (ideal tracking). Only fires when track=True.
        self.drift_fn: Callable[[list[float], list[float]], list[float]] | None = None
        # Optional steady-state offset hook: a 5-float list added to the tracked
        # arm AFTER tracking the sent target, simulating a PD servo's steady-state
        # error (constant joint offset under gravity load, e.g. elbow_flex ~2.7 deg
        # — the servo settles at command+offset instead of command). None = ideal
        # tracking (no offset). Mutually exclusive with drift_fn in practice
        # (drift_fn simulates divergence, steady_offset a settled constant bias).
        # Only fires when track=True and drift_fn is None.
        self.steady_offset: list[float] | None = None

    def connect(self, calibrate: bool = True) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        actual = self.clip_fn(dict(action)) if self.clip_fn is not None else dict(action)
        self.sent_actions.append(actual)
        if self.track:
            # Reflect the *actual* arm targets so the settle loop sees convergence
            # toward the clipped target (matching real hardware behavior).
            arm_names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
            prev_arm: list[float] = list(self._arm)  # state before this action
            target_arm: list[float] = list(self._arm)
            for i, name in enumerate(arm_names):
                key = f"{name}.pos"
                if key in actual:
                    self._arm[i] = float(actual[key])
                    target_arm[i] = float(actual[key])
            if "gripper.pos" in actual:
                self._gripper = float(actual["gripper.pos"])
            # Optional drift: a gravity-loaded servo drifting away from the
            # commanded target. The hook receives (prev_arm, target_arm) so a
            # drift_fn can accumulate from the real prior position (e.g. elbow
            # creeping toward larger angles under gravity), not just offset the
            # freshly-commandered target — that is what makes the settle error grow.
            if self.drift_fn is not None:
                self._arm = list(self.drift_fn(prev_arm, target_arm))
            elif self.steady_offset is not None:
                # Settled constant offset: servo parks at command+offset, not at
                # command — the STS3215 PD steady-state error the convergence
                # loop compensates. Applied to the freshly-tracked target so the
                # settle loop converges (err settles to |offset| < tolerance) yet
                # FK(arm) != commanded target, exposing the residual.
                self._arm = [a + o for a, o in zip(target_arm, self.steady_offset, strict=True)]
        return actual

    def get_observation(self) -> dict[str, Any]:
        obs: dict[str, Any] = {}
        arm_names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
        for i, name in enumerate(arm_names):
            obs[f"{name}.pos"] = self._arm[i]
        obs["gripper.pos"] = self._gripper
        return obs


def make_calib_file(tmp_path) -> str:
    """Create a fake calibration file so the driver's preload check passes."""
    p = tmp_path / "fake_so101_calib.json"
    p.write_text("{}")
    return str(p)


class _FakeSOFollower:
    """Fallback for the ``or SOFollower`` slot in ``connect()``.

    Never instantiated when ``so_follower_factory`` is injected (which the
    tests always do); kept only so the 4-tuple shape matches the real
    ``_import_lerobot`` return.
    """

    def __init__(self, robot_cfg: Any) -> None:  # pragma: no cover - never used
        self.robot_cfg = robot_cfg


class FakeRobotConfig:
    """Stand-in for ``lerobot...SOFollowerRobotConfig``.

    Accepts the keyword args ``connect()`` passes (port/id/calibration_dir/...)
    so the fake import path builds a robot_cfg without touching lerobot.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def fake_lerobot_import() -> tuple[Any, Any, Any, str]:
    """Return the 4-tuple ``connect()`` consumes without importing lerobot.

    Lets ``test_lowlevel.py`` exercise the connect/motion/settle control flow
    in an environment without the optional ``so101`` extra. The kinematics
    slot reuses :class:`FakeKinematics`; the SOFollower slot is a placeholder
    that is short-circuited by the injected ``so_follower_factory``.
    """
    return _FakeSOFollower, FakeRobotConfig, FakeKinematics, "0.6.0"
