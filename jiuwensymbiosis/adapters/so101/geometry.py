# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Pure coordinate conversions for the SO-101 adapter.

All public angles are **Euler angles in degrees** (intrinsic XYZ, scipy
``Rotation.from_euler("xyz", ...)``), matching the framework convention used by
the motion mixins and the Piper adapter; all public translations are in
**millimetres**. The internal SE(3) matrix uses **metres** for translation and a
3x3 rotation matrix, matching :class:`lerobot.model.kinematics.RobotKinematics`,
whose ``forward_kinematics`` returns a 4x4 and ``inverse_kinematics`` consumes a
4x4.

We deliberately do NOT expose ``ee.*`` (``ee.x``/``ee.wx``/...) conversions:
those belong to LeRobot's kinematic processor intermediate format, while
``SOFollower.send_action()`` accepts ``{"shoulder_pan.pos": ...}`` motor
targets. This adapter calls ``RobotKinematics`` FK/IK directly and sends joint
actions, so ``ee.*`` has no place on the command path.

Rotation handling uses :class:`scipy.spatial.transform.Rotation` via the
intrinsic-XYZ Euler representation (degrees). The axis order ``"xyz"`` matches
``jiuwensymbiosis.adapters.piper.geometry._RPY_AXES`` so the two adapters share
one Euler convention; multi-axis ``rx/ry/rz`` are interpreted consistently
across the framework.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = [
    "So101Pose",
    "pose_mm_deg_to_matrix_m",
    "matrix_m_to_pose_mm_deg",
    "position_error_mm",
    "orientation_error_deg",
]

_MM_PER_M = 1000.0
# Intrinsic XYZ Euler, matching the Piper adapter's _RPY_AXES so rx/ry/rz mean
# the same thing across adapters.
_EULER_AXES = "xyz"


@dataclass(frozen=True)
class So101Pose:
    """SO-101 pose: translation in mm, rotation as intrinsic-XYZ Euler degrees.

    ``rx``/``ry``/``rz`` are intrinsic XYZ Euler angles in degrees (scipy
    ``Rotation.from_euler("xyz", [rx, ry, rz], degrees=True)``), matching the
    framework-wide convention (see the motion mixins and the Piper adapter).
    This is NOT a rotation vector — multi-axis values compose as Euler, not as
    axis-angle. The name follows the project's ``rx/ry/rz`` convention.
    """

    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float


def _require_finite_pose(pose: So101Pose) -> None:
    for name, val in (
        ("x", pose.x),
        ("y", pose.y),
        ("z", pose.z),
        ("rx", pose.rx),
        ("ry", pose.ry),
        ("rz", pose.rz),
    ):
        if not np.isfinite(float(val)):
            raise ValueError(f"So101Pose.{name} must be finite, got {val!r}.")


def pose_mm_deg_to_matrix_m(pose: So101Pose) -> np.ndarray:
    """Convert a :class:`So101Pose` (mm / XYZ-Euler deg) to a 4x4 SE(3) (m / matrix).

    Translation is divided by 1000 (mm -> m). Rotation is built from the
    intrinsic-XYZ Euler angles in degrees. Returns a float64 4x4 matrix.
    """
    if not isinstance(pose, So101Pose):
        raise TypeError(f"pose must be a So101Pose, got {type(pose).__name__}.")
    _require_finite_pose(pose)

    rotation = Rotation.from_euler(_EULER_AXES, [float(pose.rx), float(pose.ry), float(pose.rz)], degrees=True)

    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = np.array([pose.x, pose.y, pose.z], dtype=float) / _MM_PER_M
    return matrix


def matrix_m_to_pose_mm_deg(matrix: np.ndarray) -> So101Pose:
    """Convert a 4x4 SE(3) (m / matrix) to a :class:`So101Pose` (mm / XYZ-Euler deg).

    Inverse of :func:`pose_mm_deg_to_matrix_m`. Translation is multiplied by
    1000 (m -> mm). Rotation is read from the leading 3x3 and emitted as
    intrinsic-XYZ Euler angles in degrees (each in ``[-180, 180]``).
    """
    arr = np.asarray(matrix, dtype=float)
    if arr.shape != (4, 4):
        raise ValueError(f"matrix must be a 4x4 array, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"matrix has non-finite value: {matrix!r}.")

    rotation = Rotation.from_matrix(arr[:3, :3])
    rx_deg, ry_deg, rz_deg = rotation.as_euler(_EULER_AXES, degrees=True).tolist()
    tx_mm, ty_mm, tz_mm = (arr[:3, 3] * _MM_PER_M).tolist()
    return So101Pose(
        x=tx_mm,
        y=ty_mm,
        z=tz_mm,
        rx=rx_deg,
        ry=ry_deg,
        rz=rz_deg,
    )


def position_error_mm(actual: So101Pose, target: So101Pose) -> float:
    """Euclidean translation error between two poses, in mm."""
    if not isinstance(actual, So101Pose) or not isinstance(target, So101Pose):
        raise TypeError("position_error_mm requires two So101Pose instances.")
    for name, val in (
        ("actual.x", actual.x),
        ("actual.y", actual.y),
        ("actual.z", actual.z),
        ("target.x", target.x),
        ("target.y", target.y),
        ("target.z", target.z),
    ):
        if not np.isfinite(float(val)):
            raise ValueError(f"position_error_mm: {name} must be finite, got {val!r}.")
    dx = float(actual.x) - float(target.x)
    dy = float(actual.y) - float(target.y)
    dz = float(actual.z) - float(target.z)
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def orientation_error_deg(actual: So101Pose, target: So101Pose) -> float:
    """Rotation error between two poses, in degrees (geodesic distance).

    Returns the angle of the relative rotation ``R_actual^-1 @ R_target``,
    i.e. the shortest angular distance between the two orientations. Zero
    means identical orientations. Result is in ``[0, 180]`` degrees.

    Computed on the underlying rotation matrices, so it is independent of the
    Euler/rotvec choice used to build each pose — only the final orientation
    matters.
    """
    if not isinstance(actual, So101Pose) or not isinstance(target, So101Pose):
        raise TypeError("orientation_error_deg requires two So101Pose instances.")
    for name, val in (
        ("actual.rx", actual.rx),
        ("actual.ry", actual.ry),
        ("actual.rz", actual.rz),
        ("target.rx", target.rx),
        ("target.ry", target.ry),
        ("target.rz", target.rz),
    ):
        if not np.isfinite(float(val)):
            raise ValueError(f"orientation_error_deg: {name} must be finite, got {val!r}.")

    r_actual = Rotation.from_euler(_EULER_AXES, [actual.rx, actual.ry, actual.rz], degrees=True)
    r_target = Rotation.from_euler(_EULER_AXES, [target.rx, target.ry, target.rz], degrees=True)
    relative = r_actual.inv() * r_target
    # The rotvec magnitude of the relative rotation is the geodesic angle (deg).
    angle_deg = float(np.linalg.norm(relative.as_rotvec(degrees=True)))
    return float(min(angle_deg, 180.0))
