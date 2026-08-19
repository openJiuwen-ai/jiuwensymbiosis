# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Pose / unit / SE(3) conversions for locally-solved (FK/IK) arms.

The tool layer speaks **millimetres + Euler degrees**; kinematics backends
(URDF FK/IK) speak **metres + rotation matrices**. This module is the single
place that bridges the two, so no business code needs a stray ``/1000`` or
``deg2rad``. Pure math — knows nothing about any specific vendor or SDK.

``euler_axes`` is the scipy ``Rotation.from_euler`` axes string; it is a
per-body convention supplied by the adapter (default intrinsic ``"xyz"``).
"""

from __future__ import annotations

from dataclasses import astuple, dataclass
from typing import Any, NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation

from jiuwensymbiosis.utils.geometry import make_transform

_MM_PER_M = 1000.0


@dataclass(frozen=True, slots=True)
class CartesianPose:
    """End-effector pose in the tool convention: mm + Euler degrees.

    Field names (``x/y/z/rx/ry/rz``) match what ``api/components.py:_pose_to_dict``
    reads, so a driver may return this object straight from ``get_pose()``.
    """

    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        return astuple(self)


class PoseXYZRPY(NamedTuple):
    """Positional pose read: mm translation + Euler-degree rotation."""

    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float


def read_pose_xyzrpy(obj: Any) -> PoseXYZRPY:
    """Read ``(x, y, z, rx, ry, rz)`` from any pose-like object.

    Tolerates the ``SimpleNamespace`` the motion mixins build and objects that
    expose ``r`` instead of ``rz``. Missing rotation fields default to 0.
    """
    x = float(obj.x)
    y = float(obj.y)
    z = float(obj.z)
    rx = float(getattr(obj, "rx", 0.0))
    ry = float(getattr(obj, "ry", 0.0))
    rz = float(getattr(obj, "rz", getattr(obj, "r", 0.0)))
    return PoseXYZRPY(x, y, z, rx, ry, rz)


def pose_to_matrix(pose: Any, *, euler_axes: str = "xyz") -> np.ndarray:
    """Tool-frame pose (mm/deg) → 4x4 SE(3) with translation in **metres**."""
    x, y, z, rx, ry, rz = read_pose_xyzrpy(pose)
    rot = Rotation.from_euler(euler_axes, [rx, ry, rz], degrees=True).as_matrix()
    t_m = np.array([x, y, z], dtype=np.float64) / _MM_PER_M
    return make_transform(np.asarray(rot, dtype=np.float64), t_m)


def matrix_to_pose(matrix: np.ndarray, *, euler_axes: str = "xyz") -> CartesianPose:
    """4x4 SE(3) (translation in metres) → tool-frame ``CartesianPose`` (mm/deg)."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"expected a 4x4 SE(3) matrix, got shape {m.shape}")
    rx, ry, rz = Rotation.from_matrix(m[:3, :3]).as_euler(euler_axes, degrees=True)
    t_mm = m[:3, 3] * _MM_PER_M
    return CartesianPose(
        x=float(t_mm[0]),
        y=float(t_mm[1]),
        z=float(t_mm[2]),
        rx=float(rx),
        ry=float(ry),
        rz=float(rz),
    )


def position_error_mm(actual: np.ndarray, target: np.ndarray) -> float:
    """Euclidean translation error (mm) between two metre-scale SE(3) matrices."""
    delta_m = np.asarray(actual, dtype=np.float64)[:3, 3] - np.asarray(target, dtype=np.float64)[:3, 3]
    return float(np.linalg.norm(delta_m) * _MM_PER_M)


def orientation_error_deg(actual: np.ndarray, target: np.ndarray) -> float:
    """Geodesic rotation error (deg) between two SE(3) matrices' rotation parts."""
    r_rel = np.asarray(actual, dtype=np.float64)[:3, :3].T @ np.asarray(target, dtype=np.float64)[:3, :3]
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(r_rel).as_rotvec())))
