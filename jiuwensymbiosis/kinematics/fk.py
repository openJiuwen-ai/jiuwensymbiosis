# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""前向运动学：URDF 链关节角 → root→leaf 4x4 变换（米）。

每关节 = T_origin(xyz, rpy) · R_axis(θ)。固定关节 θ=0。通用支持任意 axis（Rodrigues）。
"""

from __future__ import annotations

import numpy as np

from jiuwensymbiosis.kinematics.urdf_chain import Chain


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis XYZ (R = Rz·Ry·Rx) → 3x3 rotation."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def axis_angle_matrix(axis: tuple[float, float, float], theta: float) -> np.ndarray:
    """Rodrigues rotation about ``axis`` by ``theta`` (rad)."""
    a = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    a = a / n
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def _origin(xyz: tuple, rpy: tuple) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rpy_to_matrix(*rpy)
    t[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return t


def fk_chain(chain: Chain, q: dict[str, float]) -> np.ndarray:
    """Compose root→leaf transform for joint angles ``q`` (missing → 0)."""
    t = np.eye(4)
    for j in chain.joints:
        t = t @ _origin(j.xyz, j.rpy)
        if j.jtype in ("revolute", "continuous"):
            theta = float(q.get(j.name, 0.0))
            rot = np.eye(4)
            rot[:3, :3] = axis_angle_matrix(j.axis, theta)
            t = t @ rot
    return t
