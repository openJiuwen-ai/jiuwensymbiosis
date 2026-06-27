# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Cross-vendor SE(3) and pinhole-projection primitives.

Pure math. Knows nothing about any specific robot's kinematics or any
specific camera model beyond the pinhole intrinsics. Per-vendor geometry
lives in the per-vendor ``geometry.py`` and composes these.

Frame conventions used by the helpers:
  cam  — RealSense color-stream optical frame (CV convention):
           x_cam = image right (u +)
           y_cam = image down  (v +)
           z_cam = optical axis (into the scene)
  Any other frame is fully described by a 4x4 homogeneous transform.
"""

from __future__ import annotations

import math

import numpy as np


def make_transform(rot: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a 3x3 rotation and a 3-vector translation."""
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = t
    return transform


def apply_transform(transform: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply a 4x4 SE(3) transform to a (3,) point or (N,3) array of points."""
    p = np.asarray(p)
    if p.ndim == 1:
        result: np.ndarray = transform[:3, :3] @ p + transform[:3, 3]
        return result
    result = p @ transform[:3, :3].T + transform[:3, 3]
    return result


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Closed-form inverse of an SE(3) transform: (R, t) → (Rᵀ, -Rᵀ t)."""
    rot = transform[:3, :3]
    t = transform[:3, 3]
    transform_inv = np.eye(4)
    transform_inv[:3, :3] = rot.T
    transform_inv[:3, 3] = -rot.T @ t
    return transform_inv


def _rot_z(deg: float) -> np.ndarray:
    """Rotation about base Z by ``deg`` degrees."""
    c = math.cos(math.radians(deg))
    s = math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def pixel_and_depth_to_camera_xyz(uv: tuple[float, float], depth_m: float, intrinsics: np.ndarray) -> np.ndarray:
    """Back-project a single pixel + metric depth to camera-frame XYZ (in mm).

    Note: ``depth_m`` is in meters; output is in mm to match some robot's base-frame
    convention (so callers can compose with mm-valued SE(3) transforms without
    a unit jump).
    """
    u, v = uv
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    ppx, ppy = intrinsics[0, 2], intrinsics[1, 2]
    z_mm = float(depth_m) * 1000.0
    x_mm = (u - ppx) * z_mm / fx
    y_mm = (v - ppy) * z_mm / fy
    return np.array([x_mm, y_mm, z_mm], dtype=np.float64)
