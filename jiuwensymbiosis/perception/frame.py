# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Generic camera-frame container + pixel→base projection (robot-agnostic).

``CameraFrame`` holds one grabbed frame (rgb + optional depth/intrinsics/extrinsics).
``project_to_base`` back-projects a pixel (u,v)+depth to camera XYZ then maps it to
the base frame — pure pinhole + SE(3) over ``utils.geometry``. Any adapter's camera
driver returns a ``CameraFrame`` and reuses ``project_to_base``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jiuwensymbiosis.utils.geometry import apply_transform, pixel_and_depth_to_camera_xyz


def project_to_base(
    uv: tuple[float, float],
    depth_m: float,
    intrinsics: np.ndarray,
    tf_base_cam: np.ndarray,
) -> np.ndarray:
    """Back-project pixel (u,v)+depth to camera XYZ (mm), then map to base via ``tf_base_cam``."""
    xyz_cam = pixel_and_depth_to_camera_xyz(uv, depth_m, np.asarray(intrinsics, dtype=np.float64))
    return apply_transform(np.asarray(tf_base_cam, dtype=np.float64), xyz_cam)


@dataclass
class CameraFrame:
    """One grabbed frame. ``depth_m`` / ``intrinsics`` / ``tf_base_cam`` may be None when unavailable."""

    rgb: np.ndarray
    depth_m: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    tf_base_cam: np.ndarray | None = None
    frame_id: str | None = None
    captured_monotonic_s: float | None = None
