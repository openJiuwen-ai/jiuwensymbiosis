# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""SE(3) + pinhole geometry for Piper + wrist-mounted RealSense.

6-DoF eye-in-hand:

    tf_base_cam = tf_base_flange(GetArmEndPose) @ tf_flange_cam

The arm reports the full flange pose (``GetArmEndPoseMsgs``), so we never do FK
ourselves; ``tf_flange_cam`` is the calibration constant.

RPY axis order:
  ``_RPY_AXES`` is the scipy ``Rotation.from_euler`` axes string. Piper's
  ``EndPose`` RX/RY/RZ are Euler degrees but the docs don't pin the order;
  default ``"xyz"`` (intrinsic XYZ) is the common industrial convention.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from jiuwensymbiosis.adapters._common.geometry import (
    apply_transform,
    make_transform,
    pixel_and_depth_to_camera_xyz,
)

__all__ = [
    "FlangePose",
    "rpy_deg_to_rot",
    "pixel_and_depth_to_base_xyz",
]

_RPY_AXES = "xyz"


def rpy_deg_to_rot(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """RPY (degrees) → 3x3 rotation matrix using ``_RPY_AXES``."""
    return Rotation.from_euler(_RPY_AXES, [rx_deg, ry_deg, rz_deg], degrees=True).as_matrix()


@dataclass(frozen=True, slots=True)
class FlangePose:
    x_mm: float
    y_mm: float
    z_mm: float
    rx_deg: float
    ry_deg: float
    rz_deg: float

    def to_tf_base_flange(self) -> np.ndarray:
        return make_transform(
            rpy_deg_to_rot(self.rx_deg, self.ry_deg, self.rz_deg),
            np.array([self.x_mm, self.y_mm, self.z_mm], dtype=np.float64),
        )

    def as_tuple(self):
        return astuple(self)


def pixel_and_depth_to_base_xyz(
    uv: tuple[float, float],
    depth_m: float,
    flange_pose: FlangePose,
    tf_flange_cam: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Project (pixel + metric depth) → base-frame XYZ (mm) via eye-in-hand.

    Args:
      uv: pixel coords (u, v) in the color image.
      depth_m: aligned metric depth at (u, v).
      flange_pose: flange-frame pose in the base frame.
      tf_flange_cam: 4x4 calibration constant (camera pose in flange frame).
      intrinsics: 3x3 camera intrinsics.
    """
    p_cam_mm = pixel_and_depth_to_camera_xyz(uv, depth_m, intrinsics)
    tf_base_cam = flange_pose.to_tf_base_flange() @ tf_flange_cam
    return apply_transform(tf_base_cam, p_cam_mm)
