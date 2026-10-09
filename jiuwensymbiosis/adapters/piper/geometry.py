# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""SE(3) + pinhole geometry for Piper + wrist-mounted RealSense.

6-DoF eye-in-hand:

    tf_base_cam = tf_base_flange(GetArmEndPose) @ tf_flange_cam

The arm reports the full flange pose (``GetArmEndPoseMsgs``), so we never do FK
ourselves; ``tf_flange_cam`` is the calibration constant.

RPY axis order:
  ``_RPY_AXES = "xyz"`` uses extrinsic rotations, ``R = Rz @ Ry @ Rx``.
  This agrees with the matrix-to-RPY formulas in the vendor SDK's
  ``piper_sdk/kinematics/piper_fk.py``. Hardware acceptance must also verify
  the installed tool axis and firmware; see docs/en/reference/piper-coordinates.md.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import astuple, dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from jiuwensymbiosis.utils.geometry import (
    apply_transform,
    make_transform,
    matrix_mm_to_xyzrpy,
    pixel_and_depth_to_camera_xyz,
)

__all__ = [
    "FlangePose",
    "normalize_flange_pose",
    "resolve_yaw",
    "rpy_deg_to_rot",
    "tip_offset_in_base",
    "flange_pose_to_tip",
    "matrix_mm_to_flange_pose",
    "pixel_and_depth_to_base_xyz",
]

_RPY_AXES = "xyz"


def rpy_deg_to_rot(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """RPY (degrees) → 3x3 rotation matrix using ``_RPY_AXES``."""
    return np.asarray(Rotation.from_euler(_RPY_AXES, [rx_deg, ry_deg, rz_deg], degrees=True).as_matrix())


def tip_offset_in_base(rx: float, ry: float, rz: float, length_mm: float) -> np.ndarray:
    """Flange-to-tip displacement: rotate the local +Z extension into the base frame."""
    return rpy_deg_to_rot(rx, ry, rz)[:, 2] * length_mm


def flange_pose_to_tip(pose: Any, length_mm: float) -> dict[str, float]:
    """Convert one vendor x/y/z/rx/ry/rz flange sample to public TIP coordinates."""
    offset = tip_offset_in_base(pose.rx, pose.ry, pose.rz, length_mm)
    return {
        "x": float(pose.x + offset[0]),
        "y": float(pose.y + offset[1]),
        "z": float(pose.z + offset[2]),
        "rx": float(pose.rx),
        "ry": float(pose.ry),
        "rz": float(pose.rz),
    }


def matrix_mm_to_flange_pose(matrix: np.ndarray) -> FlangePose:
    """Convert a 4x4 SE(3) with **millimetre** translation to a :class:`FlangePose`.

    Lets the calibration workflow drive a flange SE(3) into Piper's command type
    without a vendor-specific pose builder. Rotation is read from the leading
    3x3 and emitted as ``_RPY_AXES`` Euler degrees; translation passes through
    unchanged (mm).
    """
    return FlangePose(*matrix_mm_to_xyzrpy(matrix, axes=_RPY_AXES))


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


def resolve_yaw(pose: Mapping[str, Any]) -> Any:
    """Select non-None rz, otherwise r; numeric conversion/readback belong to the caller."""
    rz = pose.get("rz")
    return pose.get("r") if rz is None else rz


def normalize_flange_pose(pose: Any, *, read_pose: Callable[[], Any] | None = None) -> FlangePose:
    """Normalize a native, mapping, or attribute pose to a complete FLANGE target.

    XYZ is always required. Orientation is also required unless the caller
    explicitly supplies ``read_pose``; then only missing rx/ry/rz fields are
    filled from one vendor x/y/z/rx/ry/rz sample. Complete targets never read
    hardware. ``r`` aliases yaw, with a supplied ``rz`` taking precedence.
    Geometry and workspace validation remain the driver's responsibility.
    """
    if isinstance(pose, FlangePose):
        return pose
    fields = ("x", "y", "z", "rx", "ry", "rz")
    if isinstance(pose, Mapping):
        values = {key: pose.get(key) for key in (*fields, "r")}
    else:
        absent = object()
        values = {key: getattr(pose, key, absent) for key in (*fields, "r")}
        if all(value is absent for value in values.values()):
            raise TypeError(
                f"Piper flange pose has unsupported type {type(pose).__name__}; "
                "expected FlangePose, a mapping, or an x/y/z attribute object."
            )
        values = {key: None if value is absent else value for key, value in values.items()}
    values["rz"] = resolve_yaw(values)
    required = fields if read_pose is None else fields[:3]
    missing = [key for key in required if values[key] is None]
    if missing:
        raise TypeError(f"Piper flange pose missing required fields: {', '.join(missing)}.")
    target: dict[str, float] = {}
    for key in fields:
        value = values[key]
        if value is not None:
            try:
                target[key] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Piper flange pose field {key!r} must be numeric; got {value!r}.") from exc
    missing_orientation = [key for key in fields[3:] if key not in target]
    if missing_orientation and read_pose is not None:
        current = read_pose()  # read failure propagates before any command is sent
        for key in missing_orientation:
            target[key] = float(getattr(current, key))
    return FlangePose(*(target[key] for key in fields))


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
