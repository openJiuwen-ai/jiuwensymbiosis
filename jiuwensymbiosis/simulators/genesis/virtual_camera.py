"""Independent calibrated camera; poses are virtual, never robot measurements."""

import numpy as np


def camera():
    if __package__:
        from .robot_cameras import RobotCamera
    else:
        from robot_cameras import RobotCamera
    return RobotCamera(
        "scan_cam",
        "virtual_rig",
        (640, 480),
        np.array([[389.88092041015625, 0, 332.7503967285156], [0, 389.9339599609375, 252.76022338867188], [0, 0, 1]]),
        np.zeros(5),
        np.eye(4),
        (0.03, 50.0),
    )


def optical_pose(pose):
    position = np.asarray(pose["pos"], dtype=float)
    forward = np.asarray(pose["lookat"], dtype=float) - position
    up = np.asarray(pose["up"], dtype=float)
    if not np.isfinite(np.r_[position, forward, up]).all() or np.linalg.norm(forward) < 1e-9:
        raise ValueError("Invalid virtual camera pose")
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-9:
        raise ValueError("Degenerate virtual camera up direction")
    right /= np.linalg.norm(right)
    corrected_up = np.cross(right, forward)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack((right, -corrected_up, forward))
    matrix[:3, 3] = position
    return {
        "pos": position.tolist(),
        "lookat": (position + forward).tolist(),
        "up": corrected_up.tolist(),
        "T_world_camera": matrix.tolist(),
    }
