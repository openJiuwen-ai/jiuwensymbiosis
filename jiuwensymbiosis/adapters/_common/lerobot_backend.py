# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""``LerobotKinematicsBackend`` — an ``FkIkBackend`` over LeRobot ``RobotKinematics``.

LeRobot arms (SO-100 / SO-101 / ...) ship a placo URDF solver as
``lerobot.model.kinematics.RobotKinematics``. This wraps it behind the
framework's ``FkIkBackend`` seam (see ``_common.kinematics``) so every
locally-solved LeRobot arm reuses one Cartesian layer instead of re-wrapping the
solver. SE(3) matrices are metres, joints are degrees (RobotKinematics' own
convention).

LeRobot is imported lazily — only when no solver class is injected — so
importing this module (and any adapter that composes it) stays cheap and free of
the optional hardware SDK. An adapter that already imports LeRobot for its
transport passes the ``RobotKinematics`` class in via ``robot_kinematics_cls``;
tests pass a fake; ``None`` lazily imports the real class.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["LerobotKinematicsBackend"]


class LerobotKinematicsBackend:
    """``FkIkBackend`` over ``RobotKinematics(urdf, target_frame_name, joint_names)``.

    ``forward_kinematics(q)`` maps a joint vector (degrees) to a 4x4 SE(3) pose
    (translation in metres); ``inverse_kinematics(current, target, *,
    orientation_weight)`` maps a desired SE(3) pose to a joint vector seeded by
    ``current``. Both coerce to/from float ndarrays so a caller may pass lists or
    arrays interchangeably.

    The solver is built eagerly in ``__init__`` (a bad URDF / target frame raises
    here, at connect time). ``robot_kinematics_cls`` injects the solver class: an
    adapter that already imported LeRobot passes it in, tests pass a fake, and
    ``None`` lazily imports ``lerobot.model.kinematics.RobotKinematics``.
    """

    def __init__(
        self,
        urdf_path: str,
        *,
        target_frame_name: str,
        joint_names: list[str],
        robot_kinematics_cls: Callable[..., Any] | None = None,
    ) -> None:
        self.target_frame_name = str(target_frame_name)
        self.joint_names = list(joint_names)
        if robot_kinematics_cls is None:
            from lerobot.model.kinematics import RobotKinematics

            robot_kinematics_cls = RobotKinematics
        self._solver = robot_kinematics_cls(
            urdf_path,
            target_frame_name=target_frame_name,
            joint_names=list(joint_names),
        )

    def forward_kinematics(self, q: list[float]) -> np.ndarray:
        return np.asarray(self._solver.forward_kinematics(np.asarray(q, dtype=float)), dtype=float)

    def inverse_kinematics(
        self, current: list[float], target: np.ndarray, *, orientation_weight: float, position_weight: float = 1.0
    ) -> list[float]:
        solution = self._solver.inverse_kinematics(
            np.asarray(current, dtype=float),
            np.asarray(target, dtype=float),
            position_weight=position_weight,
            orientation_weight=orientation_weight,
        )
        return [float(v) for v in np.asarray(solution, dtype=float).ravel()]
