# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Tests for LerobotKinematicsBackend — the FkIkBackend over an injected RobotKinematics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from jiuwensymbiosis.adapters._common.kinematics import FkIkBackend
from jiuwensymbiosis.adapters._common.lerobot_backend import LerobotKinematicsBackend


class _FakeSolver:
    """Stand-in for lerobot RobotKinematics: joints[:3] are the XYZ metres."""

    def __init__(self, urdf_path: str, target_frame_name: str = "tcp", joint_names: list[str] | None = None) -> None:
        self.urdf_path = urdf_path
        self.target_frame_name = target_frame_name
        self.joint_names = joint_names
        self.fk_inputs: list[np.ndarray] = []
        self.ik_orientation_weights: list[float] = []
        self.ik_position_weights: list[float] = []

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        self.fk_inputs.append(np.asarray(q))
        matrix = np.eye(4)
        matrix[:3, 3] = np.asarray(q, dtype=float)[:3]
        return matrix

    def inverse_kinematics(
        self, current: np.ndarray, target: np.ndarray, position_weight: float = 1.0, orientation_weight: float = 0.01
    ) -> np.ndarray:
        self.ik_orientation_weights.append(orientation_weight)
        self.ik_position_weights.append(position_weight)
        return np.asarray(target, dtype=float)[:3, 3]


def _backend(cls: Any = _FakeSolver) -> LerobotKinematicsBackend:
    return LerobotKinematicsBackend(
        "some.urdf", target_frame_name="tcp", joint_names=["a", "b", "c"], robot_kinematics_cls=cls
    )


class TestLerobotKinematicsBackend:
    def test_satisfies_fkikbackend_protocol(self):
        assert isinstance(_backend(), FkIkBackend)

    def test_construction_passes_urdf_frame_and_joints_to_solver(self):
        backend = _backend()
        assert backend.target_frame_name == "tcp"
        assert backend._solver.urdf_path == "some.urdf"
        assert backend._solver.target_frame_name == "tcp"
        assert backend._solver.joint_names == ["a", "b", "c"]

    def test_forward_kinematics_returns_float_ndarray(self):
        backend = _backend()
        matrix = backend.forward_kinematics([1.0, 2.0, 3.0])
        assert isinstance(matrix, np.ndarray)
        assert matrix.dtype == np.float64
        assert np.allclose(matrix[:3, 3], [1.0, 2.0, 3.0])

    def test_forward_kinematics_coerces_list_to_array_for_solver(self):
        backend = _backend()
        backend.forward_kinematics([4.0, 5.0, 6.0])
        assert isinstance(backend._solver.fk_inputs[-1], np.ndarray)

    def test_inverse_kinematics_returns_list_of_floats_and_passes_weight(self):
        backend = _backend()
        target = np.eye(4)
        target[:3, 3] = [7.0, 8.0, 9.0]
        solution = backend.inverse_kinematics([0.0, 0.0, 0.0], target, orientation_weight=0.25)
        assert isinstance(solution, list)
        assert all(isinstance(v, float) for v in solution)
        assert solution == pytest.approx([7.0, 8.0, 9.0])
        assert backend._solver.ik_orientation_weights == [0.25]

    def test_inverse_kinematics_forwards_position_weight(self):
        backend = _backend()
        target = np.eye(4)
        backend.inverse_kinematics([0.0, 0.0, 0.0], target, orientation_weight=0.1)
        backend.inverse_kinematics([0.0, 0.0, 0.0], target, orientation_weight=0.1, position_weight=0.5)
        assert backend._solver.ik_position_weights == [1.0, 0.5]

    def test_inverse_kinematics_ravels_solver_output(self):
        class ColumnSolver(_FakeSolver):
            def inverse_kinematics(self, current, target, position_weight=1.0, orientation_weight=0.01):
                return np.asarray(target, dtype=float)[:3, 3].reshape(3, 1)

        backend = _backend(ColumnSolver)
        target = np.eye(4)
        target[:3, 3] = [1.0, 2.0, 3.0]
        solution = backend.inverse_kinematics([0.0, 0.0, 0.0], target, orientation_weight=0.01)
        assert solution == pytest.approx([1.0, 2.0, 3.0])

    def test_bad_solver_construction_propagates(self):
        class BadSolver:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("bad target frame")

        with pytest.raises(RuntimeError, match="bad target frame"):
            _backend(BadSolver)
