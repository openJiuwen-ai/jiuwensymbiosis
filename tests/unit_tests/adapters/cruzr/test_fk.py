# coding: utf-8
import numpy as np
import pytest

from jiuwensymbiosis.kinematics.fk import (
    axis_angle_matrix,
    fk_chain,
    rpy_to_matrix,
)
from jiuwensymbiosis.kinematics.urdf_chain import Chain, Joint


def test_rpy_yaw_90deg():
    R = rpy_to_matrix(0.0, 0.0, np.pi / 2)
    assert np.allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-9)


def test_axis_angle_z_90deg():
    R = axis_angle_matrix((0.0, 0.0, 1.0), np.pi / 2)
    assert np.allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-9)


def test_fk_single_translation_then_rotation():
    # one fixed translate +1 in x, then a revolute z joint at +90deg
    chain = Chain(joints=[
        Joint("fix", "fixed", (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.0, 0.0),
        Joint("j", "revolute", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), -3.14, 3.14),
    ])
    T = fk_chain(chain, {"j": np.pi / 2})
    assert np.allclose(T[:3, 3], [1.0, 0.0, 0.0], atol=1e-9)
    # j's frame x-axis now points +y in root
    assert np.allclose(T[:3, :3] @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-9)


def test_fk_missing_joint_defaults_zero():
    chain = Chain(joints=[
        Joint("j", "revolute", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), -3.14, 3.14),
    ])
    T = fk_chain(chain, {})  # no angle given → 0
    assert np.allclose(T, np.eye(4), atol=1e-9)
