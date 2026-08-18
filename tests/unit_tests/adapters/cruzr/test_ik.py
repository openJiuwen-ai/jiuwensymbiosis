# coding: utf-8
from pathlib import Path

import numpy as np
import pytest

from jiuwensymbiosis.kinematics.fk import fk_chain
from jiuwensymbiosis.kinematics.ik import (
    ik_solve_5dof,
    ik_solve_pose,
    tool_normal_base,
)
from jiuwensymbiosis.kinematics.urdf_chain import parse_chain

URDF = "/home/riemann/Robot/Cruzr_ws/cruzr_s2_description/cruzr_s2_description/urdf/cruzr_s2_v1/cruzr_s2_v1.urdf"
ARM = [
    "L_shoulder_pitch_joint", "L_shoulder_roll_joint", "L_shoulder_yaw_joint",
    "L_elbow_roll_joint", "L_elbow_yaw_joint", "L_wrist_pitch_joint", "L_wrist_roll_joint",
]


@pytest.mark.skipif(not Path(URDF).exists(), reason="urdf not present")
def test_ik_recovers_a_reachable_target():
    chain = parse_chain(URDF, "base_link", "L_sixforce_link")
    q_fixed = {"lifter_pitch_1_joint": 0.0, "lifter_pitch_2_joint": 0.0,
               "lifter_pitch_3_joint": 0.0, "waist_yaw_joint": 0.0}
    # pick a known arm config, forward it, then ask IK to recover that pose
    q_known = dict(q_fixed)
    q_known.update({k: v for k, v in zip(ARM, [0.3, -0.4, 0.2, -0.6, 0.1, 0.3, 0.0])})
    T = fk_chain(chain, q_known)
    target_pos = T[:3, 3]
    target_normal = tool_normal_base(chain, q_known, (0.0, 0.0, 1.0))

    res = ik_solve_5dof(chain, q_fixed, ARM, target_pos, target_normal,
                        tool_normal_local=(0.0, 0.0, 1.0))
    assert res.converged
    assert res.pos_err_m < 0.005
    assert res.normal_err < 0.02


@pytest.mark.skipif(not Path(URDF).exists(), reason="urdf not present")
def test_ik_unreachable_returns_not_converged():
    chain = parse_chain(URDF, "base_link", "L_sixforce_link")
    q_fixed = {"lifter_pitch_1_joint": 0.0, "lifter_pitch_2_joint": 0.0,
               "lifter_pitch_3_joint": 0.0, "waist_yaw_joint": 0.0}
    res = ik_solve_5dof(chain, q_fixed, ARM, np.array([5.0, 0.0, 0.0]),
                        np.array([1.0, 0.0, 0.0]), max_iters=100)
    assert not res.converged


@pytest.mark.skipif(not Path(URDF).exists(), reason="urdf not present")
def test_ik_respects_joint_limits():
    chain = parse_chain(URDF, "base_link", "L_sixforce_link")
    limits = chain.limits()
    q_fixed = {"lifter_pitch_1_joint": 0.0, "lifter_pitch_2_joint": 0.0,
               "lifter_pitch_3_joint": 0.0, "waist_yaw_joint": 0.0}
    q_known = dict(q_fixed)
    q_known.update({k: v for k, v in zip(ARM, [0.3, -0.4, 0.2, -0.6, 0.1, 0.3, 0.0])})
    T = fk_chain(chain, q_known)
    res = ik_solve_5dof(chain, q_fixed, ARM, T[:3, 3],
                        tool_normal_base(chain, q_known, (0.0, 0.0, 1.0)),
                        tool_normal_local=(0.0, 0.0, 1.0))
    for name in ARM:
        lo, hi = limits[name]
        assert lo - 1e-6 <= res.q[name] <= hi + 1e-6


@pytest.mark.skipif(not Path(URDF).exists(), reason="urdf not present")
def test_ik_pose_two_axis_with_tcp_roundtrip():
    """ik_solve_pose recovers a known pose: TCP position + 2 oriented axes."""
    chain = parse_chain(URDF, "base_link", "L_sixforce_link")
    q_fixed = {"lifter_pitch_1_joint": 0.0, "lifter_pitch_2_joint": 0.0,
               "lifter_pitch_3_joint": 0.0, "waist_yaw_joint": 0.0}
    q_known = dict(q_fixed)
    q_known.update({k: v for k, v in zip(ARM, [0.4, -0.3, 0.2, -0.5, 0.1, 0.4, 0.3])})
    tcp_local = (0.0, 0.0, 0.09)  # paddle TCP 9 cm along tool-z for this synthetic test
    paddle_local = (1.0, 0.0, 0.0)
    approach_local = (0.0, 0.0, 1.0)

    T = fk_chain(chain, q_known)
    R = T[:3, :3]
    tcp_pos = T[:3, 3] + R @ np.asarray(tcp_local)
    approach_target = R @ np.asarray(approach_local)
    paddle_target = R @ np.asarray(paddle_local)

    res = ik_solve_pose(
        chain, q_fixed, ARM, tcp_pos,
        approach_target=approach_target, paddle_target=paddle_target,
        tool_approach_local=approach_local, tool_paddle_local=paddle_local,
        tcp_offset_local=tcp_local,
    )
    assert res.converged
    assert res.pos_err_m < 0.012
    # achieved orientation axes match the targets
    T2 = fk_chain(chain, {**q_fixed, **res.q})
    assert np.allclose(T2[:3, :3] @ np.asarray(approach_local), approach_target, atol=0.06)
    assert np.allclose(T2[:3, :3] @ np.asarray(paddle_local), paddle_target, atol=0.06)


@pytest.mark.skipif(not Path(URDF).exists(), reason="urdf not present")
def test_ik_pose_tcp_offset_shifts_wrist_target():
    """With a TCP offset, the wrist (FK origin) sits offset from the TCP target."""
    chain = parse_chain(URDF, "base_link", "L_sixforce_link")
    q_fixed = {"lifter_pitch_1_joint": 0.0, "lifter_pitch_2_joint": 0.0,
               "lifter_pitch_3_joint": 0.0, "waist_yaw_joint": 0.0}
    q_known = dict(q_fixed)
    q_known.update({k: v for k, v in zip(ARM, [0.4, -0.3, 0.2, -0.5, 0.1, 0.4, 0.3])})
    T = fk_chain(chain, q_known)
    tcp_local = (0.0, 0.0, 0.09)
    tcp_pos = T[:3, 3] + T[:3, :3] @ np.asarray(tcp_local)
    res = ik_solve_pose(
        chain, q_fixed, ARM, tcp_pos,
        approach_target=T[:3, :3] @ np.array([0.0, 0.0, 1.0]),
        paddle_target=T[:3, :3] @ np.array([1.0, 0.0, 0.0]),
        tool_approach_local=(0, 0, 1), tool_paddle_local=(1, 0, 0), tcp_offset_local=tcp_local,
    )
    assert res.converged
    wrist = fk_chain(chain, {**q_fixed, **res.q})[:3, 3]
    # wrist is ~9 cm from the TCP target (the offset), not coincident with it
    assert np.linalg.norm(wrist - tcp_pos) > 0.05
