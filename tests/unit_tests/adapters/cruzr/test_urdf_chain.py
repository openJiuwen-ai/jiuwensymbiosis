# coding: utf-8
from pathlib import Path

import pytest

from jiuwensymbiosis.kinematics.urdf_chain import parse_chain

URDF = "/home/riemann/Robot/Cruzr_ws/cruzr_s2_description/cruzr_s2_description/urdf/cruzr_s2_v1/cruzr_s2_v1.urdf"


@pytest.mark.skipif(not Path(URDF).exists(), reason="urdf not present")
def test_left_arm_chain_movable_joints():
    chain = parse_chain(URDF, "base_link", "L_sixforce_link")
    movable = chain.movable_names()
    # lifter(3) + waist_yaw(1) + arm(7) = 11 revolute joints
    assert movable == [
        "lifter_pitch_1_joint", "lifter_pitch_2_joint", "lifter_pitch_3_joint",
        "waist_yaw_joint",
        "L_shoulder_pitch_joint", "L_shoulder_roll_joint", "L_shoulder_yaw_joint",
        "L_elbow_roll_joint", "L_elbow_yaw_joint",
        "L_wrist_pitch_joint", "L_wrist_roll_joint",
    ]


@pytest.mark.skipif(not Path(URDF).exists(), reason="urdf not present")
def test_limits_are_parsed():
    chain = parse_chain(URDF, "base_link", "L_sixforce_link")
    lo, hi = chain.limits()["L_shoulder_pitch_joint"]
    assert lo == pytest.approx(-2.8623, abs=1e-3)
    assert hi == pytest.approx(2.8623, abs=1e-3)


@pytest.mark.skipif(not Path(URDF).exists(), reason="urdf not present")
def test_fixed_joint_origin_preserved():
    chain = parse_chain(URDF, "base_link", "L_sixforce_link")
    base_lifter = next(j for j in chain.joints if j.name == "base_lifter_joint")
    assert base_lifter.jtype == "fixed"
    assert base_lifter.xyz == pytest.approx((0.0, 0.0, 0.281), abs=1e-6)


def test_parse_chain_records_urdf_path_and_leaf():
    from pathlib import Path
    import pytest
    from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
    from jiuwensymbiosis.kinematics.urdf_chain import parse_chain

    cfg = CruzrConfig()
    if not Path(cfg.urdf_path).exists():
        pytest.skip("urdf not present")
    chain = parse_chain(cfg.urdf_path, "base_link", cfg.left_arm_leaf)
    assert chain.urdf_path == cfg.urdf_path
    assert chain.leaf_link == cfg.left_arm_leaf
