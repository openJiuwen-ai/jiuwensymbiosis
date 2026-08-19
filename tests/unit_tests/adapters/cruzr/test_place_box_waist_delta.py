# coding: utf-8
"""dual_arm_place rotates its paddle targets by the waist delta so a box moved by
turn_waist between grasp and place is released at its NEW location, not the old one."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.adapters.cruzr.api import CruzrApi, _rotate_arm_target
from jiuwensymbiosis.adapters.cruzr.geometry import ArmTarget, plan_clamp_targets
from jiuwensymbiosis.perception.object_geometry import ObjectGeometry3D

_LIFTER0 = {"lifter_pitch_1_joint": 0.0, "lifter_pitch_2_joint": 0.0, "lifter_pitch_3_joint": 0.0}


def test_rotate_arm_target_rotates_pos_and_dirs():
    """A 90° yaw (Rz) rotates pos + approach/paddle dirs; tcp_offset_local is invariant."""
    rz90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([0.0, 0.0, 0.0])
    tgt = ArmTarget("left", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-0.09, 0.0, 0.0))

    out = _rotate_arm_target(tgt, rz90, t)

    assert np.allclose(out.pos_m, (0.0, 1.0, 0.0))         # (1,0,0) -> (0,1,0)
    assert np.allclose(out.approach, (0.0, 1.0, 0.0))
    assert np.allclose(out.paddle, (-1.0, 0.0, 0.0))       # (0,1,0) -> (-1,0,0)
    assert out.tcp_offset_local == (-0.09, 0.0, 0.0)       # tool-frame: unchanged


def _api_with_real_cfg():
    from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig

    cfg = CruzrConfig()
    if not Path(cfg.urdf_path).exists():
        pytest.skip("urdf not present")
    env = SimpleNamespace(cfg=cfg, low_level=None)
    return CruzrApi(env), cfg


def _targets():
    box = ObjectGeometry3D(True, "", (350.0, 0.0, 700.0), 270.0, 200.0, 290.0, 800.0, 5000, back_x_mm=410.0)
    return plan_clamp_targets(box)


def test_waist_delta_noop_without_recorded_grasp_waist():
    api, cfg = _api_with_real_cfg()
    from jiuwensymbiosis.kinematics.urdf_chain import parse_chain

    chain = parse_chain(cfg.urdf_path, "base_link", cfg.left_arm_leaf)
    tgts = _targets()
    q_fixed = {**_LIFTER0, "waist_yaw_joint": 0.3}

    assert api._last_grasp_waist_yaw is None
    out = api._rotate_targets_for_waist_delta(chain, q_fixed, tgts)
    assert out is tgts                                     # untouched: plain place, no prior turn


def test_waist_delta_noop_when_no_turn():
    api, cfg = _api_with_real_cfg()
    from jiuwensymbiosis.kinematics.urdf_chain import parse_chain

    chain = parse_chain(cfg.urdf_path, "base_link", cfg.left_arm_leaf)
    tgts = _targets()
    api._last_grasp_waist_yaw = 0.4
    q_fixed = {**_LIFTER0, "waist_yaw_joint": 0.4}          # current == grasp -> no rotation

    out = api._rotate_targets_for_waist_delta(chain, q_fixed, tgts)
    assert out is tgts


def test_waist_delta_rotates_targets_on_turn():
    api, cfg = _api_with_real_cfg()
    from jiuwensymbiosis.kinematics.urdf_chain import parse_chain

    chain = parse_chain(cfg.urdf_path, "base_link", cfg.left_arm_leaf)
    approach, descend, clamp = _targets()
    api._last_grasp_waist_yaw = 0.0
    q_fixed = {**_LIFTER0, "waist_yaw_joint": 0.3}          # turned +0.3 rad since grasp

    _, _, clamp2 = api._rotate_targets_for_waist_delta(chain, q_fixed, (approach, descend, clamp))

    for a in ("left", "right"):
        old, new = np.asarray(clamp[a].pos_m), np.asarray(clamp2[a].pos_m)
        assert not np.allclose(old, new)                   # target moved with the box
        assert abs(new[2] - old[2]) < 1e-6                 # upright torso: waist axis vertical, z preserved
        assert not np.allclose(clamp[a].paddle, clamp2[a].paddle)  # paddle-face dir rotated too
