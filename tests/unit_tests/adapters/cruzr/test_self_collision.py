# coding: utf-8
"""self_collision (pin+coal): neutral is clear after excluding adjacent pairs; a folded-in arm collides."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
from jiuwensymbiosis.kinematics import self_collision as sc

_MESH = "/home/riemann/Robot/Cruzr_ws/cruzr_s2_description/cruzr_s2_description/meshes"


def _cfg_or_skip():
    cfg = CruzrConfig()
    if not Path(cfg.urdf_path).exists() or not Path(_MESH).exists():
        pytest.skip("urdf/meshes not present")
    return cfg


def test_available_and_neutral_clear():
    cfg = _cfg_or_skip()
    assert sc.available(cfg.urdf_path, cfg.urdf_package_dir) is True
    q0 = sc.full_q(cfg.urdf_path, cfg.urdf_package_dir, {})   # neutral
    assert sc.in_self_collision(cfg.urdf_path, cfg.urdf_package_dir, q0) is False


def test_folded_arm_collides():
    cfg = _cfg_or_skip()
    jv = {"L_shoulder_pitch_joint": 0.2, "L_shoulder_roll_joint": 1.2, "L_elbow_roll_joint": -1.8}
    q = sc.full_q(cfg.urdf_path, cfg.urdf_package_dir, jv)
    assert sc.in_self_collision(cfg.urdf_path, cfg.urdf_package_dir, q) is True


def test_unavailable_degrades_gracefully():
    # a bogus urdf path -> build fails -> available False, in_self_collision False (no crash)
    assert sc.available("/nope.urdf", "/nope") is False
    assert sc.in_self_collision("/nope.urdf", "/nope", np.zeros(3)) is False
