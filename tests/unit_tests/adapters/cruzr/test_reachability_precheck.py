# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Plan-time URDF reachability precheck: ``CruzrApi.check_reachable`` runs a current-pose
IK feasibility (reusing ``search_lifter_for_box``) to annotate scene objects. Conservative by
design — missing joint state / bad object / any IK error must return False (keeps approach_for_grasp)."""

from types import SimpleNamespace

import jiuwensymbiosis.adapters.cruzr.geometry as geometry
import jiuwensymbiosis.kinematics.urdf_chain as urdf_chain
from jiuwensymbiosis.adapters.cruzr.api import CruzrApi
from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
from jiuwensymbiosis.adapters.cruzr.env import CruzrEnv

_JOINTS = {"lifter_pitch_1_joint": 0.0, "lifter_pitch_2_joint": 0.0,
           "lifter_pitch_3_joint": 0.0, "waist_yaw_joint": 0.0}
_OBJ = {"center_mm": [500.0, 0.0, 600.0], "width_mm": 200.0, "height_mm": 150.0, "forward_mm": 500.0}


def _api(joints):
    env = CruzrEnv.__new__(CruzrEnv)
    env.cfg = CruzrConfig()
    api = CruzrApi(env)
    api._ll = lambda: SimpleNamespace(get_joint_positions=lambda: joints)  # type: ignore[method-assign]
    return api


def _patch_ik(monkeypatch, found):
    monkeypatch.setattr(urdf_chain, "parse_chain", lambda *a, **k: object())
    monkeypatch.setattr(geometry, "search_lifter_for_box", lambda *a, **k: SimpleNamespace(found=found))


def test_reachable_true(monkeypatch):
    _patch_ik(monkeypatch, found=True)
    assert _api(_JOINTS).check_reachable(_OBJ) is True


def test_reachable_false(monkeypatch):
    _patch_ik(monkeypatch, found=False)
    assert _api(_JOINTS).check_reachable(_OBJ) is False


def test_reachable_false_without_joint_state(monkeypatch):
    _patch_ik(monkeypatch, found=True)  # even if IK would say yes,
    assert _api({}).check_reachable(_OBJ) is False  # no joint state → conservative False


def test_reachable_false_on_bad_object():
    assert _api(_JOINTS).check_reachable({"center_mm": None}) is False


def test_reachable_false_on_ik_error(monkeypatch):
    monkeypatch.setattr(urdf_chain, "parse_chain", lambda *a, **k: object())

    def boom(*a, **k):
        raise RuntimeError("pin build failed")

    monkeypatch.setattr(geometry, "search_lifter_for_box", boom)
    assert _api(_JOINTS).check_reachable(_OBJ) is False  # any error → not reachable
