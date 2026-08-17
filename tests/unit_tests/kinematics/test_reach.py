# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Framework-generic URDF reachability (kinematics/reach.py): reachable_point wraps the numpy 5-DoF
IK (converged flag), reach_envelope probes a small grid. Both are body-agnostic; IK is mocked here so
the tests don't need a real URDF."""

from types import SimpleNamespace

import jiuwensymbiosis.kinematics.reach as reach
from jiuwensymbiosis.kinematics.ik import IKResult


def _ik(converged):
    return lambda *a, **k: IKResult(q={}, converged=converged, pos_err_m=0.0, normal_err=0.0, iters=1)


def _chain():
    return SimpleNamespace(movable_names=lambda: ["j1", "j2"])


def test_reachable_point_true(monkeypatch):
    monkeypatch.setattr(reach, "ik_solve_5dof", _ik(True))
    assert reach.reachable_point(_chain(), (500, 0, 600), {"j1": 0.0, "j2": 0.0}) is True


def test_reachable_point_false(monkeypatch):
    monkeypatch.setattr(reach, "ik_solve_5dof", _ik(False))
    assert reach.reachable_point(_chain(), (2500, 0, 600), {"j1": 0.0}) is False


def test_reach_envelope_reports_reachable_extent(monkeypatch):
    # Reachable only within ~0.6 m forward → envelope's forward max stays below the far probes.
    monkeypatch.setattr(reach, "reachable_point",
                        lambda chain, xyz, q, **k: (xyz[0] / 1000.0) <= 0.6)
    env = reach.reach_envelope(_chain(), {})
    assert env["reachable"] is True
    assert env["forward_m"][1] <= 0.6 + 1e-9
    assert set(env) == {"reachable", "forward_m", "lateral_m", "height_m"}


def test_reach_envelope_nothing_reachable(monkeypatch):
    monkeypatch.setattr(reach, "reachable_point", lambda *a, **k: False)
    assert reach.reach_envelope(_chain(), {}) == {"reachable": False}
