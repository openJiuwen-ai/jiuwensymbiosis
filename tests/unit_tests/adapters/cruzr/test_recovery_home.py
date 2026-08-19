# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Payload tracking + payload-aware recovery retreat.

Without these, a motion failure while a box is clamped sends RecoveryRail into
``home()`` -> ``home_safely()`` -> arms to zero, which opens the paddles and drops
the box. ``env.holding_payload`` is what tells the rail to preserve the grip, and
``recovery_home`` is the retreat that keeps it.
"""

from pathlib import Path

import pytest

import jiuwensymbiosis.adapters.cruzr.geometry as lifter_mod
from jiuwensymbiosis.adapters.cruzr.geometry import LIFTER_JOINTS
from tests.unit_tests.adapters.cruzr.test_grasp_box_api import (
    _DET,
    _api,
    _Env,
    _fake_solve_arm_ik,
    _fake_solve_arm_ik_nonzero,
    _fake_solve_grasp,
    _no_move_lifter,
    _no_move_place,
)


def _needs_urdf():
    if not Path(_Env.cfg.urdf_path).exists():
        pytest.skip("urdf not present")


class TestHoldingPayloadFlag:
    def test_real_env_starts_empty_handed(self):
        from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
        from jiuwensymbiosis.adapters.cruzr.env import CruzrEnv

        assert CruzrEnv(CruzrConfig()).holding_payload is False

    def test_clamped_grasp_marks_payload_held(self, monkeypatch):
        _needs_urdf()
        import jiuwensymbiosis.adapters.cruzr.geometry as _gp_mod

        monkeypatch.setattr(_gp_mod, "solve_grasp", _fake_solve_grasp)
        monkeypatch.setattr(_gp_mod, "solve_arm_ik", _fake_solve_arm_ik)
        monkeypatch.setattr(lifter_mod, "search_lifter_for_box", _no_move_lifter)

        api, env = _api(monkeypatch)
        env.low_level.read_hand_ft = lambda arm: {"ok": True, "fmag": 10.0}
        assert api.dual_arm_grasp()["ok"] is True
        assert env.holding_payload is True

    def test_grasp_without_ft_contact_leaves_payload_clear(self, monkeypatch):
        _needs_urdf()
        import jiuwensymbiosis.adapters.cruzr.geometry as _gp_mod

        monkeypatch.setattr(_gp_mod, "solve_grasp", _fake_solve_grasp)
        monkeypatch.setattr(_gp_mod, "solve_arm_ik", _fake_solve_arm_ik)
        monkeypatch.setattr(lifter_mod, "search_lifter_for_box", _no_move_lifter)

        api, env = _api(monkeypatch)
        env.holding_payload = True  # a stale flag must not survive a failed grasp
        assert api.dual_arm_grasp()["reason"] == "no_contact"
        assert env.holding_payload is False

    def test_place_clears_payload(self, monkeypatch):
        _needs_urdf()
        import jiuwensymbiosis.adapters.cruzr.geometry as _gp_mod

        monkeypatch.setattr(_gp_mod, "solve_arm_ik", _fake_solve_arm_ik_nonzero)
        monkeypatch.setattr(lifter_mod, "search_lifter_for_place", _no_move_place)

        api, env = _api(monkeypatch)
        api._last_grasped_box = dict(_DET)
        env.holding_payload = True
        assert api.dual_arm_place()["ok"] is True
        assert env.holding_payload is False


class TestRecoveryHome:
    def test_keeps_arms_clamped_while_holding(self, monkeypatch):
        api, env = _api(monkeypatch)
        env.holding_payload = True

        out = api.recovery_home()

        assert out == {"ok": True, "payload_preserved": True}
        assert env.low_level.moves == []  # arms never commanded -> paddles stay closed
        assert env.low_level.lifter_calls == [dict.fromkeys(LIFTER_JOINTS, 0.0)]

    def test_neutralizes_rotated_waist_while_holding(self, monkeypatch):
        api, env = _api(monkeypatch)
        env.holding_payload = True
        env.low_level.get_joint_positions = lambda: {
            "lifter_pitch_1_joint": 0.0,
            "lifter_pitch_2_joint": 0.0,
            "lifter_pitch_3_joint": 0.0,
            "waist_yaw_joint": 0.9,
        }

        api.recovery_home()

        assert [c["target"] for c in env.low_level.turn_calls] == [0.0]
        assert env.low_level.moves == []

    def test_delegates_to_home_safely_when_empty(self, monkeypatch):
        api, env = _api(monkeypatch)  # _LL reports lifter 0, arms 0, waist 0

        assert api.recovery_home() == {"ok": True, "skipped": "already_home"}
        assert env.low_level.moves == []
        assert not getattr(env.low_level, "lifter_calls", [])
