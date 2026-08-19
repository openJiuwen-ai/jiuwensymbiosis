# coding: utf-8
"""api.turn_waist: relative delta, limit clamp, arms held, no-state guard."""

from types import SimpleNamespace

import jiuwensymbiosis.kinematics.urdf_chain as urdf_chain_mod
from jiuwensymbiosis.adapters.cruzr.api import CruzrApi
from jiuwensymbiosis.adapters.cruzr.geometry import ARM_JOINTS

_ARMS = [j for a in ("left", "right") for j in ARM_JOINTS[a]]


class _FakeChain:
    def limits(self):
        return {"waist_yaw_joint": (-1.57, 1.57)}


class _LL:
    def __init__(self, q):
        self._q = q
        self.turn_calls = []

    def get_joint_positions(self):
        return dict(self._q)

    def turn_waist_blocking(self, target_rad, *, hold, waist_joint="waist_yaw_joint", step_rad=None):
        self.turn_calls.append({"target": target_rad, "hold": dict(hold)})
        return {"ok": True, "readback": {waist_joint: target_rad}}


class _Env:
    def __init__(self, q):
        self.low_level = _LL(q)
        self.cfg = SimpleNamespace(
            waist_yaw_joint="waist_yaw_joint",
            urdf_path="/nonexistent.urdf",
            left_arm_leaf="L_sixforce_link",
            right_arm_leaf="R_sixforce_link",
        )


def _q(waist=0.0):
    q = {"waist_yaw_joint": waist}
    for j in _ARMS:
        q[j] = 0.1
    return q


def _api(monkeypatch, q):
    monkeypatch.setattr(urdf_chain_mod, "parse_chain", lambda *a, **k: _FakeChain())
    env = _Env(q)
    return CruzrApi(env), env


def test_turn_waist_relative_target_and_holds_arms(monkeypatch):
    api, env = _api(monkeypatch, _q(waist=0.2))
    out = api.turn_waist(0.3)
    assert out["ok"] is True
    assert abs(out["from_rad"] - 0.2) < 1e-9
    assert abs(out["to_rad"] - 0.5) < 1e-9
    assert out["clamped"] is False
    call = env.low_level.turn_calls[-1]
    assert abs(call["target"] - 0.5) < 1e-9
    assert all(j in call["hold"] for j in _ARMS)     # both arms held
    assert all(v == 0.1 for v in call["hold"].values())


def test_turn_waist_clamps_to_limit(monkeypatch):
    api, env = _api(monkeypatch, _q(waist=1.5))
    out = api.turn_waist(0.3)                          # 1.5 + 0.3 = 1.8 > 1.57
    assert out["clamped"] is True
    assert abs(out["to_rad"] - 1.57) < 1e-9
    assert abs(env.low_level.turn_calls[-1]["target"] - 1.57) < 1e-9


def test_turn_waist_zero_delta_is_noop_target(monkeypatch):
    api, env = _api(monkeypatch, _q(waist=0.4))
    out = api.turn_waist(0.0)
    assert out["ok"] is True
    assert out["clamped"] is False
    assert abs(out["to_rad"] - out["from_rad"]) < 1e-9


def test_turn_waist_no_joint_state(monkeypatch):
    api, env = _api(monkeypatch, {})                  # no waist reading
    out = api.turn_waist(0.3)
    assert out == {"ok": False, "reason": "no_joint_state"}
    assert env.low_level.turn_calls == []             # never commanded motion


def test_waist_yaw_on_real_arm_chain():
    """Hardware seam: waist_yaw_joint must be on the real base_link->arm chain.

    Every other test monkeypatches parse_chain, so a missing joint would only
    surface on hardware as a KeyError in turn_waist's limits()[waist] lookup.
    This guards that seam against the real URDF (skipped when it is absent).
    """
    from pathlib import Path

    import pytest

    from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
    from jiuwensymbiosis.kinematics.urdf_chain import parse_chain

    cfg = CruzrConfig()
    if not Path(cfg.urdf_path).exists():
        pytest.skip("urdf not present")
    limits = parse_chain(cfg.urdf_path, "base_link", cfg.left_arm_leaf).limits()
    assert cfg.waist_yaw_joint in limits
