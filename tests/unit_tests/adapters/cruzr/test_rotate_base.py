# coding: utf-8
"""rotate_base: in-place base spin via navigate_relative(0,0,dyaw); never touches arms."""

from types import SimpleNamespace

from jiuwensymbiosis.adapters.cruzr.api import CruzrApi


class _LL:
    def __init__(self):
        self.nav_calls = []
        self.moves = []

    def navigate_relative(self, dx, dy, dyaw):
        self.nav_calls.append((dx, dy, dyaw))
        return {"ok": True, "yaw_reached": dyaw}

    def move_joints_blocking(self, targets, **kw):   # must NOT be called by rotate_base
        self.moves.append(dict(targets))
        return {"ok": True}


class _Env:
    def __init__(self):
        self.low_level = _LL()
        self.cfg = SimpleNamespace()


def test_rotate_base_spins_in_place_without_touching_arms():
    env = _Env()
    api = CruzrApi(env)
    out = api.rotate_base(1.57)
    assert env.low_level.nav_calls == [(0.0, 0.0, 1.57)]   # dx=dy=0, dyaw passed through
    assert env.low_level.moves == []                        # arms never commanded
    assert out["ok"] is True
