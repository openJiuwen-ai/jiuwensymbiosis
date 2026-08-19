# coding: utf-8
"""dual_arm_place surface z-shift: box bottom lands on the sensed surface, not the grasp height."""

from types import SimpleNamespace

from jiuwensymbiosis.adapters.cruzr.api import CruzrApi, _shift_target_z
from jiuwensymbiosis.adapters.cruzr.geometry import ArmTarget, plan_clamp_targets
from jiuwensymbiosis.perception.object_geometry import ObjectGeometry3D


def test_shift_target_z_moves_only_z():
    tgt = ArmTarget("left", (0.4, 0.1, 0.6), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-0.09, 0.0, 0.0))
    out = _shift_target_z(tgt, 0.05)
    assert out.pos_m == (0.4, 0.1, 0.65)              # z + 0.05
    assert out.approach == (1.0, 0.0, 0.0)            # dirs unchanged
    assert out.paddle == (0.0, 1.0, 0.0)
    assert out.tcp_offset_local == (-0.09, 0.0, 0.0)


def _box_and_targets():
    b = ObjectGeometry3D(True, "", (350.0, 0.0, 700.0), 270.0, 200.0, 290.0, 800.0, 5000, back_x_mm=410.0)
    return b, plan_clamp_targets(b)                    # pure geometry, no URDF/IK


def _api():
    env = SimpleNamespace(cfg=SimpleNamespace(), low_level=None)
    return CruzrApi(env)


def test_apply_surface_z_shifts_by_surface_minus_box_bottom():
    api = _api()
    b, (approach, descend, clamp) = _box_and_targets()
    # box bottom (base z, mm) = top_z_mm - height_mm = 800 - 200 = 600
    # place onto a surface 100 mm higher -> dz = (700 - 600)/1000 = +0.10 m
    _, _, clamp2 = api._apply_surface_z(b, (approach, descend, clamp), surface_z_mm=700.0)
    for a in ("left", "right"):
        assert clamp2[a].pos_m[2] == clamp[a].pos_m[2] + 0.10   # z shifted up 0.10 m
        assert clamp2[a].pos_m[:2] == clamp[a].pos_m[:2]        # x,y unchanged


def test_apply_surface_z_noop_when_none():
    api = _api()
    b, tgts = _box_and_targets()
    assert api._apply_surface_z(b, tgts, surface_z_mm=None) is tgts   # untouched
