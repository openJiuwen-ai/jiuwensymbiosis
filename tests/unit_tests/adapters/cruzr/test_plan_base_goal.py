# coding: utf-8
"""approach_for_grasp 的纯几何：base 系盒心 -> (turn, forward, status)。"""

import math
from types import SimpleNamespace

from jiuwensymbiosis.motion.base_goal import plan_base_goal_for_grasp


def _cfg(**over):
    base = {
        "grasp_target_forward_m": 0.40,
        "grasp_forward_min_m": 0.30,
        "grasp_forward_max_m": 0.50,
        "base_yaw_tol_rad": 0.08,
    }
    base.update(over)
    return SimpleNamespace(**base)


def test_centered_in_band_is_in_band_no_move():
    turn, forward, status = plan_base_goal_for_grasp([400.0, 0.0, 780.0], _cfg())
    assert status == "in_band"
    assert abs(turn) < 1e-9
    assert abs(forward - 0.0) < 1e-9


def test_far_ahead_is_ok_forward_positive():
    turn, forward, status = plan_base_goal_for_grasp([700.0, 0.0, 780.0], _cfg())
    assert status == "ok"
    assert abs(turn) < 1e-9
    assert abs(forward - 0.30) < 1e-9


def test_nearer_than_band_is_too_close():
    turn, forward, status = plan_base_goal_for_grasp([200.0, 0.0, 780.0], _cfg())
    assert status == "too_close"
    assert forward < 0.0


def test_in_band_but_off_center_is_ok_with_turn():
    # rng=hypot(.4,.2)=.447 (in band), turn=atan2(.2,.4)=0.4636 (> yaw_tol) -> ok, small forward
    turn, forward, status = plan_base_goal_for_grasp([400.0, 200.0, 780.0], _cfg())
    assert status == "ok"
    assert abs(turn - math.atan2(0.2, 0.4)) < 1e-9
    assert abs(forward - (math.hypot(0.4, 0.2) - 0.40)) < 1e-9


def test_turn_sign_left_positive():
    # box to the LEFT (y>0) -> turn CCW (+)
    turn, _f, _s = plan_base_goal_for_grasp([500.0, 300.0, 780.0], _cfg())
    assert turn > 0.0
