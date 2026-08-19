# coding: utf-8
"""plan_grasp_right_angle 的纯几何:base 系盒心 + 面法向 → 直角(L 形)路线把底盘弄到"面法线线上的待命点"。

先【侧移】垂直于法向上线(不消耗到箱子的前向距离,近距离也有效),再【转正法向直入】到待命点 E=C+d_e·n。
两段互相垂直的直线 + 原地转,差速底盘不需要横移;几何上覆盖所有情形(不像弧线会 reject)。
"""

import math
from types import SimpleNamespace

from jiuwensymbiosis.motion.base_goal import plan_grasp_right_angle


def _cfg(**over):
    base = {"grasp_target_forward_m": 0.60, "grasp_arc_standoff_m": 0.25, "base_pos_tol_m": 0.05}
    base.update(over)
    return SimpleNamespace(**base)


def test_on_line_is_straight_in_only():
    # box straight ahead, normal points straight back at the robot → robot already ON the line → no lateral leg
    plan = plan_grasp_right_angle([1200.0, 0.0, 780.0], (-1.0, 0.0), _cfg())
    assert plan["mode"] == "straight"
    assert plan["lat_dist"] == 0.0
    assert abs(plan["app_turn"] - 0.0) < 1e-9
    assert abs(plan["app_dist"] - (1.2 - 0.85)) < 1e-3        # drive rng−d_e to reach E


def test_corner_on_is_lateral_then_approach():
    # box ahead, face normal 30° off the robot→box line (corner-on) → robot OFF the line → full L
    plan = plan_grasp_right_angle([1500.0, 0.0, 780.0], (-0.866, 0.5), _cfg())
    assert plan["mode"] == "l"
    assert abs(plan["lat_turn"] - 1.047) < 1e-3               # turn ⊥ to the normal, toward the line (+60°)
    assert abs(plan["lat_dist"] - 0.75) < 1e-3               # perpendicular offset |cx·ny − cy·nx|
    assert abs(plan["app_turn"] - (-0.524)) < 1e-3           # then face the box along −n (−30°)
    assert abs(plan["app_dist"] - 0.449) < 1e-3             # straight-in from the line foot to E
    # verify the two legs (executed from the robot origin) actually land on E = C + d_e·n
    fx = plan["lat_dist"] * math.cos(plan["lat_turn"])       # robot starts at (0,0)
    fy = plan["lat_dist"] * math.sin(plan["lat_turn"])
    ex = fx + plan["app_dist"] * math.cos(plan["app_turn"])
    ey = fy + plan["app_dist"] * math.sin(plan["app_turn"])
    assert abs(ex - plan["E"][0]) < 1e-3 and abs(ey - plan["E"][1]) < 1e-3


def test_sign_fix_is_invariant():
    # the stored face_normal sign is arbitrary; n and −n must plan identically (flip toward robot)
    a = plan_grasp_right_angle([1500.0, 0.0, 780.0], (-0.866, 0.5), _cfg())
    b = plan_grasp_right_angle([1500.0, 0.0, 780.0], (0.866, -0.5), _cfg())
    assert a["mode"] == b["mode"] == "l"
    assert abs(a["lat_turn"] - b["lat_turn"]) < 1e-9 and abs(a["lat_dist"] - b["lat_dist"]) < 1e-9
    assert abs(a["app_turn"] - b["app_turn"]) < 1e-9 and abs(a["app_dist"] - b["app_dist"]) < 1e-9


def test_lateral_leg_sign_flips_with_box_side():
    # box off to the OTHER side → lateral turn has the opposite sign
    left = plan_grasp_right_angle([1500.0, 0.0, 780.0], (-0.866, 0.5), _cfg())
    right = plan_grasp_right_angle([1500.0, 0.0, 780.0], (-0.866, -0.5), _cfg())
    assert left["lat_turn"] * right["lat_turn"] < 0.0        # opposite lateral directions


def test_degenerate_centre_rejects():
    plan = plan_grasp_right_angle([0.0, 0.0, 780.0], (-1.0, 0.0), _cfg())
    assert plan["mode"] == "reject"
