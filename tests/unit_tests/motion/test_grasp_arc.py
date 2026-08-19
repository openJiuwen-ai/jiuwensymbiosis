# coding: utf-8
"""plan_grasp_arc_to_entry 的纯几何:base 系盒心 + 面法向 -> 一段弧把底盘搬到"面法线延长线上的待命点"。

弧 = 过原点切于 +X 且过待命点 E=C+d_e·n 的唯一圆,c=(Ex²+Ey²)/(2Ey),R=|c|,dyaw=2·atan2(Ey,Ex)。
安全全靠 plan-time 门控(半径过紧 / 转角超限 / 弧对箱心最近距离 < grasp_d)→ reject 退回离散。
"""

import math
from types import SimpleNamespace

from jiuwensymbiosis.motion.base_goal import plan_grasp_arc_to_entry


def _cfg(**over):
    base = {
        "grasp_target_forward_m": 0.60,      # grasp_d
        "grasp_arc_standoff_m": 0.10,        # d_e = 0.70
        "grasp_arc_min_radius_m": 0.35,
        "grasp_arc_max_dyaw_rad": 1.2,
    }
    base.update(over)
    return SimpleNamespace(**base)


def test_straight_when_entry_is_dead_ahead():
    # Box straight ahead, normal pointing straight back at the robot → E dead-ahead (Ey≈0) → no arc.
    plan = plan_grasp_arc_to_entry([1200.0, 0.0, 780.0], (-1.0, 0.0), _cfg())
    assert plan["mode"] == "straight"


def test_valid_arc_geometry():
    # Box ahead+left, face roughly toward the robot → a moderate, valid arc onto the line.
    n = (-0.9285, -0.3714)                    # ≈ unit box→robot for C=(1.0,0.4)
    plan = plan_grasp_arc_to_entry([1000.0, 400.0, 780.0], n, _cfg())
    assert plan["mode"] == "arc"
    # E = C + d_e·n on the normal line, one standoff (0.70) from the centre.
    ex, ey = plan["E"]
    assert abs(ex - 0.35005) < 1e-3 and abs(ey - 0.14002) < 1e-3
    assert abs(math.hypot(ex - 1.0, ey - 0.4) - 0.70) < 1e-3          # |E-C| == d_e
    # R = |c|, dyaw = 2·bearing(E); left box → CCW → s=+1.
    assert abs(plan["R"] - 0.50758) < 1e-3
    assert abs(plan["dyaw_arc"] - 0.76102) < 1e-3
    assert plan["s"] == 1.0
    # Safety: the whole arc path stays at least grasp_d from the box centre.
    assert plan["closest"] >= 0.60 - 1e-9


def test_sign_fix_is_invariant():
    # The stored face_normal sign is arbitrary; n and -n must plan identically (flip toward robot).
    C = [1000.0, 400.0, 780.0]
    a = plan_grasp_arc_to_entry(C, (-0.9285, -0.3714), _cfg())
    b = plan_grasp_arc_to_entry(C, (0.9285, 0.3714), _cfg())
    assert a["mode"] == b["mode"] == "arc"
    assert abs(a["R"] - b["R"]) < 1e-9
    assert abs(a["dyaw_arc"] - b["dyaw_arc"]) < 1e-9


def test_reject_dyaw_too_large():
    # Face tilted so the entry point sits at a large bearing → swing exceeds the FOV cap → reject.
    plan = plan_grasp_arc_to_entry([1000.0, 0.0, 780.0], (-0.8, 0.6), _cfg())
    assert plan["mode"] == "reject"
    assert plan["reason"] == "dyaw_too_large"


def test_reject_radius_too_tight():
    # Box close and well off to the side → tight arc (small R) → reject (could pivot into the box).
    plan = plan_grasp_arc_to_entry([300.0, 800.0, 780.0], (-0.351, -0.937), _cfg())
    assert plan["mode"] == "reject"


def test_reject_when_already_inside_grasp_distance():
    # Robot already closer than grasp_d to the box centre → the arc origin is inside the grasp disk /
    # the entry point falls behind → reject, so the discrete in-band loop handles it.
    plan = plan_grasp_arc_to_entry([450.0, 100.0, 780.0], (-0.976, -0.217), _cfg())
    assert plan["mode"] == "reject"


def test_no_normal_line_when_centre_at_origin():
    plan = plan_grasp_arc_to_entry([0.0, 0.0, 780.0], (-1.0, 0.0), _cfg())
    assert plan["mode"] == "reject"
    assert plan["reason"] == "no_normal_line"
