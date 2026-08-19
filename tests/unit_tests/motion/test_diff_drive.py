# coding: utf-8
"""``motion.diff_drive`` —— 差速底盘轮速控制律（纯数学，无 ROS）。

这些公式此前埋在 cruzr 的 wheel worker 循环里、只在真机上跑过；提到框架层后第一次有了
硬件无关的覆盖。约定：返回 ``(left, right)``，+yaw = 左/CCW ⇒ 左轮慢于右轮。
"""

from __future__ import annotations

import math

import pytest

from jiuwensymbiosis.motion import diff_drive as dd


class TestWrapAngle:
    @pytest.mark.parametrize("raw,expected", [
        (0.0, 0.0), (math.pi / 2, math.pi / 2), (-math.pi / 2, -math.pi / 2),
        (3 * math.pi, math.pi), (2 * math.pi + 0.3, 0.3), (-2 * math.pi - 0.3, -0.3),
    ])
    def test_normalises_into_the_shortest_angle(self, raw, expected):
        assert dd.wrap_angle(raw) == pytest.approx(expected, abs=1e-9)

    def test_a_turn_just_past_pi_becomes_the_short_way_round(self):
        # 359° 的目标不能走成 359°，必须走 -1°，否则底盘绕远路。
        assert dd.wrap_angle(math.radians(359)) == pytest.approx(math.radians(-1), abs=1e-9)


class TestRampedSpeed:
    def test_full_speed_while_far_outside_the_slow_band(self):
        assert dd.ramped_speed(0.8, 0.25, remaining=5.0, slow_band=0.25) == 0.8

    def test_tapers_linearly_inside_the_band(self):
        assert dd.ramped_speed(0.8, 0.0, remaining=0.125, slow_band=0.25) == pytest.approx(0.4)

    def test_never_drops_below_the_deadband_floor(self):
        # 关键安全性质：降到 0 会让底盘停在目标前不动（轮子进死区），必须有下限。
        assert dd.ramped_speed(0.8, 0.25, remaining=1e-9, slow_band=0.25) == 0.25

    def test_zero_slow_band_disables_the_taper(self):
        assert dd.ramped_speed(0.8, 0.25, remaining=0.001, slow_band=0.0) == 0.8

    def test_remaining_is_a_magnitude_not_a_signed_error(self):
        # 传有符号误差会在越过目标后掉到 min_speed,而不是"接近目标才减速"。
        # 方向由调用方负责(它决定哪只轮子领先),这里只管速度大小。
        assert dd.ramped_speed(0.8, 0.1, -0.1, 0.25) == 0.1
        assert dd.ramped_speed(0.8, 0.1, 0.1, 0.25) == pytest.approx(0.32)


class TestRotateWheels:
    def test_left_turn_drives_right_wheel_faster(self):
        left, right = dd.rotate_wheels(1.0, k_rot=0.6, k_rot_min=0.25, k_rot_slow_rad=0.5)
        assert left < 0 < right and left == -right

    def test_right_turn_mirrors_it(self):
        left, right = dd.rotate_wheels(-1.0, k_rot=0.6, k_rot_min=0.25, k_rot_slow_rad=0.5)
        assert right < 0 < left and left == -right

    def test_overshoot_reverses_instead_of_spinning_on(self):
        # 误差换号 ⇒ 轮速换号。这是"过冲自纠"而不是继续转圈的根据。
        before = dd.rotate_wheels(0.05, k_rot=0.6, k_rot_min=0.25, k_rot_slow_rad=0.5)
        after = dd.rotate_wheels(-0.05, k_rot=0.6, k_rot_min=0.25, k_rot_slow_rad=0.5)
        assert before == (-after[0], -after[1])

    def test_decelerates_near_the_target(self):
        far = dd.rotate_wheels(2.0, k_rot=0.6, k_rot_min=0.1, k_rot_slow_rad=0.5)
        near = dd.rotate_wheels(0.1, k_rot=0.6, k_rot_min=0.1, k_rot_slow_rad=0.5)
        assert abs(near[1]) < abs(far[1])


class TestSpinWheels:
    def test_constant_speed_both_directions(self):
        assert dd.spin_wheels(1.0, 0.6) == (-0.6, 0.6)
        assert dd.spin_wheels(-1.0, 0.6) == (0.6, -0.6)

    def test_zero_direction_counts_as_left(self):
        assert dd.spin_wheels(0.0, 0.6) == (-0.6, 0.6)


class TestForwardWheels:
    def test_both_wheels_equal_when_driving_straight(self):
        left, right = dd.forward_wheels(1.0, 1.0, k_fwd=0.8, k_fwd_min=0.25, k_fwd_slow_m=0.25)
        assert left == right > 0

    def test_reverse_is_the_same_law_negated(self):
        fwd = dd.forward_wheels(1.0, 1.0, k_fwd=0.8, k_fwd_min=0.25, k_fwd_slow_m=0.25)
        rev = dd.forward_wheels(1.0, -1.0, k_fwd=0.8, k_fwd_min=0.25, k_fwd_slow_m=0.25)
        assert rev == (-fwd[0], -fwd[1])

    def test_decelerates_into_the_target_distance(self):
        far = dd.forward_wheels(2.0, 1.0, k_fwd=0.8, k_fwd_min=0.1, k_fwd_slow_m=0.25)
        near = dd.forward_wheels(0.05, 1.0, k_fwd=0.8, k_fwd_min=0.1, k_fwd_slow_m=0.25)
        assert near[0] < far[0]


class TestSteeredWheels:
    def test_positive_bearing_curves_left(self):
        left, right = dd.steered_wheels(0.3, k_fwd=0.5, k_steer=0.6, steer_max=0.4)
        assert left < right

    def test_negative_bearing_curves_right(self):
        left, right = dd.steered_wheels(-0.3, k_fwd=0.5, k_steer=0.6, steer_max=0.4)
        assert right < left

    def test_zero_gain_degrades_to_a_straight_creep(self):
        assert dd.steered_wheels(1.5, k_fwd=0.5, k_steer=0.0, steer_max=0.4) == (0.5, 0.5)

    def test_steer_bias_is_clamped(self):
        wide = dd.steered_wheels(10.0, k_fwd=0.5, k_steer=0.6, steer_max=0.4)
        capped = dd.steered_wheels(0.4 / 0.6, k_fwd=0.5, k_steer=0.6, steer_max=0.4)
        assert wide == pytest.approx(capped)

    def test_no_wheel_is_ever_driven_backwards(self):
        # 大偏置下若允许负值,内轮反转 ⇒ 变成原地打转、接近停止推进。
        left, right = dd.steered_wheels(10.0, k_fwd=0.2, k_steer=2.0, steer_max=5.0)
        assert left >= 0.0 and right >= 0.0


class TestMeasuredCurvature:
    def test_ratio_of_heading_change_to_arc_length(self):
        assert dd.measured_curvature(0.2, 0.4, fallback=9.0) == pytest.approx(0.5)

    def test_barely_moved_falls_back_instead_of_exploding(self):
        assert dd.measured_curvature(0.2, 1e-9, fallback=9.0) == 9.0

    def test_sign_of_heading_change_is_ignored(self):
        assert dd.measured_curvature(-0.2, 0.4, fallback=9.0) == pytest.approx(0.5)


class TestArc:
    def test_delta_grows_when_measured_curvature_is_too_low(self):
        assert dd.arc_delta(0.0, target_curvature=1.0, measured=0.5, gain=0.5, k_fwd=0.6) > 0.0

    def test_delta_shrinks_when_curving_too_hard(self):
        assert dd.arc_delta(0.3, target_curvature=0.5, measured=1.5, gain=0.5, k_fwd=0.6) < 0.3

    def test_delta_never_goes_negative(self):
        assert dd.arc_delta(0.0, target_curvature=0.1, measured=99.0, gain=0.5, k_fwd=0.6) == 0.0

    def test_delta_is_capped_below_the_forward_speed(self):
        # 上限 0.9*k_fwd:再大内轮就要反转,弧线退化成原地打转。
        assert dd.arc_delta(0.0, target_curvature=99.0, measured=0.0, gain=1.0, k_fwd=0.6) == pytest.approx(0.54)

    def test_left_arc_drives_right_wheel_faster(self):
        left, right = dd.arc_wheels(1.0, 1.0, 0.1, k_fwd=0.6, k_fwd_min=0.25, slow_band_rad=0.5)
        assert left < right

    def test_right_arc_mirrors_it(self):
        left, right = dd.arc_wheels(1.0, -1.0, 0.1, k_fwd=0.6, k_fwd_min=0.25, slow_band_rad=0.5)
        assert right < left

    def test_decelerates_as_the_heading_target_is_reached(self):
        far = dd.arc_wheels(2.0, 1.0, 0.0, k_fwd=0.6, k_fwd_min=0.1, slow_band_rad=0.5)
        near = dd.arc_wheels(0.05, 1.0, 0.0, k_fwd=0.6, k_fwd_min=0.1, slow_band_rad=0.5)
        assert near[1] < far[1]

    def test_no_wheel_is_ever_driven_backwards(self):
        left, right = dd.arc_wheels(0.01, 1.0, 5.0, k_fwd=0.6, k_fwd_min=0.1, slow_band_rad=0.5)
        assert left >= 0.0 and right >= 0.0


class TestSectorMinRange:
    @staticmethod
    def _scan(ranges, *, angle_min=-math.pi / 2, increment=math.pi / 4):
        return {"ranges": ranges, "angle_min": angle_min, "angle_increment": increment}

    def test_picks_the_closest_beam_inside_the_sector(self):
        # 角度: -90 -45 0 +45 +90;半扇区 50° ⇒ 只看 -45/0/+45 三束。
        out = dd.sector_min_range(**self._scan([0.2, 3.0, 2.0, 1.0, 0.3]),
                                  half_sector_rad=math.radians(50), range_min=0.05,
                                  range_max=10.0, self_floor=0.1)
        assert out == 1.0

    def test_ignores_beams_outside_the_sector(self):
        # 正侧方 0.2m 的墙不是障碍——只有行进扇区内的才算。
        out = dd.sector_min_range(**self._scan([0.2, 3.0, 3.0, 3.0, 0.2]),
                                  half_sector_rad=math.radians(20), range_min=0.05,
                                  range_max=10.0, self_floor=0.1)
        assert out == 3.0

    def test_ignores_self_returns_below_the_floor(self):
        out = dd.sector_min_range(**self._scan([9.0, 9.0, 0.05, 9.0, 9.0]),
                                  half_sector_rad=math.pi, range_min=0.01,
                                  range_max=10.0, self_floor=0.25)
        assert out == 9.0

    def test_ignores_inf_and_nan_and_out_of_spec_beams(self):
        out = dd.sector_min_range(**self._scan([float("inf"), float("nan"), 20.0, 0.001, 4.0]),
                                  half_sector_rad=math.pi, range_min=0.05,
                                  range_max=10.0, self_floor=0.1)
        assert out == 4.0

    def test_nothing_valid_reads_as_clear_not_as_an_emergency(self):
        out = dd.sector_min_range(**self._scan([float("inf")] * 5), half_sector_rad=math.pi,
                                  range_min=0.05, range_max=10.0, self_floor=0.1)
        assert out == float("inf")
