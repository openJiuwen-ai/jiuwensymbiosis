# coding: utf-8
from jiuwensymbiosis.adapters.cruzr.geometry import plan_clamp_targets
from jiuwensymbiosis.perception.object_geometry import ObjectGeometry3D


def _box(back_x_mm=450.0):
    # front_x=350, back_x=450 -> depth middle = 400mm
    return ObjectGeometry3D(ok=True, reason="", center_mm=(400.0, 0.0, 600.0),
                          width_mm=300.0, height_mm=200.0, front_x_mm=350.0,
                          top_z_mm=700.0, n_points=5000, back_x_mm=back_x_mm)


def test_clamp_is_inward_of_descend():
    approach, descend, clamp = plan_clamp_targets(_box(), inset_mm=15.0, pre_clear_mm=60.0)
    assert clamp["left"].pos_m[1] < descend["left"].pos_m[1]    # left moved inward (-Y)
    assert clamp["right"].pos_m[1] > descend["right"].pos_m[1]  # right moved inward (+Y)


def test_approach_is_above_descend_by_dz():
    approach, descend, clamp = plan_clamp_targets(_box(), approach_dz_mm=100.0)
    for arm in ("left", "right"):
        assert abs((approach[arm].pos_m[2] - descend[arm].pos_m[2]) - 0.100) < 1e-9
        # approach is straight above descend (same x, y)
        assert approach[arm].pos_m[0] == descend[arm].pos_m[0]
        assert approach[arm].pos_m[1] == descend[arm].pos_m[1]


def test_clamp_y_uses_half_width_minus_inset_in_meters():
    approach, descend, clamp = plan_clamp_targets(_box(), inset_mm=15.0, pre_clear_mm=60.0)
    # cy=0, width=300 -> half=150; left clamp at (150-15)mm = 0.135 m
    assert abs(clamp["left"].pos_m[1] - 0.135) < 1e-6
    assert abs(clamp["right"].pos_m[1] + 0.135) < 1e-6


def test_orientation_is_forward_and_inward():
    approach, descend, clamp = plan_clamp_targets(_box())
    for wp in (approach, descend, clamp):
        for arm in ("left", "right"):
            assert wp[arm].approach == (1.0, 0.0, 0.0)   # end-plane forward +X
            assert wp[arm].paddle == (0.0, 1.0, 0.0)     # paddle face +Y (mirrored hands)


def test_tcp_offset_mirrors_per_arm():
    approach, descend, clamp = plan_clamp_targets(_box(), tcp_offset_m=0.09)
    assert clamp["left"].tcp_offset_local == (-0.09, 0.0, 0.0)
    assert clamp["right"].tcp_offset_local == (0.09, 0.0, 0.0)


def test_clamp_x_is_side_face_depth_middle():
    approach, descend, clamp = plan_clamp_targets(_box(back_x_mm=450.0))
    # depth middle = (front 350 + back 450)/2 = 400mm = 0.400m, NOT front_x
    assert abs(clamp["left"].pos_m[0] - 0.400) < 1e-6
    # clamp at the z-extent MIDDLE (top_z 700 - height 200/2 = 600); == cz here since the box is
    # geometrically consistent, but the plan no longer uses the density-biased median center_mm[2].
    assert abs(clamp["left"].pos_m[2] - 0.600) < 1e-6


def test_clamp_z_uses_extent_middle_not_top_biased_median():
    # A downward camera's cloud median (center_mm[2]) sits near the TOP face; the clamp must grip the
    # z-extent MIDDLE instead, or the paddles grasp too high. Box: top 800, height 200 -> true middle
    # 700, but the perceived median is 780 (top-biased). Clamp must be 0.700, NOT 0.780.
    box = ObjectGeometry3D(ok=True, reason="", center_mm=(400.0, 0.0, 780.0),
                           width_mm=300.0, height_mm=200.0, front_x_mm=350.0,
                           top_z_mm=800.0, n_points=5000, back_x_mm=450.0)
    approach, descend, clamp = plan_clamp_targets(box)
    for arm in ("left", "right"):
        assert abs(clamp[arm].pos_m[2] - 0.700) < 1e-6      # z-extent middle, not median 0.780
        assert abs(descend[arm].pos_m[2] - 0.700) < 1e-6


def test_clamp_x_falls_back_to_front_when_no_back_face():
    approach, descend, clamp = plan_clamp_targets(_box(back_x_mm=0.0))  # back unseen
    assert abs(clamp["left"].pos_m[0] - 0.350) < 1e-6   # front_x
