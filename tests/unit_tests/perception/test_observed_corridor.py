"""Depth guard exercises obstacles, support gaps and body-frame floor offsets."""

import numpy as np
import pytest

from jiuwensymbiosis.perception.frame import CameraFrame, check_observed_corridor, check_observed_rotation


def frame(ground=-0.1):
    t = np.diag([1.0, -1.0, -1.0, 1.0])
    t[2, 3] = 1000.0
    return CameraFrame(
        np.zeros((180, 140, 3), np.uint8),
        np.full((180, 140), 1 - ground),
        np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 90.0], [0.0, 0.0, 1.0]]),
        t,
    )


@pytest.mark.parametrize("height", [0.0, -0.1, -0.25])
def test_level_floor_is_measured_in_each_bodys_base_frame(height):
    assert check_observed_corridor(frame(height), 0.25, 0.55, 0.15) is None


def test_obstacle_and_unobserved_floor_are_distinct_failures():
    f = frame()
    f.depth_m[85:96, 90:105] = 0.8
    assert check_observed_corridor(f, 0.25, 0.55, 0.15) == "depth_obstacle_blocked"
    f = frame()
    f.depth_m[50:130, 45:95] = np.nan
    assert check_observed_corridor(f, 0.25, 0.55, 0.15) == "ground_coverage_incomplete"
    f.depth_m[:] = 2.0
    assert check_observed_corridor(f, 0.25, 0.55, 0.15) == "ground_unobserved"


def test_rectangle_corner_outside_swept_capsule_is_not_an_obstacle():
    f = frame()
    # Elevated points at x=.792, y=-.696: inside the old rectangular check,
    # but 0.882 m from the next base centre, outside its 0.7 m clearance radius.
    f.depth_m[177:180, 99:102] = 0.8
    assert check_observed_corridor(f, 0.25, 0.55, 0.15) is None


BODY = np.array([[-0.4, -0.3, 0.0], [0.4, 0.3, 2.0]])


def overhead_frame():
    """Analytic free-space observation; not a claim about the mounted camera."""
    t = np.diag([1.0, -1.0, -1.0, 1.0])
    t[2, 3] = 4000.0
    return CameraFrame(
        np.zeros((256, 256, 3), np.uint8),
        np.full((256, 256), 5.0),
        np.array([[120.0, 0.0, 128.0], [0.0, 120.0, 128.0], [0.0, 0.0, 1.0]]),
        t,
    )


@pytest.mark.parametrize("turn", [-0.35, 0.001, 0.35, 3.14])
def test_complete_measured_free_space_can_certify_rotation(turn):
    assert check_observed_rotation(overhead_frame(), turn, BODY, 0.15) is None


@pytest.mark.parametrize("missing", ["rear", "upper_body", "depth_hole", "occlusion", "sparse"])
def test_rotation_needs_the_whole_body_volume_observed(missing):
    f = overhead_frame()
    if missing == "rear":
        f.tf_base_cam[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
        f.tf_base_cam[:3, 3] = [30, 0, 1700]
    elif missing == "upper_body":
        f.tf_base_cam[2, 3] = 1500  # floor is visible, but the body extends behind the camera
    elif missing == "depth_hole":
        f.depth_m[128, 128] = np.nan
    elif missing == "occlusion":
        f.depth_m[128, 128] = 1.0
    else:
        f.depth_m[:] = np.nan
        f.depth_m[::8, ::8] = 5.0
    assert check_observed_rotation(f, 0.2, BODY, 0.15) == "turn_clearance_unobserved"


def test_zero_turn_needs_no_extra_observation_but_geometry_cannot_be_guessed():
    f = CameraFrame(np.zeros((4, 4, 3)))
    assert check_observed_rotation(f, 0.0, None, 0.15) is None
    assert check_observed_rotation(f, 0.2, None, 0.15) == "body_geometry_unavailable"
    assert check_observed_rotation(f, 0.2, BODY, 0.15) == "turn_clearance_unobserved"


@pytest.mark.parametrize("bounds", [np.zeros((2, 3)), np.zeros((2, 2)), [[0, 0, 0], [1, 1, np.nan]]])
def test_invalid_body_geometry_is_not_a_small_zero_sized_robot(bounds):
    assert check_observed_rotation(overhead_frame(), 0.2, bounds, 0.15) == "body_geometry_unavailable"
