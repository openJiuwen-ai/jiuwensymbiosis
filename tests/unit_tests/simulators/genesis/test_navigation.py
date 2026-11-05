"""Reachability, footprint, support and facing: no renderer or GPU required."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.simulators.genesis.navigation import NavigationMap, _paint, _paint_obstacles


def mesh(triangles):
    triangles = np.asarray(triangles, dtype=float)
    return SimpleNamespace(
        mesh=SimpleNamespace(
            triangles=triangles, bounds=np.array([triangles.min(axis=(0, 1)), triangles.max(axis=(0, 1))])
        )
    )


def floor(x0, y0, x1, y1):
    return mesh([[[x0, y0, 0], [x1, y0, 0], [x1, y1, 0]], [[x0, y0, 0], [x1, y1, 0], [x0, y1, 0]]])


def wall(x, low, high):
    return mesh([[[x, low, 0], [x, high, 0], [x, high, 2]], [[x, low, 0], [x, high, 2], [x, low, 2]]])


def test_path_detours_around_wall_and_all_segments_clear():
    grid = NavigationMap.from_meshes([floor(-4, -4, 4, 4), wall(0, -1, 1)], floor_z=0)
    plan = grid.plan([-2, 0], [2, 0, math.pi / 2])
    assert plan["goal"] == [2, 0, math.pi / 2]
    assert plan["path_length_m"] > 4
    previous = [-2, 0]
    for point in plan["waypoints"]:
        assert grid.segment_clear(previous, point)
        assert math.dist(previous, point) <= 0.600001
        previous = point
    assert previous == pytest.approx([2, 0])


def test_solid_wall_and_unsupported_floor_are_unreachable():
    grid = NavigationMap.from_meshes([floor(-3, -3, 3, 3), wall(0, -3, 3)], floor_z=0)
    with pytest.raises(ValueError, match="no_path"):
        grid.plan([-2, 0], [2, 0, 0])
    with pytest.raises(ValueError, match="blocked_or_unsupported"):
        grid.plan([-2, 0], [10, 0, 0])
    grid = NavigationMap.from_meshes([floor(-3, -3, -0.3, 3), floor(0.3, -3, 3, 3)], floor_z=0)
    assert not grid.clear([0, 0])
    with pytest.raises(ValueError, match="no_path"):
        grid.path([-2, 0], [2, 0])


def test_object_goal_stands_outside_footprint_and_faces_center():
    grid = NavigationMap.plane([0, 0])
    # Block the object's interior even though this is a synthetic test map.
    grid.free[90:110, 110:130] = False
    bounds = [[1, -1, 0], [3, 1, 1]]
    plan = grid.plan([-2, 0], bounds=bounds, stand_off_m=0.4)
    x, y, yaw = plan["goal"]
    assert x == pytest.approx(1 - grid.radius - 0.4)
    assert yaw == pytest.approx(math.atan2(-y, 2 - x))
    assert plan["waypoints"][-1] == pytest.approx([x, y])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "0"])
def test_nonfinite_goals_and_invalid_standoff_rejected(bad):
    grid = NavigationMap.plane([0, 0])
    with pytest.raises(ValueError):
        grid.plan([0, 0], [bad, 1, 0])
    with pytest.raises(ValueError):
        grid.plan([0, 0], bounds=[[1, 1, 0], [2, 2, 1]], stand_off_m=bad)


def test_batched_small_faces_never_remove_a_reference_obstacle():
    rng = np.random.default_rng(17)
    triangles = rng.uniform(-1, 5, (2000, 1, 3)) + rng.uniform(-0.03, 0.03, (2000, 3, 3))
    reference = np.zeros((40, 40), dtype=bool)
    batched = reference.copy()
    origin = np.array([0.0, 0.0])
    for triangle in triangles:
        _paint(reference, origin, 0.1, triangle, 0.1 / math.sqrt(2))
    _paint_obstacles(batched, origin, 0.1, triangles)
    assert np.all(batched[reference])
