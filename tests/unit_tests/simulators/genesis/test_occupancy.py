"""Synthetic geometry checks for map semantics, clipping and coordinate orientation."""

import numpy as np
import pytest

from jiuwensymbiosis.simulators.genesis.indoor import RoomMesh
from jiuwensymbiosis.simulators.genesis.occupancy import build_occupancy

# build_occupancy rasterizes through OpenCV, which the CPU test extras do not
# install (opencv lives in the genesis extra only).
pytest.importorskip("cv2")


def box(size, position, name):
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box(extents=size)
    mesh.apply_translation(position)
    return RoomMesh(name, mesh)


def cell(result, x, y):
    column, row = np.floor((np.asarray([x, y]) - result.origin) / result.resolution).astype(int)
    return result.cells[row, column]


def test_support_obstacles_ceiling_and_outside_remain_distinct():
    meshes = [
        box((4, 4, 0.1), (0, 0, -0.05), "floor"),
        box((0.6, 0.6, 3), (1, 0, 1.5), "tall_cabinet"),
        box((4, 4, 0.1), (0, 0, 2.5), "ceiling"),
        box((0.01, 1, 2), (-1, 0, 1), "thin_wall"),
    ]
    rooms = [{"polygon": [[-2, -2], [2, -2], [2, 2], [-2, 2]]}]
    result = build_occupancy(meshes, rooms)
    assert result.floor_z == pytest.approx(0)
    assert cell(result, 0, 0) == 0
    assert cell(result, 1, 0) == 100  # solid interior, despite its top being above the height band
    assert cell(result, -1, 0) == 100
    assert cell(result, 2.2, 0) == -1
    assert set(np.unique(result.cells)) == {-1, 0, 100}


def test_room_polygon_cannot_invent_floor_and_world_y_is_not_flipped():
    meshes = [box((2, 2, 0.1), (0, 1, 0.95), "floor")]
    rooms = [{"polygon": [[-2, -2], [2, -2], [2, 2], [-2, 2]]}]
    result = build_occupancy(meshes, rooms)
    assert result.floor_z == pytest.approx(1)
    assert cell(result, 0, 1) == 0
    assert cell(result, 0, -1) == -1


def test_sloped_triangle_is_clipped_before_projection():
    trimesh = pytest.importorskip("trimesh")
    meshes = [
        box((4, 4, 0.1), (0, 0, -0.05), "floor"),
        RoomMesh("slope", trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 3], [1, 1, 3]], faces=[[0, 1, 2]])),
    ]
    rooms = [{"polygon": [[-2, -2], [2, -2], [2, 2], [-2, 2]]}]
    result = build_occupancy(meshes, rooms, min_height=0.3, max_height=1.5)
    assert cell(result, 0.3, 0.15) == 100
    assert cell(result, 0.8, 0.4) == 0


def test_missing_authored_floor_fails_closed():
    with pytest.raises(ValueError, match="floor"):
        build_occupancy(
            [box((1, 1, 1), (0, 0, 0.5), "table")],
            [{"polygon": [[-2, -2], [2, -2], [2, 2], [-2, 2]]}],
        )


def test_large_underlying_slab_does_not_override_finished_floor():
    meshes = [
        box((4, 4, 0.5), (0, 0, -0.3), "floor_slab"),
        box((3.8, 3.8, 0.01), (0, 0, -0.005), "floor_finish"),
    ]
    rooms = [{"polygon": [[-2, -2], [2, -2], [2, 2], [-2, 2]]}]
    result = build_occupancy(meshes, rooms)
    assert result.floor_z == pytest.approx(0)
    assert cell(result, 0, 0) == 0


def test_free_space_agrees_with_the_runtime_navigation_map():
    meshes = [
        box((4, 4, 0.1), (0, 0, -0.05), "floor"),
        box((0.6, 0.6, 3), (1, 0, 1.5), "tall_cabinet"),
    ]
    rooms = [{"polygon": [[-2, -2], [2, -2], [2, 2], [-2, 2]]}]
    exported = build_occupancy(meshes, rooms)
    from jiuwensymbiosis.simulators.genesis.navigation import NavigationMap

    runtime = NavigationMap.from_meshes(meshes, floor_z=exported.floor_z, radius=0)
    for world_x, world_y in ((-1, -1), (0, 1), (-1.4, 1.4)):
        assert cell(exported, world_x, world_y) == 0
        assert runtime.clear((world_x, world_y))
    # The cabinet's wall footprint blocks both maps. Their interior semantics
    # deliberately differ: the export marks the whole solid as occupied, while
    # the runtime map blocks the band-crossing faces' perimeter (an enclosed
    # interior cell reads free but is unreachable) — free REACHABILITY is what
    # the two must agree on.
    assert cell(exported, 0.7, 0) == 100
    assert not runtime.clear((0.7, 0))
