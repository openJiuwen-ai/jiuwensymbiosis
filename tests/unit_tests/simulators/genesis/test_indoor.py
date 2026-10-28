"""Indoor placement uses actual supported surfaces and USD world coordinates."""

import numpy as np
import pytest

from jiuwensymbiosis.simulators.genesis.camera import PreviewSettings
from jiuwensymbiosis.simulators.genesis.indoor import RoomMesh, read_room_meshes, supported_spawn


def box(size, position, name):
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box(extents=size)
    mesh.apply_translation(position)
    return RoomMesh(name, mesh)


def test_spawn_on_raised_floor_avoids_furniture():
    floor = box((4, 4, 0.14), (0, 0, 0.07), "floor")
    furniture = box((0.8, 0.8, 1.5), (0, 0, 0.89), "cupboard")
    x, y, z = supported_spawn([floor, furniture])
    assert z == pytest.approx(0.14)
    assert np.linalg.norm(np.maximum(np.abs([x, y]) - 0.4, 0)) >= 0.42 - 1e-8


def test_no_invented_floor_in_unsupported_or_crowded_room():
    with pytest.raises(ValueError, match="出生位置"):
        supported_spawn([box((0.5, 0.5, 0.1), (0, 0, 0.05), "too_small")])
    with pytest.raises(ValueError, match="出生位置"):
        supported_spawn([box((3, 3, 0.1), (0, 0, 0.05), "floor"), box((3, 3, 2), (0, 0, 1.1), "blocked")])


def test_collision_mesh_bakes_y_up_centimeters_and_parent_transform(tmp_path):
    pytest.importorskip("pxr")
    pytest.importorskip("trimesh")
    from pxr import Usd, UsdGeom

    path = tmp_path / "room.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, "Y")
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    parent = UsdGeom.Xform.Define(stage, "/Room")
    parent.AddTranslateOp().Set((100, 200, 300))
    mesh = UsdGeom.Mesh.Define(stage, "/Room/Floor")
    mesh.CreatePointsAttr([(0, 0, 0), (100, 0, 0), (0, 0, 100)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().Save()
    original = path.read_bytes()
    result = read_room_meshes(path)
    assert len(result) == 1
    np.testing.assert_allclose(result[0].mesh.bounds, [[1, -4, 2], [2, -3, 2]])
    assert path.read_bytes() == original


def test_composite_requires_background_and_mobile_mode():
    with pytest.raises(ValueError):
        PreviewSettings(robot_id="library://robots/moz1")
    with pytest.raises(ValueError):
        PreviewSettings(scene_id="library://rooms/home", robot_id="library://robots/moz1")
    assert PreviewSettings(
        scene_id="library://rooms/home", robot_id="library://robots/moz1", robot_mode="moz1_wheels"
    ).robot_id
