"""Real USD surface attribution and camera framing, without GPU/Genesis imports."""

import pytest

from jiuwensymbiosis.simulators.genesis.selection import UsdPickIndex


def test_surface_resolution_does_not_select_empty_space_inside_a_box(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom

    source = tmp_path / "objects.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageUpAxis(stage, "Y")
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    model = UsdGeom.Xform.Define(stage, "/World/chair")
    model.GetPrim().SetMetadata("kind", "component")
    model.AddTranslateOp().Set(Gf.Vec3d(100, 200, 300))
    mesh = UsdGeom.Mesh.Define(stage, "/World/chair/mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (100, 0, 0), (0, 100, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateExtentAttr([(0, 0, 0), (100, 100, 0)])
    hidden = UsdGeom.Mesh.Define(stage, "/World/hidden")
    hidden.CreateVisibilityAttr("invisible")
    stage.GetRootLayer().Save()
    before = source.read_bytes()
    index = UsdPickIndex(source)
    hit = index.resolve([1.2, -3, 2.2])
    assert hit["path"] == "/World/chair"
    assert hit["name"] == "chair"
    assert hit["bounds_m"] == [[1, -3, 2], [2, -3, 3]]
    assert hit["surface_error_m"] < 1e-6
    with pytest.raises(ValueError, match="未能对应"):
        index.resolve([1.9, -3, 2.9])  # In the AABB, outside the triangle.
    assert len(index.meshes) == 1
    assert source.read_bytes() == before
