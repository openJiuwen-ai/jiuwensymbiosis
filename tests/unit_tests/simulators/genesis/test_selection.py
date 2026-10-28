"""Real USD surface attribution and camera framing, without GPU/Genesis imports."""

import math
from itertools import product

import pytest

from jiuwensymbiosis.simulators.genesis.camera import OrbitCamera
from jiuwensymbiosis.simulators.genesis.preview import GenesisPreview
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


@pytest.mark.parametrize("aspect", [16 / 9, 9 / 16])
def test_focus_fits_all_corners_and_sets_orbit_center(aspect):
    box = [[-2, -0.4, 0], [2, 0.4, 1.3]]
    camera = OrbitCamera().focus(box, fov_y=60, aspect=aspect)
    assert camera.target == pytest.approx([0, 0, 0.65])
    original_eye = OrbitCamera().snapshot()["pos"]
    original_direction = [original_eye[i] - camera.target[i] for i in range(3)]
    assert camera.yaw == pytest.approx(math.atan2(original_direction[1], original_direction[0]))
    eye = camera.snapshot()["pos"]
    outward = [
        math.cos(camera.pitch) * math.cos(camera.yaw),
        math.cos(camera.pitch) * math.sin(camera.yaw),
        math.sin(camera.pitch),
    ]
    right = [-math.sin(camera.yaw), math.cos(camera.yaw), 0]
    up = [
        -math.cos(camera.yaw) * math.sin(camera.pitch),
        -math.sin(camera.yaw) * math.sin(camera.pitch),
        math.cos(camera.pitch),
    ]
    for corner in product(*zip(*box, strict=True)):
        relative = [corner[i] - eye[i] for i in range(3)]
        depth = -sum(a * b for a, b in zip(relative, outward, strict=True))
        assert depth > 0
        assert abs(sum(a * b for a, b in zip(relative, right, strict=True))) < depth * math.tan(math.pi / 6) * aspect
        assert abs(sum(a * b for a, b in zip(relative, up, strict=True))) < depth * math.tan(math.pi / 6)
    assert camera.command("orbit", dx=0.2).target == camera.target


def test_small_focused_object_can_zoom_closer_without_jumping_back():
    camera = OrbitCamera().focus([[-0.02] * 3, [0.02] * 3], fov_y=60, aspect=16 / 9)
    assert camera.distance < 0.2
    assert camera.command("zoom", delta=-0.1).distance < camera.distance


def test_pick_rejects_stale_scene_or_camera_before_sending_to_worker(tmp_path, monkeypatch):

    preview = GenesisPreview(state_dir=tmp_path)
    monkeypatch.setattr(preview, "status", lambda: {"state": "ready", "scene": {"instance_id": "a" * 32}})
    values = {
        "action": "pick",
        "u": 0.5,
        "v": 0.5,
        "scene_instance_id": "a" * 32,
        "config_revision": 0,
        "camera_revision": 0,
    }
    for field, value in [("scene_instance_id", "b" * 32), ("config_revision", 1), ("camera_revision", 1)]:
        with pytest.raises(ValueError, match="已经"):
            preview.select(**{**values, field: value})
    assert not list(tmp_path.iterdir())
