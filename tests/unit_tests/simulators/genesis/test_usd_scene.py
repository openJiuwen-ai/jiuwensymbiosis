"""Asset boundaries and USD preparation; no Genesis/CUDA runtime required."""

import hashlib

import pytest
import yaml

from jiuwensymbiosis.simulators.assets import build_index, load_config
from jiuwensymbiosis.simulators.assets import index as asset_index
from jiuwensymbiosis.simulators.genesis.scene_catalog import scene_catalog, scene_path
from jiuwensymbiosis.simulators.genesis.usd_scene import initial_camera, inspect_stage, prepare_static_stage


def test_catalog_only_resolves_local_entries(tmp_path, monkeypatch):
    config = tmp_path / "assets.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "index": "index.sqlite",
                "roots": [{"id": "geniesim", "scanner": "geniesim", "path": "assets"}],
            }
        )
    )
    monkeypatch.setattr(asset_index, "DEFAULT_INDEX_PATH", tmp_path / "index.sqlite")
    folder = tmp_path / "assets/background/kitchen/kitchen_1"
    folder.mkdir(parents=True)
    source = folder / "background.usda"
    source.touch()
    build_index(load_config(config))
    config.unlink()
    identifier = "library://geniesim/background/kitchen/kitchen_1/background"
    assert scene_path(identifier) == source
    assert scene_catalog() == [
        {"id": "demo", "label": "动态物理演示"},
        {"id": identifier, "label": "geniesim · kitchen/kitchen_1", "preview_mode": "static_usd"},
    ]
    for invalid in ("../../secret", "/tmp/file", "http://host/scene", "kitchen/missing"):
        with pytest.raises(ValueError):
            scene_path(invalid)
    source.unlink()
    outside = tmp_path / "secret.usda"
    outside.touch()
    source.symlink_to(outside)
    with pytest.raises(ValueError):
        scene_path(identifier)
    # Reading the catalogue does not rescan the disk or silently remove cached entries.
    assert len(scene_catalog()) == 2


@pytest.fixture
def usd(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    UsdGeom.SetStageUpAxis(stage, "Y")
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    mesh = UsdGeom.Mesh.Define(stage, "/World/Table")
    mesh.CreatePointsAttr([(0, 0, 0), (100, 0, 0), (0, 100, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.RigidBodyAPI.Apply(mesh.GetPrim())
    UsdPhysics.FixedJoint.Define(stage, "/World/Joint")
    stage.GetRootLayer().Save()
    return source


def test_static_layer_preserves_source_and_removes_motion(usd, tmp_path):
    from pxr import Usd, UsdPhysics

    before = hashlib.sha256(usd.read_bytes()).hexdigest()
    output = tmp_path / "cache/static.usda"
    report = prepare_static_stage(usd, output)
    assert hashlib.sha256(usd.read_bytes()).hexdigest() == before
    assert report["visible_meshes"] == report["visible_triangles"] == 1
    assert report["source_up_axis"] == "Y"
    assert report["bounds_m"] == [[0, 0, 0], [1, 0, 1]]
    static = Usd.Stage.Open(str(output))
    assert not static.GetPrimAtPath("/World/Table").HasAPI(UsdPhysics.RigidBodyAPI)
    assert not static.GetPrimAtPath("/World/Joint").IsActive()
    original = Usd.Stage.Open(str(usd))
    assert original.GetPrimAtPath("/World/Table").HasAPI(UsdPhysics.RigidBodyAPI)
    assert original.GetPrimAtPath("/World/Joint").IsActive()
    with pytest.raises(ValueError, match="覆盖"):
        prepare_static_stage(usd, usd)


def test_hidden_only_stage_is_rejected(usd, tmp_path):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd))
    UsdGeom.Imageable(stage.GetPrimAtPath("/World/Table")).MakeInvisible()
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="没有可见网格"):
        prepare_static_stage(usd, tmp_path / "cache.usda")


def test_oversized_background_rejected_before_export(usd, tmp_path, monkeypatch):
    from jiuwensymbiosis.simulators.genesis import usd_scene

    report = inspect_stage(usd)
    report["visible_triangles"] = 5_000_001
    monkeypatch.setattr(usd_scene, "inspect_stage", lambda _: report)
    with pytest.raises(ValueError, match="LOD"):
        prepare_static_stage(usd, tmp_path / "cache.usda")
    assert not (tmp_path / "cache.usda").exists()


def test_room_camera_stays_in_scene_bounds_after_pan():
    from jiuwensymbiosis.simulators.genesis.camera import OrbitCamera

    camera = OrbitCamera(**initial_camera("home/test", [[10, -20, -2], [20, -10, 2]]))
    assert camera.target[0] > 10
    panned = camera.command("pan", dx=0.1)
    assert panned.target[0] > 10  # no snap back to the tabletop's [-2,2] limits
    assert panned.distance == camera.distance
