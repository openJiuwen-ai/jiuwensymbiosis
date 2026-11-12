"""SceneGen USD presentation and dependency boundaries, without a GPU."""

import json
import math

import pytest

from jiuwensymbiosis.simulators.genesis.source_presentation import dependencies, read_presentation


@pytest.fixture
def source(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdGeom, UsdLux

    path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(path))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    camera = UsdGeom.Camera.Define(stage, "/World/MainCamera")
    camera.GetPrim().CreateAttribute("userProperties:package_camera", Sdf.ValueTypeNames.Bool).Set(True)
    camera.CreateFocalLengthAttr(35)
    camera.CreateHorizontalApertureAttr(36)
    camera.CreateVerticalApertureAttr(20.25)
    camera.CreateClippingRangeAttr((0.05, 200))
    UsdGeom.Xformable(camera).AddTranslateOp().Set((1, 2, 3))
    dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
    dome.CreateColorAttr((0.2, 0.4, 0.6))
    dome.CreateIntensityAttr(0.5)
    stage.GetRootLayer().Save()
    return path


def test_authored_pose_intrinsics_and_constant_light(source, tmp_path):
    from jiuwensymbiosis.simulators.genesis.source_lighting import read_source_lighting

    p = read_presentation(source, root=tmp_path)
    assert p["pos"] == pytest.approx([1, 2, 3])
    assert p["lookat"] == pytest.approx([1, 2, 2])
    assert p["up"] == pytest.approx([0, 1, 0])
    assert p["fov_y"] == pytest.approx(math.degrees(2 * math.atan(20.25 / 70)))
    assert p["near"] == pytest.approx(0.05)
    assert p["far"] == 200
    rig = read_source_lighting(source, exposure_ev=0, directory=tmp_path / "runtime")
    assert rig.multiplier == 0.5
    assert rig.tint == pytest.approx((0.2, 0.4, 0.6))
    assert rig.rectangles == ()
    assert rig.hdr.endswith("constant-environment.hdr")


def test_ambiguous_or_offset_camera_rejected(source):
    from pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(source))
    camera = UsdGeom.Camera.Define(stage, "/World/Other")
    camera.GetPrim().CreateAttribute("userProperties:package_camera", Sdf.ValueTypeNames.Bool).Set(True)
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="ambiguous"):
        read_presentation(source)
    stage.RemovePrim("/World/Other")
    UsdGeom.Camera(stage.GetPrimAtPath("/World/MainCamera")).CreateHorizontalApertureOffsetAttr(2)
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="offset"):
        read_presentation(source)


def test_missing_reference_and_root_escape_fail(source, tmp_path):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(source))
    UsdGeom.Xform.Define(stage, "/World/Reference").GetPrim().GetReferences().AddReference("missing.usda")
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="Missing USD dependencies"):
        dependencies(source)
    stage.RemovePrim("/World/Reference")
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="outside asset root"):
        dependencies(source, root=tmp_path / "different-root")


def test_manifest_selects_camera_and_unrelated_usd_does_not_opt_in(source):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(source))
    stage.GetPrimAtPath("/World/MainCamera").RemoveProperty("userProperties:package_camera")
    UsdGeom.Camera.Define(stage, "/World/Overview")
    stage.GetRootLayer().Save()
    assert read_presentation(source) is None
    (source.parent / "manifest.json").write_text(
        json.dumps(
            {"scene_id": "scene_test", "artifact_type": "base", "camera": {"usd_prim_path": "/World/MainCamera"}}
        )
    )
    assert read_presentation(source)["camera_path"] == "/World/MainCamera"


def test_glass_metadata_remains_rejected(source):
    from pxr import Usd, UsdShade

    from jiuwensymbiosis.simulators.genesis.material_repair import require_supported_materials

    stage = Usd.Stage.Open(str(source))
    material = UsdShade.Material.Define(stage, "/World/Glass")
    material.GetPrim().SetCustomDataByKey("scene_gen:optics", json.dumps({"transmission": 1, "ior": 1.5}))
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="optical metadata"):
        require_supported_materials(source, {"remaining": []})
    with pytest.raises(ValueError, match="Unsupported Nyx materials"):
        require_supported_materials(source, {"remaining": [{"family": "Unknown"}]})


def test_missing_texture_is_a_dependency_failure(source):
    from pxr import Sdf, Usd, UsdShade

    stage = Usd.Stage.Open(str(source))
    shader = UsdShade.Shader.Define(stage, "/World/Texture")
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("missing.png"))
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="Missing USD dependencies"):
        dependencies(source)


def test_broken_material_binding_cannot_render_as_default_gray(source):
    from pxr import Usd, UsdGeom, UsdShade

    from jiuwensymbiosis.simulators.genesis.material_repair import require_supported_materials

    stage = Usd.Stage.Open(str(source))
    mesh = UsdGeom.Mesh.Define(stage, "/World/Geometry")
    material = UsdShade.Material.Define(stage, "/World/Material")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="Missing material surface"):
        require_supported_materials(source, {"remaining": []})
    stage.RemovePrim("/World/Material")
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="Invalid material binding"):
        require_supported_materials(source, {"remaining": []})


def test_y_up_centimetre_camera_is_converted(source):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(source))
    UsdGeom.SetStageUpAxis(stage, "Y")
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    stage.GetRootLayer().Save()
    p = read_presentation(source)
    assert p["pos"] == pytest.approx([0.01, -0.03, 0.02])
    assert p["up"] == pytest.approx([0, 0, 1])
    assert p["near"] == pytest.approx(0.0005)
