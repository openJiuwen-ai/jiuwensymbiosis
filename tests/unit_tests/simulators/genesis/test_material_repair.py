"""Real USD compositions exercise the Nyx material compatibility boundary."""

from pathlib import Path

import pytest

pytest.importorskip("pxr")
from pxr import Sdf, Usd, UsdGeom, UsdShade

from jiuwensymbiosis.simulators.genesis.material_repair import prepare_materials


def fixture(tmp_path, family="M_BaseMaterial", **inputs):
    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    material = UsdShade.Material.Define(stage, "/World/Material")
    shader = UsdShade.Shader.Define(stage, "/World/Material/Original")
    shader.SetSourceAsset(Sdf.AssetPath("missing.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier(family, "mdl")
    for name, (kind, value) in inputs.items():
        shader.CreateInput(name, getattr(Sdf.ValueTypeNames, kind)).Set(value)
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "out")
    stage.GetRootLayer().Save()
    return source


def converted(path):
    stage = Usd.Stage.Open(str(path))
    material = UsdShade.Material(stage.GetPrimAtPath("/World/Material"))
    shader, _, _ = material.ComputeSurfaceSource()
    return stage, shader


def test_color_gloss_opacity_and_source_are_preserved(tmp_path):
    source = fixture(
        tmp_path,
        family="M_TranslucentMaterial",
        BaseColor_Color=("Float4", (0.8, 0.1, 0.2, 1)),
        Gloss_Color=("Float4", (0.7, 0.7, 0.7, 1)),
        Metallic_Color=("Float4", (0.3, 0.3, 0.3, 1)),
        Opacity=("Float", 0.0),
    )
    original = source.read_bytes()
    path, report = prepare_materials(source, tmp_path / "cache")
    stage, shader = converted(path)
    assert shader.GetShaderId() == "UsdPreviewSurface"
    assert tuple(shader.GetInput("diffuseColor").Get()) == pytest.approx((0.8, 0.1, 0.2))
    assert shader.GetInput("roughness").Get() == pytest.approx(0.3)
    assert shader.GetInput("metallic").Get() == pytest.approx(0.3)
    assert shader.GetInput("opacity").Get() == 0
    assert source.read_bytes() == original
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1
    assert UsdGeom.GetStageUpAxis(stage) == "Z"
    assert stage.GetDefaultPrim().GetPath() == "/World"
    assert report["converted"] == 1 and report["remaining"] == []


def test_v2_texture_relative_to_authoring_layer_and_color_space(tmp_path):
    from PIL import Image

    Image.new("RGB", (4, 4), (150, 70, 20)).save(tmp_path / "wood.png")
    source = fixture(
        tmp_path,
        family="M_BaseMaterial_v2",
        ColorIsTexType=("Float", 1),
        ColorTex=("Asset", Sdf.AssetPath("wood.png")),
        ColorUVW=("Float4", (2, 3, 0.25, 1)),
        ColorOffset=("Float4", (0.1, 0.2, 0, 1)),
        RoughnessFloat=("Float", 0.4),
    )
    # Wrap the material in another layer: relative paths must retain their owner.
    wrapper = tmp_path / "cache" / "wrapper.usda"
    wrapper.parent.mkdir()
    stage = Usd.Stage.CreateNew(str(wrapper))
    stage.GetRootLayer().subLayerPaths = [str(source)]
    stage.GetRootLayer().Save()
    path, report = prepare_materials(wrapper, tmp_path / "cache" / "render")
    stage, shader = converted(path)
    connection = shader.GetInput("diffuseColor").GetConnectedSource()
    texture = UsdShade.Shader(connection[0].GetPrim())
    assert texture.GetShaderId() == "UsdUVTexture"
    assert Path(texture.GetInput("file").Get().resolvedPath).is_file()
    assert report["baked_textures"][0]["source"] == str(tmp_path / "wood.png")
    assert texture.GetInput("sourceColorSpace").Get() == "sRGB"
    assert shader.GetInput("roughness").Get() == pytest.approx(0.4)
    assert report["textures"] == 1


def test_supported_shader_is_not_rewritten_or_copied(tmp_path):
    source = fixture(tmp_path)
    stage = Usd.Stage.Open(str(source))
    shader = UsdShade.Shader(stage.GetPrimAtPath("/World/Material/Original"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.SetShaderId("UsdPreviewSurface")
    stage.GetRootLayer().Save()
    original = source.read_bytes()
    path, report = prepare_materials(source, tmp_path / "cache")
    assert path == source
    assert report["converted"] == 0
    assert source.read_bytes() == original
    assert not (tmp_path / "cache").exists()


def test_unknown_shader_is_reported_instead_of_claimed_repaired(tmp_path):
    source = fixture(tmp_path, family="UnknownCustomMaterial")
    path, report = prepare_materials(source, tmp_path / "cache")
    assert path == source
    assert report["converted"] == 0
    assert report["remaining"][0]["family"] == "UnknownCustomMaterial"


def test_missing_active_texture_is_reported_and_constant_fallback_is_explicit(tmp_path):
    source = fixture(
        tmp_path,
        IsBaseColorTex=("Float", 1),
        BaseColor_Tex=("Asset", Sdf.AssetPath("absent.png")),
        BaseColor_Color=("Float4", (0.2, 0.3, 0.4, 1)),
    )
    path, report = prepare_materials(source, tmp_path / "cache")
    stage, shader = converted(path)
    assert tuple(shader.GetInput("diffuseColor").Get()) == pytest.approx((0.2, 0.3, 0.4))
    assert report["issues"][0]["kind"] == "missing_texture"


def test_disabled_texture_is_not_loaded(tmp_path):
    source = fixture(tmp_path, IsNormalTex=("Float", 0), Normal_Tex=("Asset", Sdf.AssetPath("absent.png")))
    _, report = prepare_materials(source, tmp_path / "cache")
    assert not report["issues"]


def test_active_texture_can_use_the_authored_mdl_default(tmp_path):
    from PIL import Image

    source = fixture(tmp_path, IsBaseColorTex=("Float", 1))
    Image.new("RGB", (4, 4), (50, 100, 150)).save(tmp_path / "default.png")
    (tmp_path / "missing.mdl").write_text(
        'uniform texture_2d BaseColor_Tex = texture_2d("default.png",::tex::gamma_srgb)'
    )
    _, report = prepare_materials(source, tmp_path / "cache")
    assert report["textures"] == 1
    assert not report["issues"]


def test_malformed_module_path_is_repaired_at_its_authoring_layer(tmp_path):
    from PIL import Image

    meshes = tmp_path / "Meshes"
    meshes.mkdir()
    materials = tmp_path / "Materials"
    materials.mkdir()
    (materials / "template.mdl").write_text(
        'uniform texture_2d BaseColor_Tex = texture_2d("wood.png",::tex::gamma_srgb)'
    )
    Image.new("RGB", (4, 4), (150, 80, 20)).save(materials / "wood.png")
    source = fixture(meshes, IsBaseColorTex=("Float", 1))
    stage = Usd.Stage.Open(str(source))
    shader = UsdShade.Shader(stage.GetPrimAtPath("/World/Material/Original"))
    shader.SetSourceAsset(Sdf.AssetPath(".../Materials/template.mdl"), "mdl")
    stage.GetRootLayer().Save()
    _, report = prepare_materials(source, tmp_path / "cache")
    assert report["textures"] == 1
    assert report["issues"][0]["kind"] == "repaired_material_module_path"


def test_missing_procedural_texture_is_not_counted_as_success(tmp_path):
    source = fixture(tmp_path, family="WorldGridMaterial")
    path, report = prepare_materials(source, tmp_path / "cache")
    assert path == source and report["converted"] == 0
    assert report["remaining"][0]["reason"] == "missing_template_texture"


def test_texture_bake_applies_rotation_and_linear_gloss_inversion(tmp_path):
    import numpy as np
    from PIL import Image

    from jiuwensymbiosis.simulators.genesis.material_baking import bake_texture

    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[:, :4, 0] = 255
    pixels[:, 4:, 1] = 255
    source = tmp_path / "stripes.png"
    Image.fromarray(pixels).save(source)
    path, color, report = bake_texture(
        source, tmp_path / "baked", "diffuseColor", ((1, 1), 90, (1, 0)), True, None, None
    )
    result = np.asarray(Image.open(path))
    assert result[1, 4, 0] > 250 and result[6, 4, 1] > 250
    assert color and report["periodic_exact"]
    Image.new("RGB", (4, 4), (128, 128, 128)).save(source)
    path, color, _ = bake_texture(source, tmp_path / "baked", "roughness", None, True, (-1, -1, -1, 1), (1, 1, 1, 0))
    assert not color
    assert np.asarray(Image.open(path))[0, 0, 0] == pytest.approx(200, abs=1)


def test_hdr_scalar_texture_is_decoded_without_pillow(tmp_path):
    import numpy as np
    from PIL import Image

    from jiuwensymbiosis.simulators.genesis.material_baking import bake_texture

    cv2 = pytest.importorskip("cv2")
    source = tmp_path / "roughness.hdr"
    assert cv2.imwrite(str(source), np.full((4, 4, 3), 0.25, dtype=np.float32))
    path, color, _ = bake_texture(source, tmp_path / "cache", "roughness", None, False, None, None)
    assert not color
    assert np.asarray(Image.open(path))[0, 0, 0] == pytest.approx(64, abs=1)


def test_instanced_material_can_be_overridden_without_changing_source(tmp_path):
    source = fixture(tmp_path, BaseColor_Color=("Float4", (0.8, 0.1, 0.2, 1)))
    original = source.read_bytes()
    wrapper = tmp_path / "instance.usda"
    stage = Usd.Stage.CreateNew(str(wrapper))
    prim = stage.DefinePrim("/Instance", "Xform")
    prim.GetReferences().AddReference(str(source), "/World")
    prim.SetInstanceable(True)
    stage.SetDefaultPrim(prim)
    stage.GetRootLayer().Save()
    path, report = prepare_materials(wrapper, tmp_path / "baked")
    assert report["converted"] == 1
    repaired = Usd.Stage.Open(str(path))
    assert not repaired.GetPrimAtPath("/Instance").IsInstance()
    material = UsdShade.Material(repaired.GetPrimAtPath("/Instance/Material"))
    assert material.ComputeSurfaceSource()[0].GetShaderId() == "UsdPreviewSurface"
    assert source.read_bytes() == original
