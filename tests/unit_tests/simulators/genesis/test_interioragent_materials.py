"""Real USD compositions exercise material preparation."""

from pathlib import Path

import pytest

pytest.importorskip("pxr")
from pxr import Sdf, Usd, UsdGeom, UsdShade

# Sibling import by module name, NOT by package path: this directory has no
# __init__.py, so the package-path form would load a SECOND instance of
# test_material_repair next to the one pytest already collected.
from test_material_repair import converted, fixture

from jiuwensymbiosis.simulators.genesis.material_repair import prepare_materials
from jiuwensymbiosis.simulators.genesis.material_rules.interioragent import RULES


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
    path, report = prepare_materials(source, tmp_path / "cache", rules=RULES)
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
    path, report = prepare_materials(wrapper, tmp_path / "cache" / "render", rules=RULES)
    stage, shader = converted(path)
    connection = shader.GetInput("diffuseColor").GetConnectedSource()
    texture = UsdShade.Shader(connection[0].GetPrim())
    assert texture.GetShaderId() == "UsdUVTexture"
    assert Path(texture.GetInput("file").Get().resolvedPath).is_file()
    assert report["baked_textures"][0]["source"] == str(tmp_path / "wood.png")
    assert texture.GetInput("sourceColorSpace").Get() == "sRGB"
    assert shader.GetInput("roughness").Get() == pytest.approx(0.4)
    assert report["textures"] == 1


def test_missing_active_texture_is_reported_and_constant_fallback_is_explicit(tmp_path):
    source = fixture(
        tmp_path,
        IsBaseColorTex=("Float", 1),
        BaseColor_Tex=("Asset", Sdf.AssetPath("absent.png")),
        BaseColor_Color=("Float4", (0.2, 0.3, 0.4, 1)),
    )
    path, report = prepare_materials(source, tmp_path / "cache", rules=RULES)
    stage, shader = converted(path)
    assert tuple(shader.GetInput("diffuseColor").Get()) == pytest.approx((0.2, 0.3, 0.4))
    assert report["issues"][0]["kind"] == "missing_texture"


def test_disabled_texture_is_not_loaded(tmp_path):
    source = fixture(tmp_path, IsNormalTex=("Float", 0), Normal_Tex=("Asset", Sdf.AssetPath("absent.png")))
    _, report = prepare_materials(source, tmp_path / "cache", rules=RULES)
    assert not report["issues"]


def test_active_texture_can_use_the_authored_mdl_default(tmp_path):
    from PIL import Image

    source = fixture(tmp_path, IsBaseColorTex=("Float", 1))
    Image.new("RGB", (4, 4), (50, 100, 150)).save(tmp_path / "default.png")
    (tmp_path / "missing.mdl").write_text(
        'uniform texture_2d BaseColor_Tex = texture_2d("default.png",::tex::gamma_srgb)'
    )
    _, report = prepare_materials(source, tmp_path / "cache", rules=RULES)
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
    _, report = prepare_materials(source, tmp_path / "cache", rules=RULES)
    assert report["textures"] == 1
    assert report["issues"][0]["kind"] == "repaired_material_module_path"


def test_missing_procedural_texture_is_not_counted_as_success(tmp_path):
    source = fixture(tmp_path, family="WorldGridMaterial")
    path, report = prepare_materials(source, tmp_path / "cache", rules=RULES)
    assert path == source and report["converted"] == 0
    assert report["remaining"][0]["reason"] == "conversion_unavailable"
