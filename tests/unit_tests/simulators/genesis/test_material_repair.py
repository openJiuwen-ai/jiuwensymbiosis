"""Real USD compositions exercise material preparation."""

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


def test_unknown_shader_is_reported_instead_of_claimed_repaired(tmp_path):
    source = fixture(tmp_path, family="UnknownCustomMaterial")
    path, report = prepare_materials(source, tmp_path / "cache")
    assert path == source
    assert report["converted"] == 0
    assert report["remaining"][0]["family"] == "UnknownCustomMaterial"


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
    path, report = prepare_materials(wrapper, tmp_path / "baked", rules={"M_BaseMaterial": simple_rule})
    assert report["converted"] == 1
    repaired = Usd.Stage.Open(str(path))
    assert not repaired.GetPrimAtPath("/Instance").IsInstance()
    material = UsdShade.Material(repaired.GetPrimAtPath("/Instance/Material"))
    assert material.ComputeSurfaceSource()[0].GetShaderId() == "UsdPreviewSurface"
    assert source.read_bytes() == original


def simple_rule(stage, preview, shader, family, report, directory):
    from jiuwensymbiosis.simulators.genesis.material_surface import constant

    constant(preview, "diffuseColor", (0.2, 0.3, 0.4))
    return True


def test_no_rules_preserves_even_a_known_template(tmp_path):
    source = fixture(tmp_path)
    original = source.read_bytes()
    path, report = prepare_materials(source, tmp_path / "cache")
    assert path == source and source.read_bytes() == original
    assert report["converted"] == 0
    assert report["remaining"] == [{"path": "/World/Material", "family": "M_BaseMaterial"}]
    assert not (tmp_path / "cache").exists()


def test_caller_rule_preserves_metadata_and_records_overlay(tmp_path):
    import json

    source = fixture(tmp_path, family="SyntheticMaterial")
    original = source.read_bytes()
    path, report = prepare_materials(source, tmp_path / "cache", rules={"SyntheticMaterial": simple_rule})
    stage, shader = converted(path)
    assert path != source and source.read_bytes() == original
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1
    assert UsdGeom.GetStageUpAxis(stage) == "Z"
    assert stage.GetDefaultPrim().GetPath() == "/World"
    assert shader.GetShaderId() == "UsdPreviewSurface"
    assert tuple(shader.GetInput("diffuseColor").Get()) == pytest.approx((0.2, 0.3, 0.4))
    assert json.loads((path.parent / "material-repair.json").read_text()) == report


@pytest.mark.parametrize("root_id, expected", [("interioragent", 1), ("geniesim", 0), ("custom", 0)])
def test_rules_follow_indexed_provenance_not_material_name(tmp_path, root_id, expected):
    from types import SimpleNamespace

    from jiuwensymbiosis.simulators.genesis.scene_catalog import material_rules_for_asset

    source = fixture(tmp_path)
    asset = SimpleNamespace(asset=SimpleNamespace(root_id=root_id))
    _, report = prepare_materials(source, tmp_path / "cache", rules=material_rules_for_asset(asset))
    assert report["converted"] == expected
    assert material_rules_for_asset(None) is None
