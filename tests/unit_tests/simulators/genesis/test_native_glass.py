"""Native glass must keep source geometry and render both sides of its shell."""

import pytest

pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, UsdShade
from test_material_repair import converted, fixture

from jiuwensymbiosis.simulators.genesis.material_repair import prepare_materials
from jiuwensymbiosis.simulators.genesis.material_rules.interioragent import RULES


def test_glass_conversion_requests_native_shell_rendering(tmp_path):
    source = fixture(tmp_path, family="OmniGlass")
    path, _ = prepare_materials(source, tmp_path / "materials", rules=RULES)
    stage, shader = converted(path)
    assert shader.GetPrim().GetCustomDataByKey("symbiosis:nativeGlass") is True


def test_partition_preserves_geometry_and_source_and_leaves_other_materials_alone(tmp_path):
    from jiuwensymbiosis.simulators.genesis.native_glass import prepare_glass_layers

    source = fixture(tmp_path, family="OmniGlass")
    stage = Usd.Stage.Open(str(source))
    mat = UsdShade.Material(stage.GetPrimAtPath("/World/Material"))
    for name in ["Glass", "Wall"]:
        mesh = UsdGeom.Mesh.Define(stage, "/World/" + name)
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        if name == "Glass":
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    stage.GetRootLayer().Save()
    original = source.read_bytes()
    prepared, _ = prepare_materials(source, tmp_path / "materials", rules=RULES)
    base, groups, audit = prepare_glass_layers(prepared, tmp_path / "native")
    assert len(groups) == 1 and audit["mesh_count"] == 1
    body = Usd.Stage.Open(str(base))
    glass = Usd.Stage.Open(str(groups[0]["path"]))
    assert UsdGeom.Imageable(body.GetPrimAtPath("/World/Glass")).ComputeVisibility() == "invisible"
    assert UsdGeom.Imageable(body.GetPrimAtPath("/World/Wall")).ComputeVisibility() == "inherited"
    assert UsdGeom.Imageable(glass.GetPrimAtPath("/World/Glass")).ComputeVisibility() == "inherited"
    assert UsdGeom.Imageable(glass.GetPrimAtPath("/World/Wall")).ComputeVisibility() == "invisible"
    assert (
        UsdGeom.Mesh(glass.GetPrimAtPath("/World/Glass")).GetPointsAttr().Get()
        == UsdGeom.Mesh(stage.GetPrimAtPath("/World/Glass")).GetPointsAttr().Get()
    )
    assert source.read_bytes() == original


def test_native_glass_sets_transmission_and_double_sided_flags(tmp_path, monkeypatch):
    import sys
    from enum import IntFlag
    from types import SimpleNamespace

    flags = IntFlag(
        "Properties",
        {
            name: 1 << i
            for i, name in enumerate(
                [
                    "Opacity",
                    "Sidedness",
                    "NormalMode",
                    "AlbedoColor",
                    "Transmission",
                    "IOR",
                    "Thickness",
                    "PerceptualRoughness",
                    "Metalness",
                ]
            )
        },
    )

    class Material:
        def set_properties(self, value):
            self.flags = value

        def has_properties(self, value):
            return bool(self.flags & value)

    class Scene:
        def __init__(self):
            self.items = []

        def instance_size(self):
            return len(self.items)

        def instance_resize(self, count):
            self.items.extend(SimpleNamespace() for _ in range(count - len(self.items)))

        def get_instance(self, i):
            return self.items[i]

        def set_instance(self, i, value):
            self.items[i] = value

    nps = SimpleNamespace(
        SceneAsset=Scene,
        MaterialAsset=Material,
        EMaterialProperty=flags,
        EInstanceType=SimpleNamespace(Mesh=1),
        EMaterialOpacity=SimpleNamespace(Transmissive=2),
        EMaterialSidedness=SimpleNamespace(Double=1),
        ENormalMode=SimpleNamespace(Flip=0),
        generate_uuid=lambda: 1,
        float3=lambda *v: v,
        float4=lambda *v: v,
        quaternion=lambda *v: v,
    )
    monkeypatch.setitem(sys.modules, "gs_nyx", SimpleNamespace(nyx_py_sdk=nps))
    from jiuwensymbiosis.simulators.genesis.native_glass import append_glass_instances

    asset = nps.SceneAsset()
    append_glass_instances(
        asset, [{"path": tmp_path / "glass.usda", "color": (1, 0.8, 0.6), "roughness": 0.2, "ior": 1.45}]
    )
    material = asset.get_instance(0).matOverride
    assert material.has_properties(nps.EMaterialProperty.Sidedness)
    assert material.has_properties(nps.EMaterialProperty.Opacity)
    assert material.has_properties(nps.EMaterialProperty.NormalMode)
    assert material.sidedness == nps.EMaterialSidedness.Double
    assert material.normalMode == nps.ENormalMode.Flip
    assert material.opacity == nps.EMaterialOpacity.Transmissive
    assert material.transmission == 1
    assert material.ior == pytest.approx(1.45)
    assert material.perceptualRoughness == pytest.approx(0.2)


def test_no_glass_does_not_create_layers(tmp_path):
    from jiuwensymbiosis.simulators.genesis.native_glass import prepare_glass_layers

    source = fixture(tmp_path, family="OmniPBR")
    path, groups, audit = prepare_glass_layers(source, tmp_path / "native")
    assert path == source and not groups and audit["mesh_count"] == 0
    assert not (tmp_path / "native").exists()


@pytest.mark.parametrize("rules", [None, {}])
def test_common_glass_converts_without_asset_rules(tmp_path, rules):
    source = fixture(
        tmp_path,
        family="OmniGlass",
        glass_ior=("Float", 1.4),
        frosting_roughness=("Float", 0.3),
        glass_color=("Color3f", (0.9, 0.8, 0.7)),
    )
    original = source.read_bytes()
    path, report = prepare_materials(source, tmp_path / "materials", rules=rules)
    stage, shader = converted(path)
    assert report["converted"] == 1 and not report["remaining"]
    assert shader.GetPrim().GetCustomDataByKey("symbiosis:nativeGlass") is True
    assert shader.GetInput("ior").Get() == pytest.approx(1.4)
    assert shader.GetInput("roughness").Get() == pytest.approx(0.3)
    assert tuple(shader.GetInput("diffuseColor").Get()) == pytest.approx((0.9, 0.8, 0.7))
    assert source.read_bytes() == original


def test_common_glass_does_not_require_or_duplicate_vendor_rule():
    from jiuwensymbiosis.simulators.genesis.material_rules.common import RULES as COMMON_RULES

    assert "OmniGlass" in COMMON_RULES
    assert "OmniGlass" not in RULES


@pytest.mark.parametrize("family", ["OmniPBR", "UsdPreviewSurface"])
def test_common_rules_do_not_treat_arbitrary_transparency_as_glass(tmp_path, family):
    source = fixture(tmp_path, family=family, opacity=("Float", 0.15))
    original = source.read_bytes()
    path, report = prepare_materials(source, tmp_path / "materials")
    assert path == source and report["converted"] == 0
    assert source.read_bytes() == original
    stage, shader = converted(source)
    assert not shader.GetPrim().GetCustomDataByKey("symbiosis:nativeGlass")
