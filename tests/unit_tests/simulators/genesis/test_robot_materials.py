"""Source-material fidelity across the USD → baked mesh → Nyx companion boundary."""

import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from jiuwensymbiosis.simulators.genesis.robot_materials import (
    material_parameters,
    rendering_urdf,
    restore_materials,
)


def test_interface_inputs_and_mdl_pbr_keep_linear_values():
    pytest.importorskip("pxr", reason="Install the simulator-scenes extra for USD material tests")
    from pxr import Sdf, Usd, UsdShade

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/mat")
    shader = UsdShade.Shader.Define(stage, "/mat/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    material.CreateInput("color", Sdf.ValueTypeNames.Color3f).Set((0.018, 0.022, 0.028))
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        material.ConnectableAPI(), "color", UsdShade.AttributeType.Input
    )
    result = material_parameters(material)
    assert result["baseColorFactor"] == pytest.approx([0.018, 0.022, 0.028, 1])
    assert result["roughnessFactor"] == 0.5
    shader.CreateIdAttr("")
    shader.SetSourceAsset(Sdf.AssetPath("gltf/pbr.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("gltf_material", "mdl")
    shader.CreateInput("base_color_factor", Sdf.ValueTypeNames.Color3f).Set((0.1, 0.2, 0.3))
    shader.CreateInput("metallic_factor", Sdf.ValueTypeNames.Float).Set(0.9)
    shader.CreateInput("emissive_strength", Sdf.ValueTypeNames.Float).Set(10000)
    result = material_parameters(material)
    assert result["baseColorFactor"] == pytest.approx([0.1, 0.2, 0.3, 1])
    assert result["metallicFactor"] == pytest.approx(0.9)
    assert result["emissiveFactor"] == [0, 0, 0]


def test_companion_preserves_baked_vertices_face_partitions_and_physics(tmp_path):
    pytest.importorskip("pxr", reason="Install the simulator-scenes extra for USD material tests")
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    source = tmp_path / "robot.usda"
    stage = Usd.Stage.CreateNew(str(source))
    mesh = UsdGeom.Mesh.Define(stage, "/robot/mesh")
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * 4)
    mesh.CreateFaceVertexIndicesAttr(faces.ravel())
    mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 12)
    mesh.SetNormalsInterpolation("faceVarying")
    material = UsdShade.Material.Define(stage, "/robot/red")
    shader = UsdShade.Shader.Define(stage, "/robot/red/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.4, 0.01, 0.02))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.8)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    subset = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).CreateMaterialBindSubset("metal", [1, 3])
    UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    before = source.read_bytes()
    # Known bake: rotate around X and translate. Source normals must follow it.
    rotation = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    baked = points @ rotation + (2, 3, 4)
    package = tmp_path / "package"
    package.mkdir()
    obj = package / "mesh.obj"
    obj.write_text("# Exported from USD: /robot/mesh\n" + "".join(f"v {x} {y} {z}\n" for x, y, z in baked))
    urdf = package / "robot.urdf"
    urdf.write_text(
        '<robot name="r"><link name="base"><visual><geometry><mesh filename="mesh.obj"/></geometry></visual><collision><geometry><mesh filename="mesh.obj"/></geometry></collision></link></robot>'
    )
    physical = urdf.read_bytes()
    report = restore_materials(source, urdf, package / "materials")
    assert report["texture_images"] == 0
    assert report["meshes"][0]["subsets"] == 1
    assert source.read_bytes() == before
    assert urdf.read_bytes() == physical
    rendered = rendering_urdf(urdf)
    assert rendered is not None
    path = ET.parse(rendered).find(".//visual/geometry/mesh").get("filename")
    glb = (rendered.parent / path).read_bytes()
    size = struct.unpack_from("<I", glb, 12)[0]
    data = json.loads(glb[20 : 20 + size])
    binary = glb[28 + size :]
    assert len(data["meshes"][0]["primitives"]) == 2
    p = data["bufferViews"][0]
    actual = np.frombuffer(binary[p["byteOffset"] : p["byteOffset"] + p["byteLength"]], dtype="<f4").reshape(-1, 3)
    np.testing.assert_allclose(actual, baked[faces.ravel()])
    p = data["bufferViews"][1]
    actual = np.frombuffer(binary[p["byteOffset"] : p["byteOffset"] + p["byteLength"]], dtype="<f4").reshape(-1, 3)
    np.testing.assert_allclose(actual, np.tile([0, -1, 0], (12, 1)), atol=1e-6)
    assert data["materials"][1]["pbrMetallicRoughness"]["baseColorFactor"] == pytest.approx([0.4, 0.01, 0.02, 1])
    assert path.endswith("mesh.glb")
    relocated = package.rename(tmp_path / "relocated")
    urdf = relocated / urdf.name
    rendered = rendering_urdf(urdf)
    for mesh_ref in ET.parse(rendered).findall(".//mesh"):
        filename = Path(mesh_ref.attrib["filename"])
        assert not filename.is_absolute()
        assert (rendered.parent / filename).is_file()
    collision = ET.parse(rendered).find(".//collision/geometry/mesh").get("filename")
    assert (rendered.parent / collision).read_bytes() == (relocated / "mesh.obj").read_bytes()
    urdf.write_text(physical.decode() + "\n")
    with pytest.raises(ValueError, match="different URDF"):
        rendering_urdf(urdf)


def test_texture_network_is_explicitly_rejected():
    pytest.importorskip("pxr", reason="Install the simulator-scenes extra for USD material tests")
    from pxr import Sdf, Usd, UsdShade

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/mat")
    shader = UsdShade.Shader.Define(stage, "/mat/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    texture = UsdShade.Shader.Define(stage, "/mat/texture")
    texture.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    with pytest.raises(ValueError, match="Texture/network"):
        material_parameters(material)


def test_collision_guide_is_hidden_only_in_render_companion(tmp_path):
    pytest.importorskip("pxr", reason="Install the simulator-scenes extra for USD material tests")
    from pxr import Usd, UsdGeom, UsdPhysics

    source = tmp_path / "robot.usda"
    stage = Usd.Stage.CreateNew(str(source))
    link = UsdGeom.Xform.Define(stage, "/robot/roller")
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    sphere = UsdGeom.Sphere.Define(stage, "/robot/roller/sphere")
    sphere.CreatePurposeAttr("guide")
    stage.GetRootLayer().Save()
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        '<robot name="r"><link name="roller"><visual name="sphere"><geometry><sphere radius=".08"/></geometry></visual><collision><geometry><sphere radius=".08"/></geometry></collision></link></robot>'
    )
    (tmp_path / "export_manifest.json").write_text(
        json.dumps({"compatibility": {"prim_names": {"/robot/roller": "roller"}}})
    )
    before = urdf.read_bytes()
    result = restore_materials(source, urdf, tmp_path / "materials")
    assert result["hidden_visual_prims"] == ["/robot/roller/sphere"]
    rendered = ET.parse(rendering_urdf(urdf))
    assert rendered.find(".//visual") is None
    assert rendered.find(".//collision/geometry/sphere").get("radius") == ".08"
    assert urdf.read_bytes() == before
