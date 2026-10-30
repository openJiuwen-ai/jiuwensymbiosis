"""Regression coverage for nested-USD export failures."""

import math
import xml.etree.ElementTree as ET

import pytest

from jiuwensymbiosis.simulators.genesis.urdf_compat import (
    finish_export,
    iter_link_geometry,
    prepare_export_stage,
    validate_urdf,
)


def test_nested_link_meshes_are_not_duplicated_and_disabled_joint_is_excluded():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    a = UsdGeom.Xform.Define(stage, "/robot/a").GetPrim()
    b = UsdGeom.Xform.Define(stage, "/robot/a/b").GetPrim()
    for prim in (a, b):
        UsdPhysics.RigidBodyAPI.Apply(prim)
        prim.SetMetadata("displayName", "wheel")
        UsdGeom.Cube.Define(stage, str(prim.GetPath()) + "/mesh")
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/robot/disabled")
    joint.CreateJointEnabledAttr(False)
    before = stage.GetRootLayer().ExportToString()
    result = prepare_export_stage(stage)
    assert stage.GetRootLayer().ExportToString() == before
    assert len(set(result["prim_names"].values())) == 2
    assert not stage.GetPrimAtPath("/robot/disabled").IsActive()
    assert [str(p.GetPath()) for p in iter_link_geometry(a)] == ["/robot/a/mesh"]
    assert [str(p.GetPath()) for p in iter_link_geometry(b)] == ["/robot/a/b/mesh"]


@pytest.mark.parametrize("override", [None, "export:name"])
def test_export_names_use_standard_metadata_and_explicit_custom_attribute(override):
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    for name in ("base", "arm", "tool"):
        prim = UsdGeom.Xform.Define(stage, f"/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        if name != "tool":
            prim.CreateAttribute("export:name", Sdf.ValueTypeNames.String).Set(f"custom_{name}")
        if name == "base":
            prim.SetMetadata("displayName", "standard_name")
    before = stage.GetRootLayer().ExportToString()
    result = prepare_export_stage(stage, name_override_attribute=override)
    assert result["prim_names"] == {
        "/base": "standard_name",
        "/arm": "custom_arm" if override else "arm",
        "/tool": "tool",
    }
    assert stage.GetRootLayer().ExportToString() == before


def test_mimic_conversion_and_tree_validation(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text("""<robot name="r"><link name="world"/><link name="base"/>
    <link name="finger"/><link name="follower"/>
    <joint name="floating_base" type="floating"><parent link="world"/><child link="base"/></joint>
    <joint name="drive" type="revolute"><parent link="base"/><child link="finger"/>
    <limit lower="-.61" upper="0" effort="0" velocity="0"/></joint>
    <joint name="follow" type="revolute"><parent link="finger"/><child link="follower"/>
    <mimic joint="drive" multiplier="1" offset="3"/></joint></robot>""")
    metadata = {
        "joints": {
            "drive": {"effort": 16.5, "velocity": 2.55},
            "follow": {"mimic": {"joint": "drive", "multiplier": -1, "offset": -math.radians(3)}},
        }
    }
    result = finish_export(path, metadata)
    assert (result["root"], result["links"], result["joints"]) == ("base", 3, 2)
    root = ET.parse(path).getroot()
    mimic = root.find(".//mimic")
    assert float(mimic.get("multiplier")) == -1
    assert float(mimic.get("offset")) == pytest.approx(-math.radians(3))
    assert root.find(".//limit").get("effort") == "16.5"
    ET.SubElement(root, "link", name="base")
    ET.ElementTree(root).write(path)
    with pytest.raises(ValueError, match="Duplicate link"):
        validate_urdf(path)


def test_robot_catalog_requires_exported_asset(tmp_path, monkeypatch):
    import yaml

    from jiuwensymbiosis.simulators.assets import build_index, load_config
    from jiuwensymbiosis.simulators.assets import index as asset_index
    from jiuwensymbiosis.simulators.genesis.scene_catalog import scene_catalog, scene_path

    directory = tmp_path / "another"
    directory.mkdir()
    (directory / "model.usda").touch()
    config = tmp_path / "assets.yaml"
    identifier = "library://spiritai/another/model"
    config.write_text(
        yaml.safe_dump(
            {
                "index": "index.sqlite",
                "roots": [{"id": "spiritai", "scanner": "spiritai", "path": "."}],
            }
        )
    )
    monkeypatch.setattr(asset_index, "DEFAULT_INDEX_PATH", tmp_path / "index.sqlite")
    path = directory / "model.urdf"
    build_index(load_config(config))
    assert len(scene_catalog()) == 1
    with pytest.raises(ValueError, match="URDF"):
        scene_path(identifier)
    path.touch()
    build_index(load_config(config))
    assert scene_path(identifier) == path
    assert scene_catalog()[1] == {"id": identifier, "label": "spiritai · another · URDF 检视", "preview_mode": "urdf"}


def test_physx_mimic_sign_units_and_source_integrity():
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.RevoluteJoint.Define(stage, "/driver")
    follower = UsdPhysics.RevoluteJoint.Define(stage, "/follower").GetPrim()
    follower.AddAppliedSchema("PhysxMimicJointAPI:rotX")
    prefix = "physxMimicJoint:rotX:"
    follower.CreateAttribute(prefix + "gearing", Sdf.ValueTypeNames.Float).Set(1.5)
    follower.CreateAttribute(prefix + "offset", Sdf.ValueTypeNames.Float).Set(3.0)
    follower.CreateRelationship(prefix + "referenceJoint").SetTargets(["/driver"])
    before = stage.GetRootLayer().ExportToString()
    result = prepare_export_stage(stage)
    assert stage.GetRootLayer().ExportToString() == before
    mimic = result["joints"]["follower"]["mimic"]
    assert mimic["multiplier"] == -1.5
    assert mimic["offset"] == pytest.approx(-math.radians(3))


@pytest.mark.parametrize("edge", ['<child link="tip"/>', '<parent link="base"/>'])
def test_missing_joint_endpoint_is_rejected_before_loading(tmp_path, edge):
    path = tmp_path / "broken.urdf"
    path.write_text(f'<robot><link name="base"/><link name="tip"/><joint name="j" type="fixed">{edge}</joint></robot>')
    with pytest.raises(ValueError, match="Invalid parent/child"):
        validate_urdf(path)
