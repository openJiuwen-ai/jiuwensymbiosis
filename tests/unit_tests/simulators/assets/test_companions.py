"""Scanner association is based on provenance or exact names, never arbitrary URDF choice."""

import hashlib
import json

import pytest

from jiuwensymbiosis.simulators.assets import AssetError, build_index, probe_asset


def test_export_manifest_can_identify_a_differently_named_urdf(asset_fixture):
    library, config, roots, _, write = asset_fixture
    directory = roots["spiritai"] / "moz/urdf"
    (directory / "robot.urdf").rename(directory / "kinematics.urdf")
    write(
        "spiritai",
        "moz/urdf/export_manifest.json",
        json.dumps(
            {
                "source": "../robot.usda",
                "urdf": "kinematics.urdf",
                "file_meshes_are_zup": True,
            }
        ),
    )
    build_index(config)
    resolved = library.resolve("library://spiritai/moz/robot")
    assert resolved.preview_path.name == "kinematics.urdf"
    assert set(resolved.paths) == {"source", "genesis_urdf", "nyx_urdf", "material_manifest", "export_manifest"}
    assert probe_asset(library, resolved.asset.id)["material_bundle"] == "sha256_verified"


def test_plain_same_named_urdf_is_discovered_for_both_robot_layouts(asset_fixture):
    library, config, _, _, write = asset_fixture
    for root, relative in (("geniesim", "robot/Arm/robot.urdf"), ("catalog", "Robots/Vendor/Arm/urdf/arm.urdf")):
        write(root, relative, '<robot name="test"><link name="base"/></robot>')
    build_index(config)
    assert library.search(preview_mode="urdf").total == 3
    for root in ("geniesim", "catalog"):
        asset = library.search(root_id=root, preview_mode="urdf").items[0]
        assert asset.options["file_meshes_are_zup"] is False
        assert set(library.resolve(asset.id).paths) == {"source", "genesis_urdf"}


@pytest.mark.parametrize(
    "issue",
    [
        "ambiguous",
        "wrong_export_source",
        "wrong_material_source",
        "outside_urdf",
        "missing_render",
        "malformed",
        "missing_manifest",
    ],
)
def test_invalid_companion_associations_leave_old_index_intact(asset_fixture, issue, tmp_path):
    library, config, roots, _, write = asset_fixture
    before = config.index.read_bytes()
    if issue == "ambiguous":
        write("geniesim", "robot/Arm/robot.urdf", '<robot name="one"/>')
        write("geniesim", "robot/Arm/urdf/robot.urdf", '<robot name="two"/>')
    elif issue == "wrong_export_source":
        write(
            "spiritai",
            "moz/urdf/export_manifest.json",
            json.dumps(
                {
                    "source": "../variant.usda",
                    "urdf": "robot.urdf",
                }
            ),
        )
    elif issue == "wrong_material_source":
        path = roots["spiritai"] / "moz/urdf/materials/manifest.json"
        data = json.loads(path.read_text())
        data["source"] = "../../variant.usda"
        path.write_text(json.dumps(data))
    elif issue == "outside_urdf":
        outside = tmp_path / "outside.urdf"
        outside.touch()
        write(
            "spiritai", "moz/urdf/export_manifest.json", json.dumps({"source": "../robot.usda", "urdf": str(outside)})
        )
    elif issue == "missing_render":
        (roots["spiritai"] / "moz/urdf/materials/robot.urdf").unlink()
    elif issue == "malformed":
        write("spiritai", "moz/urdf/export_manifest.json", "[]")
    else:
        (roots["spiritai"] / "moz/urdf/materials/manifest.json").unlink()
    with pytest.raises(AssetError):
        build_index(config)
    assert config.index.read_bytes() == before
    assert library.search().total == 10


def test_unrelated_urdf_does_not_become_a_robot_representation(asset_fixture):
    library, config, _, _, write = asset_fixture
    write("geniesim", "robot/Arm/urdf/gripper.urdf", '<robot name="gripper"/>')
    build_index(config)
    asset = library.get("library://geniesim/robot/Arm/robot")
    assert asset.preview_mode is None
    assert set(asset.representations) == {"source"}


def test_material_hashes_are_checked_from_the_index_snapshot(asset_fixture):
    library, config, roots, _, write = asset_fixture
    texture = write("spiritai", "moz/urdf/materials/texture.png", "original")
    manifest = roots["spiritai"] / "moz/urdf/materials/manifest.json"
    data = json.loads(manifest.read_text())
    data["files_sha256"]["texture.png"] = hashlib.sha256(texture.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(data))
    build_index(config)
    texture.write_text("modified")
    # A live source-manifest edit cannot overwrite the database's published digest.
    data["files_sha256"]["texture.png"] = hashlib.sha256(texture.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(data))
    with pytest.raises(AssetError, match="摘要不匹配"):
        probe_asset(library, "library://spiritai/moz/robot")
