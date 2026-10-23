"""Tiny realistic source layouts; no workstation assets, USD SDK or GPU needed."""

import hashlib
import json

import pytest
import yaml

from jiuwensymbiosis.simulators.assets import AssetLibrary, build_index, load_config


@pytest.fixture
def asset_fixture(tmp_path):
    roots = {name: tmp_path / name for name in ("catalog", "geniesim", "spiritai")}

    def write(root, relative, content="#usda 1.0\n"):
        path = roots[root] / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    for relative in (
        "Robots/Vendor/Arm/arm.usd",
        "Robots/Vendor/Arm/meshes/mesh.usd",
        "Robots/Vendor/Arm/arm_physics.usda",
        "Robots/Vendor/Arm/parts/finger.usd",
        "Environments/Room/room.usd",
        "Environments/Room/Props/chair.usd",
        "Environments/Room/Layers/ceiling.usd",
        "Materials/paint.usd",
        "Sensors/Camera/cam.usd",
    ):
        write("catalog", relative)
    write("geniesim", "background/kitchen/kitchen_1/background.usda")
    write("geniesim", "robot/Arm/robot.usda")
    write("geniesim", "objects/benchmark/cup/Aligned.usd")
    write(
        "geniesim",
        "objects/benchmark/cup/Aligned.usda",
        '#usda 1.0\n(upAxis = "Z")\ndef Xform "root" (references = @Aligned.usd@) {}\n',
    )
    write("geniesim", "objects/benchmark/cup/description.py", "raise RuntimeError('Do not execute asset scripts')")
    write("geniesim", "real2sim/environment_assets/house/house.usd")
    write("spiritai", "moz/robot.usda")
    write("spiritai", "moz/variant.usda")
    write("spiritai", "moz/config.yaml", "asset_path: robot.usda\n")
    urdf = write(
        "spiritai",
        "moz/urdf/robot.urdf",
        '<robot name="fixture"><link name="base"><visual><geometry><mesh filename="meshes/base.obj"/></geometry></visual></link></robot>',
    )
    write("spiritai", "moz/urdf/meshes/base.obj", "mtllib base.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    write("spiritai", "moz/urdf/meshes/base.mtl", "newmtl gray\nKd 0.2 0.2 0.2\n")
    write(
        "spiritai",
        "moz/urdf/export_manifest.json",
        json.dumps(
            {
                "source": "../robot.usda",
                "urdf": "robot.urdf",
                "file_meshes_are_zup": True,
            }
        ),
    )
    visual = write("spiritai", "moz/urdf/materials/robot.urdf", '<robot name="fixture"><link name="base"/></robot>')
    manifest = {
        "source": "../../robot.usda",
        "physics_urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "render_urdf": "robot.urdf",
        "files_sha256": {"robot.urdf": hashlib.sha256(visual.read_bytes()).hexdigest()},
    }
    write("spiritai", "moz/urdf/materials/manifest.json", json.dumps(manifest))
    source = tmp_path / "assets.yaml"
    data = {
        "version": 1,
        "index": "index.sqlite",
        "roots": [
            {
                "id": "catalog",
                "path": "catalog",
                "scanner": "content_tree",
                "families": {
                    "Robots": {"kind": "robot", "min_depth": 3},
                    "Environments": {"kind": "environment"},
                    "Sensors": {"kind": "sensor"},
                },
            },
            *[{"id": name, "path": name, "scanner": name} for name in ("geniesim", "spiritai")],
        ],
    }
    source.write_text(yaml.safe_dump(data, allow_unicode=True))
    config = load_config(source)
    build_index(config)
    return AssetLibrary(config.index), config, roots, data, write
