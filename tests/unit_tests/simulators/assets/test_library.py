"""Asset discovery, cache publication and failure boundaries using real files/SQLite."""

import json
import shutil
import sqlite3
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from jiuwensymbiosis.simulators.assets import AssetError, AssetLibrary, build_index, load_config, probe_asset
from jiuwensymbiosis.simulators.assets.models import asset_id


def test_three_libraries_entry_filtering_and_preview_admission(asset_fixture):
    library, config, _, _, _ = asset_fixture
    page = library.search(limit=1000)
    assert page.total == 10
    assert {a.root_id for a in page.items} == {"catalog", "geniesim", "spiritai"}
    assert library.search(kind="robot").total == 3
    assert library.search(preview_mode="urdf").total == 1
    assert library.search(preview_mode="static_usd").total == 3
    assert library.search(query="chair").items[0].kind == "object"
    assert library.search(query="ceiling").items[0].preview_mode is None
    assert library.search(query="physics").total == 0
    assert library.search(query="finger").total == 0
    cup = library.search(query="cup").items
    assert len(cup) == 1 and cup[0].representations["source"].endswith("Aligned.usda")
    assert library.search(root_id="spiritai").total == 1
    assert library.get("library://spiritai/moz/robot").name == "moz"
    assert library.status()["validation"] == "entry_files_only"


def test_query_is_literal_paginated_and_no_scan_on_read(asset_fixture):
    library, config, _, _, write = asset_fixture
    first = library.search(limit=3)
    second = library.search(limit=3, offset=3)
    assert len(first.items) == len(second.items) == 3
    assert not {a.id for a in first.items} & {a.id for a in second.items}
    for query in ("%", "' OR 1=1 --"):
        assert library.search(query=query).total == 0
    write("geniesim", "background/home/new/background.usd")
    assert library.search(query="home/new").total == 0
    build_index(config)
    assert library.search(query="home/new").total == 1
    with sqlite3.connect(config.index) as db:
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    for arguments in ({"limit": 0}, {"limit": 1001}, {"offset": -1}):
        with pytest.raises(AssetError):
            library.search(**arguments)


def test_failed_rebuild_retains_previous_database(asset_fixture):
    library, config, roots, _, _ = asset_fixture
    before = config.index.read_bytes()
    roots["geniesim"].rename(roots["geniesim"].with_name("temporarily_offline"))
    with pytest.raises(AssetError, match="不存在"):
        build_index(config)
    assert config.index.read_bytes() == before
    assert library.search(query="cup").total == 1  # catalogue is still queryable while the drive is offline
    with pytest.raises(AssetError, match="不存在"):
        library.resolve(library.search(query="cup").items[0].id)


def test_configuration_changes_and_root_relocation_only_apply_after_rebuild(asset_fixture):
    library, config, roots, data, _ = asset_fixture
    new_root = roots["spiritai"].with_name("relocated")
    shutil.copytree(roots["spiritai"], new_root, copy_function=shutil.copy2)
    data["roots"][2]["path"] = str(new_root)
    config.source.write_text(yaml.safe_dump(data))
    assert library.resolve("library://spiritai/moz/robot").root == roots["spiritai"]
    build_index(load_config(config.source))
    shutil.rmtree(roots["spiritai"])
    moved = AssetLibrary(config.index)
    assert moved.resolve("library://spiritai/moz/robot").preview_path.is_relative_to(new_root)
    config.source.write_text("invalid: [")
    assert AssetLibrary(config.index).search().total == 10


def test_changed_missing_or_escaped_file_never_loads(asset_fixture, tmp_path):
    library, config, roots, _, write = asset_fixture
    robot = library.resolve("library://spiritai/moz/robot")
    write("spiritai", "moz/urdf/robot.urdf", "changed")
    with pytest.raises(AssetError, match="已变化"):
        library.resolve("library://spiritai/moz/robot")
    robot.preview_path.unlink()
    outside = tmp_path / "secret.urdf"
    outside.write_text("outside")
    robot.preview_path.symlink_to(outside)
    with pytest.raises(AssetError, match="离开根目录"):
        library.resolve("library://spiritai/moz/robot")
    for identifier in ("../../secret", str(outside), "https://example.com/robot.usda"):
        with pytest.raises(AssetError, match="未登记"):
            library.resolve(identifier)


def test_urdf_and_material_probe_checks_actual_files(asset_fixture):
    library, config, roots, _, _ = asset_fixture
    report = probe_asset(library, "library://spiritai/moz/robot")
    assert not report["runtime_validated"]
    assert report["material_bundle"] == "sha256_verified"
    assert len(report["urdf"]["genesis_urdf"]["dependencies"]) == 2
    mesh = roots["spiritai"] / "moz/urdf/meshes/base.mtl"
    mesh.unlink()
    with pytest.raises(AssetError, match="依赖缺失"):
        probe_asset(library, "library://spiritai/moz/robot")


def test_invalid_material_bundle_does_not_replace_the_existing_index(asset_fixture):
    library, config, _, _, write = asset_fixture
    write("spiritai", "moz/urdf/materials/robot.urdf", '<robot name="changed"><link name="base"/></robot>')
    before = config.index.read_bytes()
    with pytest.raises(AssetError, match="摘要不匹配"):
        build_index(config)
    assert config.index.read_bytes() == before
    with pytest.raises(AssetError, match="已变化"):
        library.resolve("library://spiritai/moz/robot")


def test_glb_external_texture_is_included_in_reference_probe(tmp_path):
    from jiuwensymbiosis.simulators.assets.probe import probe_urdf

    source = tmp_path / "robot.urdf"
    source.write_text(
        '<robot name="test"><link name="base"><visual><geometry><mesh filename="mesh.glb"/></geometry></visual></link></robot>'
    )
    chunk = json.dumps({"asset": {"version": "2.0"}, "images": [{"uri": "texture.png"}]}).encode()
    chunk += b" " * (-len(chunk) % 4)
    (tmp_path / "mesh.glb").write_bytes(
        struct.pack("<4sIIII", b"glTF", 2, 20 + len(chunk), len(chunk), 0x4E4F534A) + chunk
    )
    with pytest.raises(AssetError, match="依赖缺失"):
        probe_urdf(source, tmp_path)
    (tmp_path / "texture.png").touch()
    assert probe_urdf(source, tmp_path)["dependencies"] == ["mesh.glb", "texture.png"]


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_root",
        "outside_content",
        "unknown_scanner",
        "entries",
        "aliases",
        "wrong_index",
        "missing_families",
        "outside_family",
        "invalid_preview_depth",
    ],
)
def test_invalid_configuration_rejected_before_build(asset_fixture, change):
    library, config, _, data, _ = asset_fixture
    if change == "duplicate_root":
        data["roots"].append(data["roots"][0])
    elif change == "outside_content":
        data["roots"][0]["content_path"] = "../outside"
    elif change == "unknown_scanner":
        data["roots"][0]["scanner"] = "synthesis"
    elif change in {"entries", "aliases"}:
        data[change] = [] if change == "entries" else {}
    elif change == "wrong_index":
        data["index"] = "robot.usda"
    elif change == "missing_families":
        data["roots"][0]["families"] = {}
    elif change == "outside_family":
        data["roots"][0]["families"] = {"../outside": {"kind": "environment"}}
    else:
        data["roots"][0]["families"]["Environments"]["preview_depth_overrides"] = {"Outdoor": 0}
    config.source.write_text(yaml.safe_dump(data))
    with pytest.raises(AssetError):
        load_config(config.source)


def test_ambiguous_spiritai_entry_and_geniesim_noncanonical_wrapper(asset_fixture):
    library, config, _, _, write = asset_fixture
    write("spiritai", "ambiguous/a.usd")
    write("spiritai", "ambiguous/b.usd")
    write("geniesim", "objects/benchmark/cup/Aligned.usda", '#usda 1.0\n(upAxis = "Y")\n')
    report = build_index(config)
    assert len(report["diagnostics"]) == 1
    assert library.search(query="ambiguous").total == 0
    assert library.search(query="cup").items[0].representations["source"].endswith("Aligned.usd")


def test_standalone_cli_is_json_and_does_not_import_agent_or_gpu(asset_fixture):
    library, config, _, _, _ = asset_fixture
    import jiuwensymbiosis.simulators.assets as package

    directory = Path(package.__file__).parent
    result = subprocess.run(
        [sys.executable, str(directory), "--index", str(config.index), "search", "moz"],
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    assert json.loads(result.stdout)["total"] == 1
    code = f"import sys; sys.path.insert(0, {str(directory.parent)!r}); import assets; assert not {{'genesis','torch','pxr','openjiuwen','jiuwensymbiosis'}} & sys.modules.keys()"
    subprocess.run([sys.executable, "-c", code], capture_output=True, check=True, timeout=20)


def test_ids_are_path_based_and_escape_special_characters():
    assert (
        asset_id("example", "room with spaces/入口.usda") == "library://example/room%20with%20spaces/%E5%85%A5%E5%8F%A3"
    )
    assert asset_id("example", "room.usd") == asset_id("example", "room.usda")


def test_optional_usd_probe_reports_units_and_unresolved_resources(asset_fixture):
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdGeom

    library, config, roots, _, _ = asset_fixture
    source = roots["catalog"] / "Environments/Room/room.usd"
    stage = Usd.Stage.CreateNew(str(source))
    prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(prim)
    UsdGeom.SetStageUpAxis(stage, "Y")
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    prim.CreateAttribute("missing", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("missing.png"))
    stage.GetRootLayer().Save()
    build_index(config)
    report = probe_asset(library, "library://catalog/Environments/Room/room", usd=True)
    assert report["usd"]["default_prim"] == "/World"
    assert report["usd"]["up_axis"] == "Y"
    assert report["usd"]["meters_per_unit"] == pytest.approx(0.01)
    assert report["usd"]["dependency_status"] == "incomplete"
    assert report["usd"]["unresolved_dependencies"] == ["missing.png"]
    assert report["runtime_validated"] is False


def test_content_tree_uses_configured_folders_and_depths(asset_fixture):
    library, config, _, data, write = asset_fixture
    write("catalog", "MyScenes/room.usd")
    write("catalog", "MyScenes/Campus/site/scene.usda")
    write("catalog", "MyScenes/Campus/site/layers/lighting.usd")
    write("catalog", "MyScenes/house/Props/chair.usd")
    write("catalog", "MyRobots/arm.usda")
    data["roots"][0]["families"] = {
        "MyScenes": {"kind": "environment", "preview_depth": 1, "preview_depth_overrides": {"Campus": 3}},
        "MyRobots": {"kind": "robot"},
    }
    config.source.write_text(yaml.safe_dump(data))
    build_index(load_config(config.source))
    assert library.search(root_id="catalog").total == 5
    assert library.search(root_id="catalog", preview_mode="static_usd").total == 2
    assert library.search(query="lighting").items[0].kind == "other"
    assert library.search(query="chair").items[0].kind == "object"
    assert library.search(root_id="catalog", kind="robot").items[0].name == "arm"


def test_queries_and_loading_use_sqlite_after_all_build_inputs_are_removed(asset_fixture, tmp_path, monkeypatch):
    from jiuwensymbiosis.simulators.assets import index as asset_index
    from jiuwensymbiosis.simulators.genesis.scene_catalog import resolve_scene, scene_catalog

    library, config, roots, _, _ = asset_fixture
    config.source.unlink()
    for path in ("moz/config.yaml", "moz/urdf/export_manifest.json", "moz/urdf/materials/manifest.json"):
        (roots["spiritai"] / path).unlink()
    index_copy = tmp_path / "standalone.sqlite"
    shutil.copy2(config.index, index_copy)
    config.index.unlink()
    monkeypatch.setattr(asset_index, "DEFAULT_INDEX_PATH", index_copy)
    monkeypatch.chdir(tmp_path)
    library = AssetLibrary()
    assert library.search().total == 10
    assert library.get("library://spiritai/moz/robot").options["file_meshes_are_zup"] is True
    assert probe_asset(library, "library://spiritai/moz/robot")["material_bundle"] == "sha256_verified"
    assert len(scene_catalog()) == 5
    assert resolve_scene("library://spiritai/moz/robot").paths["genesis_urdf"].is_file()
    with pytest.raises(AssetError, match="未登记"):
        library.get("robot/test")
    import jiuwensymbiosis.simulators.assets as package

    result = subprocess.run(
        [
            sys.executable,
            str(Path(package.__file__).parent),
            "--index",
            str(index_copy),
            "show",
            "library://spiritai/moz/robot",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    assert json.loads(result.stdout)["paths"]["genesis_urdf"] == str(roots["spiritai"] / "moz/urdf/robot.urdf")


def test_missing_or_old_index_is_rejected_without_fallback(asset_fixture, tmp_path):
    library, config, _, _, _ = asset_fixture
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(AssetError, match="未构建"):
        AssetLibrary(missing).search()
    assert not missing.exists()
    with sqlite3.connect(config.index) as db:
        manifest = json.loads(db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
        manifest["schema_version"] = 1
        db.execute("UPDATE metadata SET value=? WHERE key='manifest'", (json.dumps(manifest),))
    with pytest.raises(AssetError, match="版本不兼容"):
        library.search()


@pytest.mark.parametrize("extension", [".USD", ".UsDa", ".USDC"])
@pytest.mark.parametrize("scanner", ["content_tree", "geniesim", "spiritai"])
def test_case_insensitive_usd_extensions_are_indexed(tmp_path, extension, scanner):
    stems = {"content_tree": "room", "geniesim": "background/room/demo/background", "spiritai": "demo/robot"}
    stem = stems[scanner]
    source = tmp_path / (stem + extension)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("#usda 1.0\n")
    root = {"id": "demo", "path": ".", "scanner": scanner}
    if scanner == "content_tree":
        root["families"] = {".": {"kind": "environment"}}
    config = tmp_path / "assets.yaml"
    config.write_text(yaml.safe_dump({"index": "index.sqlite", "roots": [root]}))
    assert build_index(load_config(config))["total"] == 1
    assert AssetLibrary(tmp_path / "index.sqlite").resolve("library://demo/" + stem).paths["source"] == source


@pytest.mark.parametrize("filename", ["mesh.obj", "mesh.mtl", "mesh.gltf"])
def test_non_utf8_mesh_references_raise_asset_error(tmp_path, filename):
    from jiuwensymbiosis.simulators.assets.probe import probe_urdf

    mesh = "mesh.gltf" if filename.endswith("gltf") else "mesh.obj"
    source = tmp_path / "robot.urdf"
    source.write_text(
        f'<robot name="demo"><link name="base"><visual><geometry><mesh filename="{mesh}"/></geometry></visual></link></robot>'
    )
    (tmp_path / "mesh.obj").write_text("mtllib mesh.mtl\n")
    (tmp_path / filename).write_bytes(b"\xff")
    with pytest.raises(AssetError, match="UTF-8"):
        probe_urdf(source, tmp_path)


@pytest.mark.parametrize("extension", [".gltf", ".glb"])
@pytest.mark.parametrize("content", [[], None, "invalid", 1, True])
def test_gltf_non_object_json_raises_asset_error(tmp_path, extension, content):
    from jiuwensymbiosis.simulators.assets.probe import probe_urdf

    mesh = tmp_path / ("mesh" + extension)
    payload = json.dumps(content).encode("utf-8")
    if extension == ".glb":
        payload += b" " * (-len(payload) % 4)
        payload = struct.pack("<4sIIII", b"glTF", 2, 20 + len(payload), len(payload), 0x4E4F534A) + payload
    mesh.write_bytes(payload)
    source = tmp_path / "robot.urdf"
    source.write_text(
        f'<robot name="demo"><link name="base"><visual><geometry><mesh filename="{mesh.name}"/></geometry></visual></link></robot>'
    )
    with pytest.raises(AssetError, match="JSON 顶层必须是对象") as error:
        probe_urdf(source, tmp_path)
    assert str(mesh) in str(error.value)


@pytest.mark.parametrize(
    "filename, payload, message", [("mesh.gltf", b"[]", "JSON 顶层必须是对象"), ("mesh.obj", b"\xff", "UTF-8")]
)
def test_probe_cli_reports_invalid_asset_as_json(asset_fixture, filename, payload, message):
    import jiuwensymbiosis.simulators.assets as package

    _, config, roots, _, write = asset_fixture
    write(
        "catalog",
        "Robots/Vendor/Arm/arm.urdf",
        f'<robot name="demo"><link name="base"><visual><geometry><mesh filename="{filename}"/></geometry></visual></link></robot>',
    )
    (roots["catalog"] / "Robots/Vendor/Arm" / filename).write_bytes(payload)
    build_index(config)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(Path(package.__file__).parent),
            "--index",
            str(config.index),
            "probe",
            "library://catalog/Robots/Vendor/Arm/arm",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert message in json.loads(result.stderr)["error"]


def test_directory_cli_ignores_unrelated_assets_modules(asset_fixture):
    library, config, _, _, _ = asset_fixture
    import jiuwensymbiosis.simulators.assets as package

    code = """
import runpy, sys, types
sys.modules["assets"] = types.ModuleType("assets")
sys.modules["_paths"] = types.ModuleType("_paths")
directory, index = sys.argv[1:]
before = list(sys.path)
sys.argv = [directory, "--index", index, "search", "moz"]
try:
    runpy.run_path(directory, run_name="__main__")
except SystemExit as exc:
    assert exc.code == 0
assert sys.path == before
assert not {"jiuwensymbiosis", "openjiuwen", "torch", "genesis", "pxr"} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(Path(package.__file__).parent), str(config.index)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["total"] == 1


def test_case_variants_keep_canonical_wrapper_and_lowercase_preference(asset_fixture):
    library, config, roots, _, _ = asset_fixture
    directory = roots["geniesim"] / "objects/benchmark/cup"
    (directory / "Aligned.usd").rename(directory / "Aligned.USD")
    (directory / "Aligned.usda").unlink()
    (directory / "Aligned.USDA").write_text(
        '#usda 1.0\n(upAxis = "Z")\ndef Xform "Cup" (references = @Aligned.USD@) {}\n'
    )
    build_index(config)
    assert library.search(query="cup").items[0].representations["source"].endswith("Aligned.USDA")
    (directory / "Aligned.usda").write_text('#usda 1.0\n(upAxis = "Y")\n')
    build_index(config)
    assert library.search(query="cup").items[0].representations["source"].endswith("Aligned.USD")
