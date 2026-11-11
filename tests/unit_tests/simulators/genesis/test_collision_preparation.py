"""CPU contract tests for explicit, robot-independent collision policies."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import subprocess
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from jiuwensymbiosis.simulators.genesis.collision_preparation import apply_collision_policy

POLICY = {
    "schema_version": 1,
    "name": "fixture",
    "disabled_links": {"loop_a": "fixture exclusion", "loop_b": "fixture exclusion"},
    "decomposed_links": ["jaw_a", "jaw_b", "jaw_c"],
}


def apply_profile(root, directory):
    return apply_collision_policy(root, directory, POLICY)


np = pytest.importorskip("numpy")
trimesh = pytest.importorskip("trimesh")


@pytest.fixture
def collision_case(tmp_path, monkeypatch):
    """Substitute only the external decomposition engine, with a known solid."""
    solid = trimesh.creation.box(extents=[0.02, 0.03, 0.04])
    solid.apply_translation([0.12, -0.03, 0.07])
    source = tmp_path / "jaw.obj"
    source.write_text(trimesh.exchange.obj.export_obj(solid, digits=17))
    backend = types.SimpleNamespace(
        Mesh=lambda vertices, faces: (vertices, faces),
        run_coacd=lambda mesh, **kwargs: [(solid.vertices.copy(), solid.faces.copy())],
    )
    monkeypatch.setitem(sys.modules, "coacd", backend)
    original_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "1.0.14" if name == "coacd" else original_version(name),
    )
    root = ET.Element("robot", name="fixture")
    for name in (
        "jaw_a",
        "jaw_b",
        "jaw_c",
        "loop_a",
        "loop_b",
        "untouched",
    ):
        link = ET.SubElement(root, "link", name=name)
        ET.SubElement(ET.SubElement(link, "inertial"), "mass", value="0.123")
        visual = ET.SubElement(link, "visual", name="original_visual")
        ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", filename=str(source))
        collision = ET.SubElement(link, "collision", name="original_collision")
        ET.SubElement(collision, "origin", xyz="0.01 -0.02 0.03", rpy="0.2 0.3 -0.4")
        ET.SubElement(ET.SubElement(collision, "geometry"), "mesh", filename=str(source), scale="2 3 4")
    joint = ET.SubElement(root, "joint", name="passive", type="revolute")
    ET.SubElement(joint, "mimic", joint="master", multiplier="-1", offset="0.1")
    return root, source, solid, backend, tmp_path / "prepared"


def test_profile_preserves_frames_visuals_inertia_and_mimics(collision_case):
    root, source, solid, _, directory = collision_case
    before = copy.deepcopy(root)

    manifest = apply_profile(root, directory)

    assert json.loads(json.dumps(manifest))["profile"] == "fixture"
    assert ET.tostring(root.find("joint")) == ET.tostring(before.find("joint"))
    for link, old_link in zip(root.findall("link"), before.findall("link"), strict=True):
        name = link.get("name")
        for tag in ("visual", "inertial"):
            assert ET.tostring(link.find(tag)) == ET.tostring(old_link.find(tag))
        if name in {"loop_a", "loop_b"}:
            assert link.findall("collision") == []
        elif name == "untouched":
            assert ET.tostring(link) == ET.tostring(old_link)
        else:
            collisions = link.findall("collision")
            assert len(collisions) == 1
            collision = collisions[0]
            assert collision.find("origin").attrib == old_link.find("collision/origin").attrib
            mesh = collision.find("geometry/mesh")
            assert mesh.get("scale") == "2 3 4"
            assert Path(mesh.get("filename")).is_absolute()
            assert Path(mesh.get("filename")) != source
            exported = trimesh.load(mesh.get("filename"), force="mesh", process=True)
            np.testing.assert_allclose(exported.bounds, solid.bounds, atol=1e-12)
            assert exported.is_watertight and exported.is_convex
            assert exported.volume == pytest.approx(0.000024)


@pytest.mark.parametrize("defect", ["empty", "nonfinite", "open", "inverted", "concave", "translated"])
def test_invalid_decomposition_is_rejected_before_urdf_changes(collision_case, defect):
    root, _, solid, backend, directory = collision_case
    vertices, faces = solid.vertices.copy(), solid.faces.copy()
    if defect == "nonfinite":
        vertices[0, 0] = float("nan")
    elif defect == "open":
        faces = faces[:-1]
    elif defect == "inverted":
        faces = faces[:, ::-1]
    elif defect == "concave":
        vertices[0] += [0.009, 0.014, 0.019]
    elif defect == "translated":
        vertices += [1.0, 0.0, 0.0]
    backend.run_coacd = lambda mesh, **kwargs: [] if defect == "empty" else [(vertices, faces)]
    before = ET.tostring(root)

    with pytest.raises(ValueError, match="decomposition"):
        apply_profile(root, directory)

    assert ET.tostring(root) == before


def test_repeated_preparation_reuses_verified_content_addressed_parts(collision_case):
    root, _, _, backend, directory = collision_case
    original = copy.deepcopy(root)
    first = apply_profile(root, directory)

    def unavailable(*args, **kwargs):
        raise AssertionError("valid cached decomposition must be reusable without running CoACD")

    backend.run_coacd = unavailable
    second = apply_profile(original, directory)
    for name, records in second["decomposed_links"].items():
        assert records[0]["cache_hit"] is True
        assert records[0]["cache_key"] == first["decomposed_links"][name][0]["cache_key"]
        assert records[0]["parts"] == first["decomposed_links"][name][0]["parts"]


def test_corrupt_cached_part_cannot_reach_urdf(collision_case):
    root, _, _, _, directory = collision_case
    original = copy.deepcopy(root)
    report = apply_profile(root, directory)
    record = report["decomposed_links"]["jaw_a"][0]
    part = Path(record["parts"][0]["path"])
    part.write_text(part.read_text() + "# unverified change\n")
    before = ET.tostring(original)

    with pytest.raises(ValueError, match="cached decomposition"):
        apply_profile(original, directory)

    assert ET.tostring(original) == before


def test_source_content_change_invalidates_cache(collision_case):
    root, source, solid, backend, directory = collision_case
    original = copy.deepcopy(root)
    first = apply_profile(root, directory)
    changed = solid.copy()
    changed.apply_translation([0.001, 0.002, 0.003])
    source.write_text(trimesh.exchange.obj.export_obj(changed, digits=17))
    backend.run_coacd = lambda mesh, **kwargs: [(changed.vertices, changed.faces)]

    second = apply_profile(original, directory)

    before = first["decomposed_links"]["jaw_a"][0]
    after = second["decomposed_links"]["jaw_a"][0]
    assert before["cache_key"] != after["cache_key"]
    assert before["source_sha256"] != after["source_sha256"]
    exported = trimesh.load(after["parts"][0]["path"], force="mesh", process=True)
    np.testing.assert_allclose(exported.bounds, changed.bounds, atol=1e-12)


def test_backend_version_change_invalidates_cache(collision_case, monkeypatch):
    root, _, _, _, directory = collision_case
    original = copy.deepcopy(root)
    first = apply_profile(root, directory)
    original_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "1.0.15" if name == "coacd" else original_version(name),
    )

    second = apply_profile(original, directory)

    assert first["algorithm_version"] == "1.0.14"
    assert second["algorithm_version"] == "1.0.15"
    assert first["decomposed_links"]["jaw_a"][0]["cache_key"] != second["decomposed_links"]["jaw_a"][0]["cache_key"]


def test_missing_collision_does_not_partially_apply_profile(collision_case):
    root, _, _, _, directory = collision_case
    link = root.find("link[@name='jaw_c']")
    link.remove(link.find("collision"))
    before = ET.tostring(root)

    with pytest.raises(ValueError, match="missing collision"):
        apply_profile(root, directory)

    assert ET.tostring(root) == before


def test_relative_source_is_rejected_without_mutation(collision_case):
    root, _, _, _, directory = collision_case
    root.find("link[@name='jaw_c']/collision/geometry/mesh").set("filename", "relative.obj")
    before = ET.tostring(root)

    with pytest.raises(ValueError, match="must be absolute"):
        apply_profile(root, directory)

    assert ET.tostring(root) == before


def test_module_import_does_not_load_geometry_or_gpu_dependencies():
    module = Path(__file__).resolve().parents[4] / "jiuwensymbiosis/simulators/genesis/collision_preparation.py"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1]); "
            "assert not {'numpy', 'scipy', 'trimesh', 'coacd', 'torch', 'genesis'} & sys.modules.keys()",
            str(module),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_multiple_convex_parts_become_separate_collisions(collision_case):
    root, _, _, backend, directory = collision_case
    parts = []
    for x in (0.115, 0.125):
        part = trimesh.creation.box(extents=[0.01, 0.03, 0.04])
        part.apply_translation([x, -0.03, 0.07])
        parts.append((part.vertices, part.faces))
    backend.run_coacd = lambda mesh, **kwargs: parts

    report = apply_profile(root, directory)

    for name, records in report["decomposed_links"].items():
        collisions = root.find(f"link[@name='{name}']").findall("collision")
        assert len(collisions) == 2
        assert len({collision.get("name") for collision in collisions}) == 2
        assert {collision.find("geometry/mesh").get("filename") for collision in collisions} == {
            part["path"] for part in records[0]["parts"]
        }


def test_cache_manifest_cannot_redirect_part_paths(collision_case):
    root, _, _, _, directory = collision_case
    original = copy.deepcopy(root)
    report = apply_profile(root, directory)
    record = report["decomposed_links"]["jaw_a"][0]
    cache_manifest = Path(record["cache_manifest"])
    metadata = json.loads(cache_manifest.read_text())
    metadata["parts"][0]["file"] = "../unverified.obj"
    cache_manifest.write_text(json.dumps(metadata))
    before = ET.tostring(original)

    with pytest.raises(ValueError, match="cached decomposition.*filename"):
        apply_profile(original, directory)

    assert ET.tostring(original) == before


def test_installed_coacd_cpu_roundtrip(collision_case, monkeypatch):
    root, _, solid, _, directory = collision_case
    monkeypatch.delitem(sys.modules, "coacd")
    pytest.importorskip("coacd")

    report = apply_profile(root, directory)

    assert report["parameters"]["pca"] is False
    assert report["parameters"]["seed"] == 0
    for records in report["decomposed_links"].values():
        record = records[0]
        assert len(record["parts"]) == 1
        exported = trimesh.load(record["parts"][0]["path"], force="mesh", process=False)
        np.testing.assert_allclose(exported.bounds, solid.bounds, atol=1e-8)
        assert exported.volume == pytest.approx(0.000024, rel=1e-6)


def test_distinct_sources_keep_distinct_meshes_in_mujoco_urdf_import(collision_case):
    """Genesis' MuJoCo URDF pass keys mesh assets by filename basename."""
    mujoco = pytest.importorskip("mujoco")
    root, source, solid, backend, directory = collision_case
    names = ("jaw_a", "jaw_b", "jaw_c")
    for index, name in enumerate(names):
        mesh = solid.copy()
        mesh.apply_scale(1 + index * 0.1)
        filename = source.with_name(f"jaw-{index}.obj")
        filename.write_text(trimesh.exchange.obj.export_obj(mesh, digits=17))
        root.find(f"link[@name='{name}']/collision/geometry/mesh").set("filename", str(filename))
    backend.run_coacd = lambda mesh, **kwargs: [(mesh[0].copy(), mesh[1].copy())]

    report = apply_profile(root, directory)

    # Exercise the real CPU importer on separate bodies, rather than only
    # inspecting the generated XML, which did not expose the original aliasing.
    robot = ET.Element("robot", name="generated_mesh_import")
    ET.SubElement(robot, "link", name="base")
    compiler = ET.SubElement(ET.SubElement(robot, "mujoco"), "compiler")
    compiler.set("strippath", "false")
    for index, name in enumerate(names):
        link = ET.SubElement(robot, "link", name=name)
        link.append(copy.deepcopy(root.find(f"link[@name='{name}']/collision")))
        joint = ET.SubElement(robot, "joint", name=f"{name}_joint", type="fixed")
        ET.SubElement(joint, "parent", link="base")
        ET.SubElement(joint, "child", link=name)
        ET.SubElement(joint, "origin", xyz=f"{index} 0 0")
    imported = mujoco.MjModel.from_xml_string(ET.tostring(robot).decode())
    assert imported.nmesh == len(names)
    assert len(set(imported.geom_dataid)) == len(names)
    filenames = []
    for records in report["decomposed_links"].values():
        record = records[0]
        filename = Path(record["parts"][0]["path"]).name
        assert filename == f"{record['cache_key']}_part_000.obj"
        filenames.append(filename)
        assert json.loads(Path(record["cache_manifest"]).read_text())["identity"]["schema"] == 2
    assert len(set(filenames)) == len(names)


@pytest.mark.parametrize(
    "override", [{"max_convex_hull": 0}, {"threshold": float("nan")}, {"seed": True}, {"pca": "yes"}, {"unknown": 1}]
)
def test_invalid_policy_parameters_fail_before_mutation(collision_case, override):
    root, _, _, _, directory = collision_case
    before = ET.tostring(root)
    with pytest.raises(ValueError):
        apply_collision_policy(root, directory, {**POLICY, "parameters": override})
    assert ET.tostring(root) == before
    assert not directory.exists()
