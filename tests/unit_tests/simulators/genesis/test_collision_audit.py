"""Collision audit reads real parts and keeps model provenance separate."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.simulators.genesis.collision_audit import collision_geometry_audit


class Geometry:
    def __init__(self, idx, uri=None):
        self.init_pos = np.zeros(3)
        self.init_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.idx = idx
        self.type = "MESH"
        self.is_convex = True
        self.contype = 1
        self.conaffinity = 1
        self.init_verts = np.array([[0, 0, 0], [0.02, 0, 0], [0, 0.03, 0.01]], dtype=float)
        self.init_faces = np.array([[0, 1, 2]])
        self.n_verts = 3
        self.n_faces = 1
        # Already transformed and deliberately unlike local vertices/metadata.
        self.world = np.array([[4, 5, 6], [4, 5.02, 6], [3.97, 5, 6.01]])
        self.metadata = {} if uri is None else {"mesh_path": str(uri)}
        self.reads = 0

    def get_verts(self):
        self.reads += 1
        return self.world

    def get_pos(self):
        raise AssertionError("world vertices must not receive a second transform")

    def get_vverts(self):
        raise AssertionError("collision audit must not use visual geometry")


@pytest.fixture
def sample(tmp_path):
    mesh = tmp_path / "part.obj"
    mesh.write_text("source bytes deliberately not the runtime mesh")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        '<robot name="test"><link name="finger">'
        '<collision><geometry><mesh filename="part.obj"/></geometry></collision>'
        '<collision><geometry><mesh filename="part.obj"/></geometry></collision>'
        '</link><link name="loop"/></robot>'
    )
    parts = [Geometry(10, mesh), Geometry(11, mesh)]
    parts[1].init_verts = parts[1].init_verts + [0, 0, 0.1]
    robot = SimpleNamespace(
        is_built=True,
        solver=SimpleNamespace(n_envs=0),
        links=[SimpleNamespace(name="finger", idx=3, geoms=parts), SimpleNamespace(name="loop", idx=4, geoms=[])],
    )
    model = {"urdf": str(urdf), "model_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest()}
    return robot, model, parts


def test_parts_count_actual_shapes_world_coordinates_and_provenance(sample):
    robot, model, parts = sample
    before = parts[0].init_verts.copy()
    result = collision_geometry_audit(robot, model)
    finger = result["links"]["finger"]
    assert finger["count_matches_urdf"] is True
    assert finger["source_matches_urdf"] is True
    assert result["source_mismatch_links"] == []
    assert finger["source_comparison"]["expected_mesh_path_counts"] == {parts[0].metadata["mesh_path"]: 2}
    assert finger["source_comparison"]["runtime_mesh_path_counts"] == {parts[0].metadata["mesh_path"]: 2}
    assert finger["collision_geom_count"] == result["collision_geom_count"] == 2
    first, second = finger["geometries"]
    assert (first["geom_idx"], second["geom_idx"]) == (10, 11)
    assert (first["vertex_count"], first["face_count"]) == (3, 1)
    assert first["processed_geometry_sha256"] != second["processed_geometry_sha256"]
    assert first["source_reference"]["sha256"] == second["source_reference"]["sha256"]
    assert first["processed_geometry_sha256"] != first["source_reference"]["sha256"]
    assert first["bounds_geom_local_m"] == [[0, 0, 0], [0.02, 0.03, 0.01]]
    assert first["bounds_world_m"] == [[3.97, 5, 6], [4, 5.02, 6.01]]
    assert all(part.reads == 1 for part in parts)
    np.testing.assert_array_equal(parts[0].init_verts, before)
    loop = result["links"]["loop"]
    assert loop["status"] == "available" and loop["collision_geom_count"] == 0
    assert loop["count_matches_urdf"] is True
    assert loop["source_matches_urdf"] is True
    assert result["model"]["matches_manifest_sha256"] is True
    json.dumps(result, allow_nan=False)


def test_collapsed_parts_and_missing_loop_are_not_success(sample):
    robot, model, parts = sample
    robot.links[0].geoms = parts[:1]
    robot.links = robot.links[:1]
    result = collision_geometry_audit(robot, model)
    assert result["count_mismatch_links"] == ["finger"]
    assert result["missing_runtime_links"] == ["loop"]
    assert result["source_mismatch_links"] == ["finger", "loop"]
    loop = result["links"]["loop"]
    assert loop["collision_geom_count"] is None
    assert loop["count_matches_urdf"] is None


def test_disabled_masks_do_not_hide_loaded_geometry(sample):
    robot, model, parts = sample
    robot.links[1].geoms = [Geometry(12)]
    robot.links[1].geoms[0].contype = robot.links[1].geoms[0].conaffinity = 0
    result = collision_geometry_audit(robot, model)
    loop = result["links"]["loop"]
    assert loop["collision_geom_count"] == 1
    assert loop["collision_masks_nonzero_count"] == 0
    assert result["count_mismatch_links"] == ["loop"]


def test_source_file_change_is_not_runtime_shape_change(sample):
    robot, model, parts = sample
    first = collision_geometry_audit(robot, model)["links"]["finger"]["geometries"][0]
    Path(parts[0].metadata["mesh_path"]).write_text("changed since build")
    second = collision_geometry_audit(robot, model)["links"]["finger"]["geometries"][0]
    assert first["processed_geometry_sha256"] == second["processed_geometry_sha256"]
    assert first["source_reference"]["sha256"] != second["source_reference"]["sha256"]
    model["model_sha256"] = "wrong"
    assert collision_geometry_audit(robot, model)["model"]["matches_manifest_sha256"] is False


def test_missing_source_has_explicit_status_but_no_shape_fallback(sample):
    robot, model, parts = sample
    parts[0].metadata["mesh_path"] = "/not/a/real/mesh.obj"
    parts[1].metadata["mesh_path"] = "package://robot/part.obj"
    result = collision_geometry_audit(robot, model)["links"]["finger"]["geometries"]
    assert result[0]["source_reference"]["status"] == "unavailable"
    assert result[1]["source_reference"]["status"] == "unresolved_uri"
    assert all(g["vertex_count"] == 3 and g["source_reference"]["sha256"] is None for g in result)
    audit = collision_geometry_audit(robot, model)
    assert audit["source_mismatch_links"] == ["finger"]
    assert audit["links"]["finger"]["source_comparison"]["unverifiable_runtime_geom_indices"] == [10, 11]


@pytest.fixture
def distinct_same_basename_parts(tmp_path):
    """Actual different OBJ shapes with equal filenames reproduce the cache alias."""
    links = []
    xml_links = []
    paths = {}
    for index, name in enumerate(("narrow3", "wide3", "wide2")):
        mesh = tmp_path / name / "part_000.obj"
        mesh.parent.mkdir()
        vertices = np.array([[0, 0, 0], [0.01 * (index + 1), 0, 0], [0, 0.02, 0], [0, 0, 0.03]])
        faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        mesh.write_text(
            "".join("v " + " ".join(map(str, vertex)) + "\n" for vertex in vertices)
            + "".join("f " + " ".join(str(value + 1) for value in face) + "\n" for face in faces)
        )
        # Read the actual fixture bytes, without importing a mesh or physics backend.
        rows = [line.split() for line in mesh.read_text().splitlines()]
        geom = Geometry(index, mesh)
        geom.init_verts = np.array([[float(v) for v in row[1:]] for row in rows if row[0] == "v"])
        geom.init_faces = np.array([[int(v) - 1 for v in row[1:]] for row in rows if row[0] == "f"])
        geom.n_verts, geom.n_faces = len(geom.init_verts), len(geom.init_faces)
        geom.world = geom.init_verts + [4, 5, 6]
        links.append(SimpleNamespace(name=name, idx=index, geoms=[geom]))
        xml_links.append(
            f'<link name="{name}"><collision><geometry><mesh filename="{name}/part_000.obj"/>'
            "</geometry></collision></link>"
        )
        paths[name] = str(mesh)
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="test">' + "".join(xml_links) + "</robot>")
    robot = SimpleNamespace(is_built=True, solver=SimpleNamespace(n_envs=0), links=links)
    return robot, {"urdf": str(urdf)}, paths


def test_same_basename_replacement_is_detected_with_matching_counts(distinct_same_basename_parts):
    robot, model, paths = distinct_same_basename_parts
    valid = collision_geometry_audit(robot, model)
    assert valid["source_mismatch_links"] == []
    assert len({item["geometries"][0]["processed_geometry_sha256"] for item in valid["links"].values()}) == 3
    replacement = robot.links[2].geoms[0]
    for link in robot.links[:2]:
        geom = link.geoms[0]
        geom.metadata = dict(replacement.metadata)
        geom.init_verts = replacement.init_verts.copy()
        geom.init_faces = replacement.init_faces.copy()
        geom.world = replacement.world.copy()
    audit = collision_geometry_audit(robot, model)
    assert audit["count_mismatch_links"] == []
    assert audit["source_mismatch_links"] == ["narrow3", "wide3"]
    assert len({item["geometries"][0]["processed_geometry_sha256"] for item in audit["links"].values()}) == 1
    for name in ("narrow3", "wide3"):
        comparison = audit["links"][name]["source_comparison"]
        assert comparison["missing_mesh_path_counts"] == {paths[name]: 1}
        assert comparison["unexpected_mesh_path_counts"] == {paths["wide2"]: 1}
        assert comparison["reasons"] == ["mesh_source_path_or_instance_count_mismatch"]
    json.dumps(audit, allow_nan=False)


def test_source_instance_multiplicity_and_runtime_order(distinct_same_basename_parts):
    robot, model, paths = distinct_same_basename_parts
    first, second, _ = robot.links
    urdf = Path(model["urdf"])
    urdf.write_text(
        '<robot name="test"><link name="narrow3">'
        f'<collision><geometry><mesh filename="{paths["narrow3"]}"/></geometry></collision>'
        f'<collision><geometry><mesh filename="{paths["wide3"]}"/></geometry></collision>'
        "</link></robot>"
    )
    first.geoms = [second.geoms[0], first.geoms[0]]
    robot.links = [first]
    assert collision_geometry_audit(robot, model)["source_mismatch_links"] == []
    # Same geometry count and valid source file, but one source occurs twice.
    first.geoms[1].metadata = {"mesh_path": paths["wide3"]}
    audit = collision_geometry_audit(robot, model)
    assert audit["count_mismatch_links"] == []
    assert audit["source_mismatch_links"] == ["narrow3"]
    comparison = audit["links"]["narrow3"]["source_comparison"]
    assert comparison["runtime_mesh_path_counts"] == {paths["wide3"]: 2}
    assert comparison["missing_mesh_path_counts"] == {paths["narrow3"]: 1}
    assert comparison["unexpected_mesh_path_counts"] == {paths["wide3"]: 1}


@pytest.fixture
def equivalent_source_alias(distinct_same_basename_parts, tmp_path):
    pytest.importorskip("trimesh")
    robot, model, paths = distinct_same_basename_parts
    original = Path(paths["narrow3"])
    alias = tmp_path / "cache" / original.name
    alias.parent.mkdir()
    alias.write_text("# Different file bytes; collision vertices and faces are identical.\n" + original.read_text())
    robot.links[0].geoms[0].metadata = {"mesh_path": str(alias)}
    return robot, model, original, alias


def test_different_source_paths_and_bytes_require_exact_topology_evidence(equivalent_source_alias):
    robot, model, original, alias = equivalent_source_alias
    audit = collision_geometry_audit(robot, model)
    assert audit["source_mismatch_links"] == []
    assert audit["source_path_mismatch_links"] == ["narrow3"]
    evidence = audit["resolved_source_aliases"]
    assert len(evidence) == 1
    evidence = evidence[0]
    assert evidence["link"] == "narrow3"
    assert evidence["expected_path"] == str(original)
    assert evidence["runtime_path"] == str(alias)
    assert evidence["expected_file_sha256"] != evidence["runtime_file_sha256"]
    assert evidence["expected_file_sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
    assert evidence["runtime_file_sha256"] == hashlib.sha256(alias.read_bytes()).hexdigest()
    assert evidence["instance_count"] == 1
    assert (evidence["vertex_count"], evidence["face_count"]) == (4, 4)
    geom = audit["links"]["narrow3"]["geometries"][0]
    assert evidence["source_topology_sha256"] == geom["processed_geometry_sha256"]
    comparison = audit["links"]["narrow3"]["source_comparison"]
    assert comparison["missing_mesh_path_counts"] == {str(original): 1}
    assert comparison["unexpected_mesh_path_counts"] == {str(alias): 1}
    assert comparison["unresolved_missing_mesh_path_counts"] == {}
    assert comparison["unresolved_unexpected_mesh_path_counts"] == {}
    json.dumps(audit, allow_nan=False)


@pytest.mark.parametrize("change", ["vertex", "face"])
def test_source_alias_rejects_any_coordinate_or_face_difference(equivalent_source_alias, change):
    robot, model, _, alias = equivalent_source_alias
    content = alias.read_text()
    if change == "vertex":
        content = content.replace("v 0.0 0.0 0.0", "v 0.000000000001 0.0 0.0", 1)
    else:
        content = content.replace("f 1 3 2", "f 1 2 3", 1)
    assert content != alias.read_text()
    alias.write_text(content)
    audit = collision_geometry_audit(robot, model)
    assert audit["source_mismatch_links"] == ["narrow3"]
    assert audit["resolved_source_aliases"] == []


def test_alias_matching_consumes_instances_without_hiding_extra_shape(equivalent_source_alias):
    robot, model, original, alias = equivalent_source_alias
    other = robot.links[1].geoms[0].metadata["mesh_path"]
    urdf = Path(model["urdf"])
    urdf.write_text(
        '<robot name="test"><link name="narrow3">'
        + "".join(
            f'<collision><geometry><mesh filename="{path}"/></geometry></collision>'
            for path in (original, original, other)
        )
        + "</link></robot>"
    )
    geom = robot.links[0].geoms[0]
    robot.links = robot.links[:1]
    robot.links[0].geoms = [deepcopy(geom) for _ in range(3)]
    for index, instance in enumerate(robot.links[0].geoms):
        instance.idx = index
    audit = collision_geometry_audit(robot, model)
    assert audit["count_mismatch_links"] == []
    assert audit["source_mismatch_links"] == ["narrow3"]
    assert len(audit["resolved_source_aliases"]) == 1
    assert audit["resolved_source_aliases"][0]["instance_count"] == 2
    comparison = audit["links"]["narrow3"]["source_comparison"]
    assert comparison["unresolved_missing_mesh_path_counts"] == {other: 1}
    assert comparison["unresolved_unexpected_mesh_path_counts"] == {str(alias): 1}
    # The remaining distinct source must have its own instance, not reuse an alias match.
    robot.links[0].geoms[2].metadata = {"mesh_path": other}
    assert collision_geometry_audit(robot, model)["source_mismatch_links"] == []


def test_unparseable_source_alias_cannot_be_resolved(equivalent_source_alias):
    robot, model, _, alias = equivalent_source_alias
    alias.write_text("not an OBJ mesh")
    audit = collision_geometry_audit(robot, model)
    assert audit["source_mismatch_links"] == ["narrow3"]
    assert audit["resolved_source_aliases"] == []
    assert str(alias) in audit["links"]["narrow3"]["source_comparison"]["source_topology_read_errors"]


def test_exact_paths_do_not_load_mesh_parser(sample, monkeypatch):
    from jiuwensymbiosis.simulators.genesis import collision_audit

    def unexpected(*args, **kwargs):
        raise AssertionError("exact source paths must not invoke topology parsing")

    monkeypatch.setattr(collision_audit, "_source_topology", unexpected)
    robot, model, _ = sample
    assert collision_geometry_audit(robot, model)["source_mismatch_links"] == []


@pytest.mark.parametrize("metadata", [None, {}, {"mesh_path": None}, {"mesh_path": []}, {"mesh_path": ""}])
def test_missing_or_ambiguous_runtime_source_fails_closed(sample, metadata):
    robot, model, parts = sample
    parts[0].metadata = metadata
    audit = collision_geometry_audit(robot, model)
    assert audit["count_mismatch_links"] == []
    assert audit["source_mismatch_links"] == ["finger"]
    assert audit["links"]["finger"]["source_matches_urdf"] is False
    assert audit["links"]["finger"]["source_comparison"]["reasons"]
    json.dumps(audit, allow_nan=False)


def test_same_resolved_path_matches_absolute_and_relative_references(sample):
    robot, model, parts = sample
    parts[0].metadata["mesh_path"] = "./part.obj"
    assert collision_geometry_audit(robot, model)["source_mismatch_links"] == []


def test_unavailable_authored_source_fails_closed_even_if_path_matches(sample):
    robot, model, parts = sample
    Path(parts[0].metadata["mesh_path"]).unlink()
    audit = collision_geometry_audit(robot, model)
    assert audit["source_mismatch_links"] == ["finger"]
    comparison = audit["links"]["finger"]["source_comparison"]
    assert comparison["unavailable_urdf_collision_indices"] == [0, 1]
    assert comparison["unverifiable_runtime_geom_indices"] == [10, 11]


def test_primitive_collision_does_not_require_a_mesh_source(sample):
    robot, model, parts = sample
    Path(model["urdf"]).write_text(
        '<robot name="test"><link name="finger"><collision><geometry><box size="1 2 3"/>'
        '</geometry></collision></link><link name="loop"/></robot>'
    )
    robot.links[0].geoms = parts[:1]
    parts[0].metadata = {}
    parts[0].type = "BOX"
    audit = collision_geometry_audit(robot, model)
    assert audit["source_mismatch_links"] == []
    assert audit["links"]["finger"]["source_comparison"]["expected_non_mesh_collision_count"] == 1


@pytest.mark.parametrize("field,value,match", [("is_built", False, "built"), ("n_envs", 2, "one scene")])
def test_rejects_unbuilt_or_parallel_robot(sample, field, value, match):
    robot, model, _ = sample
    setattr(robot.solver if field == "n_envs" else robot, field, value)
    with pytest.raises(ValueError, match=match):
        collision_geometry_audit(robot, model)


@pytest.mark.parametrize("failure", ["nonfinite", "parallel", "topology", "invalid_face", "read_error"])
def test_bad_runtime_geometry_cannot_fall_back_to_source(sample, failure):
    robot, model, parts = sample
    if failure == "nonfinite":
        parts[0].world[0, 0] = np.nan
    elif failure == "parallel":
        parts[0].world = np.stack([parts[0].world] * 2)
    elif failure == "topology":
        parts[0].world = parts[0].world[:2]
    elif failure == "invalid_face":
        parts[0].init_faces[0, 0] = 99
    else:

        def broken():
            raise RuntimeError("readback failed")

        parts[0].get_verts = broken
    with pytest.raises((ValueError, RuntimeError)):
        collision_geometry_audit(robot, model)


def test_link_selection_and_duplicate_global_indices(sample):
    robot, model, parts = sample
    result = collision_geometry_audit(robot, model, link_names=["loop", "absent"])
    assert result["collision_geom_count"] == 0
    assert result["missing_runtime_links"] == ["absent"]
    assert all(part.reads == 0 for part in parts)
    parts[1].idx = parts[0].idx
    with pytest.raises(ValueError, match="duplicate runtime collision"):
        collision_geometry_audit(robot, model)
