"""Shared model admission: structural variants, provenance and loaded defects."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jiuwensymbiosis.simulators.genesis.collision_audit import audit_loaded_model
from jiuwensymbiosis.simulators.genesis.omni_base import Moz1Base
from jiuwensymbiosis.simulators.genesis.robot_model import prepare_robot_model, verify_model_files


@pytest.fixture
def model(tmp_path):
    source = tmp_path / "source.urdf"
    source.write_text("""<robot name="other_robot"><link name="root"/>
<link name="slider"><collision><origin xyz=".1 .2 .3" rpy="0 0 .4"/>
<geometry><box size=".1 .2 .3"/></geometry></collision>
<collision><origin xyz="0 0 1"/><geometry><box size=".3 .2 .1"/></geometry></collision></link>
<link name="follower"/><joint name="drive" type="prismatic"><parent link="root"/><child link="slider"/>
<axis xyz="1 0 0"/><limit lower="-.2" upper=".2" effort="inf" velocity="1"/></joint>
<joint name="follow" type="revolute"><parent link="slider"/><child link="follower"/>
<mimic joint="drive" multiplier="-1" offset=".1"/></joint></robot>""")
    return prepare_robot_model(source, tmp_path / "derived")


def test_structure_and_source_are_preserved(model):
    assert [j["type"] for j in model["joints"]] == ["prismatic", "revolute"]
    assert model["joints"][0]["limit"]["effort"] == "+infinity"
    assert model["joints"][1]["mimic"]["multiplier"] == -1
    assert len(model["links"][1]["collisions"]) == 2
    assert model["collision_preparation"] is None
    verify_model_files(model)
    assert json.loads(json.dumps(model, allow_nan=False)) == model


@pytest.mark.parametrize("field", ["source", "urdf"])
def test_file_mutation_rejected(model, field):
    path = Path(model["source"]["path"] if field == "source" else model["urdf"])
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        verify_model_files(model)


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda s: s.replace('joint="drive"', 'joint="follow"'), "mimic"),
        (lambda s: s.replace('type="prismatic"', 'type="floating"'), "unsupported"),
        (lambda s: s.replace('size=".1 .2 .3"', 'size="0 .2 .3"'), "numeric"),
        (lambda s: s.replace('xyz="1 0 0"', 'xyz="0 0 0"'), "zero"),
        (lambda s: s.replace('child link="follower"', 'child link="slider"'), "parents"),
        (lambda s: s.replace('<link name="root"/>', '<link name="root"/><link name="root"/>'), "duplicate"),
    ],
)
def test_bad_structure_fails_before_publication(model, tmp_path, change, match):
    path = Path(model["source"]["path"])
    path.write_text(change(path.read_text()))
    with pytest.raises(ValueError, match=match):
        prepare_robot_model(path, tmp_path / "rejected")
    assert not (tmp_path / "rejected/model.json").exists()


def test_package_resolution_and_dependency_hash(tmp_path):
    pytest.importorskip("trimesh")
    package = tmp_path / "pkg"
    package.mkdir()
    mesh = package / "mesh.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    source = tmp_path / "source.urdf"
    source.write_text(
        '<robot name="r"><link name="root"><collision><geometry><mesh filename="package://p/mesh.obj" scale="2 3 4"/></geometry></collision></link></robot>'
    )
    with pytest.raises(ValueError, match="unresolved package"):
        prepare_robot_model(source, tmp_path / "derived")
    model = prepare_robot_model(source, tmp_path / "derived", packages={"p": package})
    assert model["links"][0]["collisions"][0]["scale"] == [2, 3, 4]
    mesh.write_text("changed")
    with pytest.raises(ValueError, match="resource changed"):
        verify_model_files(model)


def built_robot(model):
    import trimesh

    geoms = []
    for idx, collision in enumerate(model["links"][1]["collisions"]):
        shape = trimesh.creation.box(extents=collision["size"])
        origin = collision["origin"]
        q = Rotation.from_euler("xyz", origin["rpy"]).as_quat(scalar_first=True)
        geoms.append(
            SimpleNamespace(
                idx=idx,
                type="BOX",
                is_convex=True,
                contype=1,
                conaffinity=1,
                init_verts=shape.vertices,
                init_faces=shape.faces,
                init_pos=np.asarray(origin["xyz"]),
                init_quat=q,
                n_verts=len(shape.vertices),
                n_faces=len(shape.faces),
                metadata={},
                get_verts=lambda v=shape.vertices: v[None],
            )
        )
    return SimpleNamespace(
        is_built=True,
        solver=SimpleNamespace(
            n_envs=1,
            _enable_collision=True,
            _enable_self_collision=True,
            _enable_adjacent_collision=False,
            _enable_neutral_collision=False,
            collider=SimpleNamespace(_valid_collision_pairs=np.array([[0, 1]])),
        ),
        links=[
            SimpleNamespace(name=link["name"], idx=i, geoms=geoms if i == 1 else [])
            for i, link in enumerate(model["links"])
        ],
    )


@pytest.mark.parametrize("defect", [None, "translation", "scale", "rotation", "mask", "readback", "source"])
def test_loaded_admission_persists_evidence_and_rejects_defects(model, tmp_path, defect):
    pytest.importorskip("trimesh")
    robot = built_robot(model)
    shape = robot.links[1].geoms[0]
    if defect == "translation":
        shape.init_pos += [0.01, 0, 0]
    if defect == "scale":
        shape.init_verts = shape.init_verts * 1.1
    if defect == "rotation":
        shape.init_quat = np.array([1.0, 0, 0, 0])
    if defect == "mask":
        shape.contype = shape.conaffinity = 0
    if defect == "readback":
        shape.get_verts = Mock(side_effect=RuntimeError("readback"))
    if defect == "source":
        Path(model["urdf"]).write_text("changed")
    report = tmp_path / "audit.json"
    if defect:
        with pytest.raises(ValueError, match="admission failed"):
            audit_loaded_model(robot, model, report)
        assert json.loads(report.read_text())["status"] == "error"
    else:
        robot.links[1].geoms.reverse()  # importer ordering is irrelevant
        assert audit_loaded_model(robot, model, report)["status"] == "pass"


def test_failed_admission_precedes_all_motor_writes(model, tmp_path):
    robot = Moz1Base.__new__(Moz1Base)
    robot.entity, robot.model_manifest, robot.path = Mock(), model, Path(model["urdf"])
    robot.entity.is_built = False
    robot.upper_body = Mock()
    with pytest.raises(ValueError, match="admission failed"):
        robot.build()
    robot.entity.set_dofs_kp.assert_not_called()
    robot.entity.set_dofs_position.assert_not_called()
    robot.upper_body.build.assert_not_called()
