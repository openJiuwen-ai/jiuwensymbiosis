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
    # ``trimesh`` is an optional simulation extra; keep the whole helper (and
    # every test that builds a robot through it) skipped when it is absent.
    trimesh = pytest.importorskip("trimesh")

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


def test_processed_reordering_cannot_borrow_a_sphere_tolerance(model, monkeypatch):
    from jiuwensymbiosis.simulators.genesis import collision_audit

    robot = built_robot(model)
    # Importer order differs from authored order. The first processed geometry
    # is a box; it must not inherit the authored sphere's radius tolerance.
    model["links"][1]["collisions"][0] = {
        "type": "sphere",
        "radius": 1.0,
        "origin": {"xyz": [0, 0, 0], "rpy": [0, 0, 0]},
    }

    def reprocess(robot, directions):
        values = []
        for geom in robot.links[1].geoms:
            rotation = Rotation.from_quat(geom.init_quat, scalar_first=True)
            vertices = rotation.apply(geom.init_verts) + geom.init_pos
            values.append(np.max(vertices @ directions.T, axis=0))
        values[1] -= directions @ np.array([0.001, 0, 0])
        return {"slider": values[::-1]}

    monkeypatch.setattr(collision_audit, "_processed_expectations", reprocess)
    result = collision_audit._spatial_comparison(robot, model, {"decimate": True})
    box = next(row for row in result if row["geom_idx"] == 1)
    assert box["status"] == "error"
    assert box["tolerance_m"] == 1e-5
    assert "processed_index" in box and "collision_index" not in box


def test_empty_model_is_not_physical_admission(tmp_path):
    source = tmp_path / "empty.urdf"
    source.write_text('<robot name="empty"><link name="root"/></robot>')
    model = prepare_robot_model(source, tmp_path / "derived")
    robot = SimpleNamespace(
        is_built=True, solver=SimpleNamespace(n_envs=1), links=[SimpleNamespace(name="root", idx=0, geoms=[])]
    )
    destination = tmp_path / "audit.json"
    with pytest.raises(ValueError):
        audit_loaded_model(robot, model, destination)
    assert any(row["check"] == "no_collision_geometry" for row in json.loads(destination.read_text())["errors"])


def test_processed_expectations_reparse_and_transform(monkeypatch):
    import sys

    trimesh = pytest.importorskip("trimesh")

    from jiuwensymbiosis.simulators.genesis.collision_audit import _processed_expectations

    mesh = trimesh.creation.box(extents=[0.2, 0.4, 0.6])
    rotation = Rotation.from_euler("z", 0.4)
    collision = {
        "contype": 1,
        "conaffinity": 0,
        "mesh": SimpleNamespace(verts=mesh.vertices),
        "pos": [0.1, 0.2, 0.3],
        "quat": rotation.as_quat(scalar_first=True),
    }
    visual = {"contype": 0, "conaffinity": 0}
    parse = Mock(return_value=([{"name": "root"}], [], [[visual, collision]], None))
    process = Mock(return_value=[collision])
    monkeypatch.setitem(sys.modules, "genesis", SimpleNamespace(surfaces=SimpleNamespace(Default=lambda: "surface")))
    monkeypatch.setitem(
        sys.modules,
        "genesis.utils",
        SimpleNamespace(
            mesh=SimpleNamespace(postprocess_collision_geoms=process), mjcf=SimpleNamespace(parse_xml=parse)
        ),
    )
    morph = SimpleNamespace(
        convexify=True,
        watertighten=5,
        decompose_robot_error_threshold=float("inf"),
        decimate=True,
        decimate_face_num=100,
        decimate_aggressiveness=2,
        coacd_options=None,
        model_copy=Mock(return_value="copy"),
    )
    result = _processed_expectations(SimpleNamespace(morph=morph), np.eye(3))
    expected = (rotation.apply(mesh.vertices) + [0.1, 0.2, 0.3]).max(axis=0)
    assert result["root"][0] == pytest.approx(expected)
    morph.model_copy.assert_called_once_with(update={"visualization": False})
    parse.assert_called_once_with("copy", "surface")
    assert process.call_args.args == ([collision], True, 100, 2, True, float("inf"), None, 5)


def test_cli_destroys_scene_and_genesis_after_build_failure(model, monkeypatch, tmp_path):
    import sys

    from jiuwensymbiosis.simulators.genesis.robot_model import main

    scene = Mock()
    scene.build.side_effect = RuntimeError("build failure")
    gs = SimpleNamespace(
        init=Mock(),
        destroy=Mock(),
        cpu="cpu",
        Scene=Mock(return_value=scene),
        options=SimpleNamespace(RigidOptions=Mock()),
        morphs=SimpleNamespace(URDF=Mock()),
    )
    monkeypatch.setitem(sys.modules, "genesis", gs)
    monkeypatch.setattr(
        sys, "argv", ["audit", model["source"]["path"], "--output", str(tmp_path / "cli"), "--load-audit"]
    )
    with pytest.raises(RuntimeError, match="build failure"):
        main()
    scene.destroy.assert_called_once()
    gs.destroy.assert_called_once()


def test_unavailable_readback_retains_cause_and_refuses_admission(model):
    robot = built_robot(model)
    del robot.solver.collider
    destination = Path(model["urdf"]).parent / "unavailable.json"
    with pytest.raises(ValueError) as failure:
        audit_loaded_model(robot, model, destination)
    assert isinstance(failure.value.__cause__, AttributeError)
    report = json.loads(destination.read_text())
    assert report["status"] == "unverifiable"
    assert report["unverifiable"]


def test_aligned_frames_are_rejected_with_actionable_error(model):
    robot = built_robot(model)
    robot.links[0].aligned = True
    with pytest.raises(ValueError) as failure:
        audit_loaded_model(robot, model, Path(model["urdf"]).parent / "aligned.json")
    assert "align=False" in str(failure.value.__cause__)
