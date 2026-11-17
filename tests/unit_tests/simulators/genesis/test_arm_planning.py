"""Robot-independent motion contracts: smooth limits, stale state and physical stop."""

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from jiuwensymbiosis.api.actions import OPEN_GRIPPER
from jiuwensymbiosis.simulators.common.agent import SimulationManipulationApi, make_session
from jiuwensymbiosis.simulators.common.observation_policy import TRUTH_PROMPT, allowed_tools, history_for
from jiuwensymbiosis.simulators.genesis.arm_collision import CollisionGate, link_pose, snapshot, world_unchanged
from jiuwensymbiosis.simulators.genesis.arm_motion import ArmMotion
from jiuwensymbiosis.simulators.genesis.arm_planning import CuroboPlanner, forward_kinematics
from jiuwensymbiosis.simulators.genesis.arm_trajectory import Trajectory, TrajectoryFollower
from jiuwensymbiosis.simulators.genesis.camera import PreviewSettings
from jiuwensymbiosis.simulators.genesis.planning_profile import load_profile
from jiuwensymbiosis.simulators.genesis.preview import GenesisPreview
from jiuwensymbiosis.simulators.genesis.robot_model import prepare_robot_model


def trajectory(**overrides):
    data = {
        "names": ("slide",),
        "positions": np.array([[0.0], [0.1]]),
        "dt": 2.0,
        "binding": "scene-a",
        "start": {"slide": 0.0, "held": 0.2},
        "velocity_limits": np.array([0.2]),
        "acceleration_limits": np.array([0.3]),
        "jerk_limits": np.array([1.0]),
    }
    return Trajectory(**(data | overrides))


def test_smooth_limits_include_start_stop_and_speed_multiplier():
    t = trajectory()
    assert np.allclose(t.sample([0, 1, 2]).ravel(), [0, 0.05, 0.1])
    for order in (1, 2):
        assert np.max(abs(t._spline.derivative(order)([0, 2]))) < 1e-12
    assert all(
        np.all(v <= limit)
        for v, limit in zip(
            t.derivative_maxima(), (t.velocity_limits, t.acceleration_limits, t.jerk_limits), strict=True
        )
    )
    with pytest.raises(ValueError, match="exceeds"):
        trajectory(dt=2 / 3)
    assert trajectory(dt=4).duration == 4.0
    for bad in (0, -1, True, float("nan")):
        with pytest.raises(ValueError):
            CuroboPlanner.plan(object.__new__(CuroboPlanner), "", {}, {}, "", speed=bad)
    with pytest.raises(TypeError):
        t.start["slide"] = 3
    with pytest.raises(ValueError):
        t.positions[0] = 1


def harness():
    state = {"slide": 0.0, "held": 0.2}
    velocity = {"slide": 0.0, "held": 0.0}
    binding = ["scene-a"]
    writes = []
    f = TrajectoryFollower(
        lambda: (state.copy(), velocity.copy()), lambda target: writes.append(target), lambda: binding[0]
    )
    return f, state, velocity, binding, writes


def test_only_measurement_can_complete_and_cancel_holds_measured_pose():
    f, q, v, _, writes = harness()
    f.start(trajectory())
    for _ in range(40):
        f.tick(0.05)
        q.update(writes[-1])
    assert not f.result["settled"]
    v["slide"] = 0.1
    for _ in range(10):
        f.tick(0.05)
    assert not f.result["settled"]
    f.cancel()
    assert writes[-1]["slide"] == q["slide"]
    v["slide"] = 0.0
    for _ in range(6):
        f.tick(0.05)
    assert f.result["stop_confirmation"] == "measured" and not f.result["ok"]


@pytest.mark.parametrize(
    "change",
    [
        lambda q, v, b: q.update(held=0.3),
        lambda q, v, b: v.update(slide=float("nan")),
        lambda q, v, b: b.__setitem__(0, "scene-b"),
    ],
)
def test_invalid_start_writes_nothing(change):
    f, q, v, b, writes = harness()
    change(q, v, b)
    with pytest.raises(ValueError):
        f.start(trajectory())
    assert not writes


def test_tracking_failure_and_unknown_stop_are_terminal():
    f, q, v, _, writes = harness()
    f.start(trajectory())
    q["slide"] = 0.2
    v["slide"] = 0.1
    f.tick(0.05)
    assert f.phase == "braking" and writes[-1]["slide"] == 0.2
    for _ in range(82):
        f.tick(0.05)
    assert f.terminal and f.result["stop_confirmation"] == "unknown"
    with pytest.raises(RuntimeError):
        f.start(trajectory())


def test_readback_loss_preserves_last_target_without_claiming_stopped():
    f, q, _, _, writes = harness()
    f.start(trajectory())
    q["slide"] = float("nan")
    f.tick(0.05)
    assert writes[-1] == {"slide": 0.0} and f.terminal and not f.result["settled"]


def model(tmp_path):
    source = tmp_path / "source.urdf"
    source.write_text("""<robot name="slider"><link name="root"/><link name="tip"/><link name="finger"/><link name="fixed_tip"/>
<joint name="fixed" type="fixed"><parent link="root"/><child link="fixed_tip"/><axis xyz="0 0 0"/></joint>
<joint name="slide" type="prismatic"><parent link="root"/><child link="tip"/><axis xyz="1 0 0"/><limit lower="0" upper="1" effort="10" velocity="1"/></joint>
<joint name="follower" type="prismatic"><parent link="tip"/><child link="finger"/><axis xyz="0 1 0"/><mimic joint="slide" multiplier="-1" offset=".1"/></joint></robot>""")
    manifest = prepare_robot_model(source, tmp_path / "derived")
    profile = {
        "schema_version": 1,
        "base_link": "root",
        "groups": {"tool": {"joints": ["slide"], "tools": ["tip"]}},
        "limits": {"slide": {"velocity": 1.0, "acceleration": 2.0, "jerk": 10.0, "effort": 10.0}},
        "collision_spheres": {"tip": [{"center": [0, 0, 0], "radius": 0.1}]},
    }
    return manifest, profile


@pytest.mark.parametrize("bad", ["name", "limit", "tool", "mimic", "hash", "uncontrolled"])
def test_bad_profile_or_changed_model_rejected_before_gpu(tmp_path, bad):
    manifest, profile = model(tmp_path)
    if bad == "uncontrolled":
        profile["uncontrolled_support_joints"] = ["slide"]
    if bad == "name":
        profile["groups"]["tool"]["joints"] = ["invented"]
    if bad == "limit":
        profile["limits"]["slide"]["velocity"] = 0
    if bad == "tool":
        profile["groups"]["tool"]["tools"] = ["invented"]
    if bad == "mimic":
        profile["groups"]["tool"]["joints"] = ["follower"]
    if bad == "hash":
        Path(manifest["urdf"]).write_text("<robot/>")
    with pytest.raises(ValueError):
        CuroboPlanner(manifest, profile, tmp_path / "plans")


@pytest.fixture
def collision_model(tmp_path):
    pytest.importorskip("fcl")
    import trimesh

    manifest, profile = model(tmp_path)
    box = trimesh.creation.box(extents=[0.04] * 3)
    return (
        ET.parse(manifest["urdf"]).getroot(),
        profile,
        {"vertices": box.vertices, "faces": box.faces, "pose": np.eye(4), "convex": True},
    )


def test_collision_gate_checks_loaded_shapes_between_clear_endpoints(collision_model):
    root, profile, geometry = collision_model
    obstacle = np.eye(4)
    obstacle[0, 3] = 0.05
    captured = {
        "geometries": {
            1: dict(geometry, link="tip"),
            2: dict(geometry, link=None, pose=obstacle),
        },
        "pairs": [(1, 2)],
    }
    gate = CollisionGate(root, profile, captured)
    gate.check_state({"slide": 0.0})
    gate.check_state({"slide": 0.1})
    with pytest.raises(ValueError, match="physical_geometry_collision"):
        gate.check(trajectory())
    profile["collision_spheres"] = {}
    assert gate.coverage()["tip"][0]["uncovered"] == len(geometry["vertices"]) + len(geometry["faces"])


def test_payload_contact_permission_does_not_hide_world_collision(collision_model):
    root, profile, geometry = collision_model
    captured = {
        "geometries": {
            1: dict(geometry, link="tip", payload=True, contact_links=["tip"]),
            2: dict(geometry, link="tip"),
            3: dict(geometry, link=None),
        },
        "pairs": [(1, 2)],
    }
    gate = CollisionGate(root, profile, captured)
    gate.check_state({"slide": 0.0})
    captured["pairs"].append((1, 3))
    with pytest.raises(ValueError, match="physical_geometry_collision"):
        gate.check_state({"slide": 0.0})


def test_world_binding_rejects_velocity_before_displacement():
    link = SimpleNamespace(idx=8, get_vel=lambda: np.array([[0.1, 0.0, 0.0]]), get_ang=lambda: np.zeros((1, 3)))
    entity = SimpleNamespace(solver=SimpleNamespace(entities=[SimpleNamespace(links=[link])]))
    assert not world_unchanged(entity, {"dynamic": {8: np.eye(4)}})


@pytest.fixture
def transport(monkeypatch, tmp_path):
    def build(*, configured=True, stopped=True):
        preview = GenesisPreview(
            state_dir=tmp_path,
            settings=PreviewSettings(robot_mode="moz1_wheels", planning_profile="/profile.json" if configured else ""),
        )
        state = {
            "state": "ready",
            "measured_at": time.monotonic(),
            "scene": {"instance_id": "scene", "arm_planning_ready": configured, "effectors_ready": configured},
            "robot": {"control_mode": "wheel_velocity_physics"},
            "base_motion": {},
            "effectors": {"arbitrary_name": {"holding_payload": None}} if configured else {},
        }
        requests = []

        def worker():
            request = preview._motion_request
            if request:
                requests.append(request.copy())
                state["base_motion"] = {
                    "id": request["id"],
                    "status": "completed",
                    "ok": stopped,
                    "settled": stopped,
                    "stop_confirmation": "measured" if stopped else "unknown",
                    "holding_payload": None,
                    "grasp_confirmed": False,
                    "contact_confirmed": True,
                }

        monkeypatch.setattr(preview, "status", lambda: state)
        monkeypatch.setattr(preview, "_send_control", worker)
        session = make_session(preview, observation_mode="ground_truth")
        session.strict_capabilities = True
        session.connect()
        return session, requests, state

    return build


def test_named_gripper_uses_shared_session_transport(transport):
    meta = SimulationManipulationApi.open_gripper.__tool_meta__
    assert not meta.input_params.get("required")
    assert OPEN_GRIPPER.param_schema["effector"]["type"] == "string"
    session, requests, state = transport()
    try:
        result = session.api.close_gripper(effector="arbitrary_name")
        assert result["ok"] and not result["grasp_confirmed"]
        assert result["holding_payload"] is None
        assert requests[-1]["kind"] == "arm" and requests[-1]["effector"] == "arbitrary_name"
        assert requests[-1]["opened"] is False
        assert session.env.holding_payload is None
        state["effectors"]["other"] = {"holding_payload": True}
        assert session.env.holding_payload is True
        count = len(requests)
        for selector in (None, "unknown"):
            with pytest.raises(ValueError):
                session.api.open_gripper(effector=selector)
        assert len(requests) == count
        state["measured_at"] = 0
        assert session.env.holding_payload is None
        with pytest.raises(RuntimeError, match="stale"):
            session.api.close_gripper(effector="arbitrary_name")
    finally:
        session.disconnect()


def test_profile_external_source_is_hashed_and_limits_are_explicit(tmp_path, monkeypatch):
    manifest, profile = model(tmp_path)
    source = tmp_path / "spheres.json"
    kin = {"collision_spheres": profile.pop("collision_spheres"), "self_collision_ignore": {}}
    source.write_text(json.dumps({"robot_cfg": {"kinematics": kin}}))
    profile["collision_source"] = {"format": "curobo", "path": "${TEST_SPHERES}", "ignore_rationale": "No exclusions"}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    monkeypatch.setenv("TEST_SPHERES", str(source))
    loaded = load_profile(path)
    planner = CuroboPlanner(manifest, loaded, tmp_path / "valid")
    assert np.allclose(forward_kinematics(planner.root, {"slide": 0.3})["finger"][:3, 3], [0.3, -0.2, 0])
    assert np.array_equal(forward_kinematics(planner.root, {})["fixed_tip"], np.eye(4))
    planner.close()
    source.write_text("{}")
    with pytest.raises(ValueError, match="source changed"):
        CuroboPlanner(manifest, loaded, tmp_path / "invalid")


def test_arm_transport_uses_existing_owner_and_unknown_stop_is_terminal(transport):
    session, requests, _ = transport(stopped=False)
    assert "motion.tool_pose" in session.env.capabilities
    assert "move_tool_poses" not in allowed_tools("ground_truth", True)
    assert "move_tool_poses" in allowed_tools("ground_truth", True, planning_enabled=True)
    assert all(name in TRUTH_PROMPT for name in ("move_joint", "move_tool_poses", "open_gripper", "close_gripper"))
    old = {"role": "user", "content": "old", "observation_policy": "simulation-ground-truth-v1"}
    assert not history_for("ground_truth", [old])
    result = session.api.move_tool_poses("arm", {"tip": [0, 0, 0, 1, 0, 0, 0]}, 1.5)
    assert result["stop_confirmation"] == "unknown" and requests[0]["kind"] == "arm"
    assert requests[0]["poses"] == {"tip": [0, 0, 0, 1, 0, 0, 0]}
    with pytest.raises(RuntimeError):
        session.api.move_joint({"slide": 0.1})
    with pytest.raises(RuntimeError):
        session.env.disconnect()


def test_unconfigured_scene_keeps_strict_capability_contract(transport):
    session, _, _ = transport(configured=False)
    assert "motion.tool_pose" not in session.api.capabilities
    assert "motion.joint" not in session.api.capabilities
    session.disconnect()


def test_status_summary_never_copies_large_trajectory_or_unbounded_error():
    motion = object.__new__(ArmMotion)
    motion.result = {
        "id": "request",
        "status": "completed",
        "ok": False,
        "settled": True,
        "reason": "x" * 100000,
        "positions": [[0.0] * 14] * 10000,
        "timing": {"command_derivative_maxima": [[1.0] * 14] * 3},
    }
    result = motion.snapshot()
    assert "positions" not in result and "timing" not in result
    assert len(json.dumps(result).encode()) < 1024
    assert len(motion.result["reason"]) == 100000


@pytest.mark.parametrize("batched", [False, True])
def test_shared_link_pose_preserves_frame_and_rejects_nonfinite_readback(batched):
    position, quaternion = np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0, 0.0])
    link = SimpleNamespace(
        get_pos=lambda: position[None] if batched else position,
        get_quat=lambda: quaternion[None] if batched else quaternion,
    )
    assert np.allclose(link_pose(link)[:3, 3], position)
    assert np.allclose(link_pose(link)[:3, :3], np.eye(3))
    position[0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        link_pose(link)


def test_snapshot_rejects_multilink_payload_before_readback():
    body = SimpleNamespace(idx=4, links=[SimpleNamespace(), SimpleNamespace()])
    entity = SimpleNamespace(solver=SimpleNamespace(entities=[body]))
    with pytest.raises(ValueError, match="single-link payload"):
        snapshot(entity, {}, {4: {}})


@pytest.mark.parametrize("poses", [{}, {"unknown": [0] * 7}])
def test_tool_names_are_rejected_before_snapshot_or_planner(poses):
    motion = object.__new__(ArmMotion)
    motion.follower, motion.process = Mock(terminal=False), None
    motion.runtime, motion.stopped, motion.read = Mock(), lambda: True, lambda: ({}, {})
    motion.runtime.motion.active = False
    motion.profile = {"groups": {"arm": {"joints": ["joint"], "tools": ["tool"]}}}
    motion.base_pose = Mock()
    with pytest.raises(ValueError, match="all group tool poses"):
        motion.start({"group": "arm", "poses": poses})
    motion.base_pose.assert_not_called()


def test_tick_readback_or_artifact_failure_stops_and_reports_unknown():
    motion = object.__new__(ArmMotion)
    motion.follower, motion.effectors, motion.runtime = Mock(), Mock(), Mock()
    motion.result = {"ok": True, "settled": True}
    motion._tick = Mock(side_effect=OSError("readback or disk unavailable"))
    motion.cancel = Mock()
    motion.tick(0.01)
    motion.cancel.assert_called_once()
    motion.effectors.invalidate.assert_not_called()
    assert motion.follower.terminal and motion.phase == "completed"
    assert not motion.result["ok"] and not motion.result["settled"]
    assert motion.result["stop_confirmation"] == "unknown"
