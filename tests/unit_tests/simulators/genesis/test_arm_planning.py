"""Robot-independent motion contracts: smooth limits, stale state and physical stop."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from jiuwensymbiosis.simulators.genesis.arm_planning import CuroboPlanner, forward_kinematics
from jiuwensymbiosis.simulators.genesis.arm_trajectory import Trajectory, TrajectoryFollower
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
        t.scaled(3.0)
    assert t.scaled(0.5).duration == 4.0
    for bad in (0, -1, True, float("nan")):
        with pytest.raises(ValueError):
            t.scaled(bad)
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
    source.write_text("""<robot name="slider"><link name="root"/><link name="tip"/><link name="finger"/>
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


def test_one_prismatic_joint_and_mimic_require_no_robot_name(tmp_path):
    manifest, profile = model(tmp_path)
    p = CuroboPlanner(manifest, profile, tmp_path / "plans")
    frames = forward_kinematics(p.root, {"slide": 0.3})
    assert np.allclose(frames["finger"][:3, 3], [0.3, -0.2, 0])
    p.close()


@pytest.mark.parametrize("bad", ["name", "limit", "tool", "mimic", "hash"])
def test_bad_profile_or_changed_model_rejected_before_gpu(tmp_path, bad):
    manifest, profile = model(tmp_path)
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


def test_fixed_zero_axis_is_not_normalized():
    root = ET.fromstring(
        '<robot><link name="a"/><link name="b"/><joint name="j" type="fixed"><parent link="a"/><child link="b"/><axis xyz="0 0 0"/></joint></robot>'
    )
    assert np.array_equal(forward_kinematics(root, {})["b"], np.eye(4))


@pytest.mark.parametrize("position", [-0.0000466, 1.0000466])
@pytest.mark.parametrize("goal", [{"targets": {"slide": 0.5}}, {"poses": {"tip": [0.5, 0, 0, 1, 0, 0, 0]}}])
def test_measured_start_outside_limits_is_reported_before_backend(tmp_path, monkeypatch, position, goal):
    manifest, profile = model(tmp_path)
    planner = CuroboPlanner(manifest, profile, tmp_path / "plans")

    def unexpected_backend(*args):
        pytest.fail("an invalid measured start reached cuRobo")

    monkeypatch.setattr(planner, "_prepare", unexpected_backend)
    state = {"slide": position, "follower": 0.1 - position}
    original = state.copy()
    with pytest.raises(ValueError, match="start_joint_limit: slide"):
        planner.plan("tool", state, {}, "snapshot", **goal)
    assert state == original  # Never clamp measured state to manufacture feasibility.


def test_collision_gate_checks_loaded_shapes_between_clear_endpoints(tmp_path):
    pytest.importorskip("fcl")
    import trimesh

    from jiuwensymbiosis.simulators.genesis.arm_collision import CollisionGate

    manifest, profile = model(tmp_path)
    root = ET.parse(manifest["urdf"]).getroot()
    box = trimesh.creation.box(extents=[0.04, 0.04, 0.04])
    obstacle = np.eye(4)
    obstacle[0, 3] = 0.05
    captured = {
        "geometries": {
            1: {"link": "tip", "vertices": box.vertices, "faces": box.faces, "pose": np.eye(4), "convex": True},
            2: {"link": None, "vertices": box.vertices, "faces": box.faces, "pose": obstacle, "convex": True},
        },
        "pairs": [(1, 2)],
    }
    gate = CollisionGate(root, profile, captured)
    gate.check_state({"slide": 0.0})
    gate.check_state({"slide": 0.1})
    with pytest.raises(ValueError, match="physical_geometry_collision"):
        gate.check(trajectory())
    profile["collision_spheres"] = {}
    assert gate.coverage()["tip"][0]["uncovered"] == len(box.vertices) + len(box.faces)


def test_payload_contact_permission_does_not_hide_world_collision(tmp_path):
    pytest.importorskip("fcl")
    import trimesh

    from jiuwensymbiosis.simulators.genesis.arm_collision import CollisionGate

    manifest, profile = model(tmp_path)
    root = ET.parse(manifest["urdf"]).getroot()
    box = trimesh.creation.box(extents=[0.04] * 3)
    geometry = {"vertices": box.vertices, "faces": box.faces, "pose": np.eye(4), "convex": True}
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
    from types import SimpleNamespace

    from jiuwensymbiosis.simulators.genesis.arm_collision import world_unchanged

    link = SimpleNamespace(idx=8, get_vel=lambda: np.array([[0.1, 0.0, 0.0]]), get_ang=lambda: np.zeros((1, 3)))
    entity = SimpleNamespace(solver=SimpleNamespace(entities=[SimpleNamespace(links=[link])]))
    assert not world_unchanged(entity, {"dynamic": {8: np.eye(4)}})


def test_named_gripper_uses_shared_session_transport(monkeypatch, tmp_path):
    import time

    from jiuwensymbiosis.simulators.genesis.agent import make_session
    from jiuwensymbiosis.simulators.genesis.camera import PreviewSettings
    from jiuwensymbiosis.simulators.genesis.preview import GenesisPreview

    preview = GenesisPreview(
        state_dir=tmp_path, settings=PreviewSettings(robot_mode="moz1_wheels", planning_profile="/profile.json")
    )
    state = {
        "state": "ready",
        "measured_at": time.monotonic(),
        "scene": {"instance_id": "scene", "arm_planning_ready": True, "effectors_ready": True},
        "robot": {"control_mode": "wheel_velocity_physics"},
        "base_motion": {},
        "effectors": {"arbitrary_name": {"holding_payload": None}},
    }
    monkeypatch.setattr(preview, "status", lambda: state)
    requests = []

    def worker():
        request = preview._motion_request
        if request:
            requests.append(request.copy())
            state["base_motion"] = {
                "id": request["id"],
                "status": "completed",
                "ok": True,
                "settled": True,
                "stop_confirmation": "measured",
                "holding_payload": None,
                "grasp_confirmed": False,
                "contact_confirmed": True,
            }

    monkeypatch.setattr(preview, "_send_control", worker)
    session = make_session(preview, observation_mode="ground_truth")
    session.strict_capabilities = True
    session.connect()
    try:
        result = session.api.set_gripper("arbitrary_name", False)
        assert result["ok"] and not result["grasp_confirmed"]
        assert result["holding_payload"] is None
        assert requests[-1]["kind"] == "arm" and requests[-1]["effector"] == "arbitrary_name"
        assert requests[-1]["opened"] is False
        assert session.env.holding_payload is None
    finally:
        session.disconnect()


def test_uncontrolled_support_exception_cannot_hide_arm_motion(tmp_path):
    manifest, profile = model(tmp_path)
    profile["uncontrolled_support_joints"] = ["slide"]
    with pytest.raises(ValueError, match="uncontrolled"):
        CuroboPlanner(manifest, profile, tmp_path / "plans")


def test_profile_external_source_is_hashed_and_limits_are_explicit(tmp_path, monkeypatch):
    import json

    from jiuwensymbiosis.simulators.genesis.planning_profile import load_profile

    manifest, profile = model(tmp_path)
    source = tmp_path / "spheres.json"
    source.write_text(
        json.dumps(
            {
                "robot_cfg": {
                    "kinematics": {"collision_spheres": profile.pop("collision_spheres"), "self_collision_ignore": {}}
                }
            }
        )
    )
    profile["collision_source"] = {"format": "curobo", "path": "${TEST_SPHERES}", "ignore_rationale": "No exclusions"}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    monkeypatch.setenv("TEST_SPHERES", str(source))
    loaded = load_profile(path)
    CuroboPlanner(manifest, loaded, tmp_path / "valid").close()
    source.write_text("{}")
    with pytest.raises(ValueError, match="source changed"):
        CuroboPlanner(manifest, loaded, tmp_path / "invalid")


def test_arm_transport_uses_existing_owner_and_unknown_stop_is_terminal(monkeypatch, tmp_path):
    import time

    from jiuwensymbiosis.simulators.genesis.agent import make_session
    from jiuwensymbiosis.simulators.genesis.camera import PreviewSettings
    from jiuwensymbiosis.simulators.genesis.observation_policy import allowed_tools
    from jiuwensymbiosis.simulators.genesis.preview import GenesisPreview

    preview = GenesisPreview(
        state_dir=tmp_path, settings=PreviewSettings(robot_mode="moz1_wheels", planning_profile="/configured.json")
    )
    state = {
        "state": "ready",
        "measured_at": time.monotonic(),
        "scene": {"instance_id": "scene", "arm_planning_ready": True},
        "robot": {"control_mode": "wheel_velocity_physics"},
        "base_motion": {},
    }
    monkeypatch.setattr(preview, "status", lambda: state)
    requests = []

    def worker():
        request = preview._motion_request
        if request:
            requests.append(request.copy())
            state["base_motion"] = {
                "id": request["id"],
                "status": "completed",
                "ok": False,
                "goal_reached": False,
                "settled": False,
                "stop_confirmation": "unknown",
                "plan_success": False,
                "reason": "test_stop_unknown",
            }

    monkeypatch.setattr(preview, "_send_control", worker)
    session = make_session(preview, observation_mode="ground_truth")
    session.strict_capabilities = True
    session.connect()
    assert "motion.tool_pose" in session.env.capabilities
    assert "move_tool_poses" not in allowed_tools("ground_truth", True)
    assert "move_tool_poses" in allowed_tools("ground_truth", True, planning_enabled=True)
    result = session.api.move_tool_poses("arm", {"tip": [0, 0, 0, 1, 0, 0, 0]}, 1.5)
    assert result["stop_confirmation"] == "unknown" and requests[0]["kind"] == "arm"
    assert requests[0]["poses"] == {"tip": [0, 0, 0, 1, 0, 0, 0]}
    with pytest.raises(RuntimeError):
        session.api.move_joint({"slide": 0.1})
    with pytest.raises(RuntimeError):
        session.env.disconnect()


def test_unconfigured_scene_keeps_strict_capability_contract(tmp_path, monkeypatch):
    import time

    from jiuwensymbiosis.simulators.genesis.agent import make_session
    from jiuwensymbiosis.simulators.genesis.preview import GenesisPreview

    p = GenesisPreview(state_dir=tmp_path)
    state = {
        "state": "ready",
        "measured_at": time.monotonic(),
        "scene": {"instance_id": "scene"},
        "robot": {"control_mode": "wheel_velocity_physics"},
        "base_motion": {},
    }
    monkeypatch.setattr(p, "status", lambda: state)
    monkeypatch.setattr(p, "_send_control", lambda: None)
    session = make_session(p, observation_mode="ground_truth")
    session.strict_capabilities = True
    session.connect()
    assert "motion.tool_pose" not in session.api.capabilities
    assert "motion.joint" not in session.api.capabilities
    session.disconnect()


def test_status_summary_never_copies_large_trajectory_or_unbounded_error():
    import json

    from jiuwensymbiosis.simulators.genesis.arm_motion import ArmMotion

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
