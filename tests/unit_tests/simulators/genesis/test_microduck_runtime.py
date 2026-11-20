"""Native policy inputs retain validated command magnitudes without optional SDKs."""

import importlib.util
import math
import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest


@pytest.fixture
def robot(monkeypatch):
    module = importlib.import_module("jiuwensymbiosis.simulators.mujoco.microduck_runtime")
    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace(mj_step=lambda *_: None))
    runtime = module.MicroduckRuntime.__new__(module.MicroduckRuntime)
    runtime.model = object()
    runtime.data = SimpleNamespace(time=0.0, qpos=np.zeros(7), body=lambda _: SimpleNamespace(xmat=np.eye(3).ravel()))
    runtime.data.qpos[2] = 0.125
    runtime.rollers = False
    runtime.hold_roller_seat = False
    runtime.kicking_bodies = set()
    runtime.show_started = None
    runtime.gait, runtime.gait_scale = "velstand_gait", 1.0
    runtime.physics_steps = 0
    runtime._command = runtime._policy_command = (0.0, 0.0, 0.0)
    runtime._command_until = 0.0
    runtime.policy = SimpleNamespace(
        vel_cmd=np.zeros(3),
        infer=lambda: np.zeros(14),
        apply_action=Mock(),
        sit_mode=False,
        ground_pick_mode=False,
        behavior_mode=None,
        current_policy="walking",
        update_ground_pick_phase=Mock(),
        update_behavior=Mock(),
    )
    runtime.policy.set_vel_cmd = Mock(
        side_effect=lambda *values: setattr(runtime.policy, "vel_cmd", np.asarray(values))
    )
    runtime.actuators = SimpleNamespace(update=Mock())
    return runtime


@pytest.mark.parametrize("command", [(0.05, 0, 0.01), (0.18, 0.24, 0.6), (-0.3, 0, -1.5)])
def test_policy_receives_exact_validated_velocity_and_stop(robot, command):
    robot.command(*command, 0.2)
    robot._physics_step()
    robot.policy.set_vel_cmd.assert_called_once_with(*command)
    assert robot._command == robot._policy_command == command
    robot.stop()
    robot.physics_steps = 4
    robot._physics_step()
    assert robot._policy_command == (0, 0, 0)
    assert not np.any(robot.policy.vel_cmd)


def test_expired_command_is_zero_at_next_policy_tick(robot):
    robot.command(0.05, 0, 0.01, 0.2)
    robot.data.time = 0.3
    robot._physics_step()
    assert robot._policy_command == (0, 0, 0)
    robot.policy.set_vel_cmd.assert_not_called()


@pytest.mark.parametrize("behavior", [None, "kick_left"])
def test_pose_keeps_policy_and_existing_twist_or_phase(robot, behavior):
    policy = robot.policy
    policy.behavior_mode = behavior
    policy.head_offset = np.zeros(4)
    policy.body_cmd = np.zeros(6)
    policy.command = np.zeros(13)
    if behavior is None:
        policy.current_policy = "sit"
        policy.command[0] = 1
    initial_command = policy.command.copy()
    current_policy = policy.current_policy

    robot.set_policy_pose(head=[0, 0.5, 0, 0], body=[0, 0, 0, 0, 0.1, 0])

    assert policy.current_policy == current_policy
    policy.set_vel_cmd.assert_not_called()
    assert policy.head_offset[1] == 0.5 and policy.body_cmd[4] == 0.1
    if behavior is None:
        initial_command[4] = 0.5
        initial_command[11] = 0.1
    np.testing.assert_array_equal(policy.command, initial_command)


@pytest.mark.parametrize("command", [(0.41, 0, 0), (0.4, 0.01, 0), (0, 0, 1.51), (float("nan"), 0, 0)])
def test_out_of_envelope_request_never_reaches_policy(robot, command):
    with pytest.raises(ValueError):
        robot.command(*command, 0.2)
    assert robot._command == (0, 0, 0)
    robot.policy.set_vel_cmd.assert_not_called()


def test_native_yaw_envelope_does_not_expand_generic_manual_limits(robot):
    from jiuwensymbiosis.simulators.genesis.omni_base import validate_twist

    with pytest.raises(ValueError, match="0.6"):
        validate_twist(0, 0, 1.5)
    with pytest.raises(ValueError, match="0.3"):
        validate_twist(0.4, 0, 0)
    robot.command(0, 0, 1.5, 0.2)
    robot._physics_step()
    assert robot._policy_command == (0, 0, 1.5)


def test_normalized_boundary_command_tolerates_roundoff_only(robot):
    command = (0.4 * math.cos(0.08), 0.4 * math.sin(0.08), 0)
    robot.command(*command, 0.2)
    robot._physics_step()
    robot.policy.set_vel_cmd.assert_called_once_with(*command)
    with pytest.raises(ValueError):
        robot.command(0.4000001, 0, 0, 0.2)


@pytest.mark.parametrize("tampered", ["script", "policy"])
def test_reference_digest_failure_precedes_module_execution(robot, tmp_path, monkeypatch, tampered):
    # The fixture loads by file location; load a separate worker module for its public loader.
    module = importlib.import_module("jiuwensymbiosis.simulators.mujoco.microduck_runtime")
    script, policy = tmp_path / "infer_policy.py", tmp_path / "policy.onnx"
    script.write_text("raise AssertionError('must not execute')")
    policy.write_bytes(b"policy")
    import hashlib

    monkeypatch.setattr(module, "REFERENCE_SHA256", hashlib.sha256(script.read_bytes()).hexdigest())
    monkeypatch.setattr(module, "WALKING_SHA256", hashlib.sha256(policy.read_bytes()).hexdigest())
    (script if tampered == "script" else policy).write_bytes(b"tampered")
    loader = Mock(side_effect=AssertionError("must not import unverified code"))
    monkeypatch.setattr(module.importlib.util, "spec_from_file_location", loader)
    with pytest.raises(ValueError, match="digest mismatch"):
        module.load_reference(script, policy)
    loader.assert_not_called()


def test_native_runtime_reports_unsupported_interfaces_explicitly(robot):
    from jiuwensymbiosis.simulators.mujoco.mujoco_rendering import MujocoRenderer

    with pytest.raises(ValueError, match="真值"):
        robot.query_truth("robot_state", {})
    with pytest.raises(ValueError, match="RGB-D"):
        MujocoRenderer.read_rgbd("head_cam")


@pytest.mark.parametrize(
    ("distance_m", "lateral_m", "cancelled", "expected_reason"),
    [
        (2.02, 0.04, False, "reached"),
        (1.999, 0.3, False, "sprint_short"),
        (0, 0, False, "sprint_short"),
        (2.02, 0.04, True, "cancelled"),
    ],
)
def test_sprint_result_uses_forward_distance_after_measured_stop(distance_m, lateral_m, cancelled, expected_reason):
    from jiuwensymbiosis.simulators.mujoco.microduck_motion import MicroduckMotion

    # Start facing +Y so sideways travel cannot satisfy the forward distance.
    pose = {
        "position_m": [1.0, 2.0, 0.12],
        "body_yaw_rad": math.pi / 2,
        "linear_velocity_m_s": [0, 0, 0],
        "angular_velocity_rad_s": [0, 0, 0],
    }
    duck = SimpleNamespace(
        observe_base=lambda: pose.copy(),
        policy=SimpleNamespace(sit_mode=False, current_policy="standing", ground_pick_mode=False, behavior_mode=None),
        actions=["sprint_2m"],
        upright=lambda: True,
        show_started=None,
        max_linear_rate=0.4,
        select_gait=Mock(),
        command=Mock(),
        stop=Mock(),
    )
    motion = MicroduckMotion(duck, SimpleNamespace(active=False))
    motion.start({"id": "sprint", "kind": "sprint_2m"}, 0)
    motion.tick(0.1, 1)
    duck.select_gait.assert_called_once_with("sprint_2m")
    duck.command.assert_called_once_with(0.4, 0, 0, 0.2)
    if cancelled:
        motion.cancel()

    duck.command.reset_mock()
    pose["position_m"] = [1.0, 3.91 if distance_m else 2.0, 0.12]
    pose["linear_velocity_m_s"] = [0, 0.5, 0]
    motion.tick(0.1 if distance_m else 12, 2)
    assert motion.active and not motion.result["ok"]
    duck.command.assert_not_called()
    duck.select_gait.assert_called_once_with("sprint_2m")
    assert duck.stop.call_count >= 2

    pose["position_m"] = [1.0 + lateral_m, 2.0 + distance_m, 0.12]
    pose["linear_velocity_m_s"] = [0, 0, 0]
    for step in range(3, 7):
        motion.tick(0.1, step)
    assert not motion.active and motion.result["settled"]
    assert motion.result["forward_displacement_m"] == pytest.approx(distance_m)
    assert motion.result["reason"] == expected_reason
    assert motion.result["ok"] == (expected_reason == "reached")
    assert duck.select_gait.call_args_list == [call("sprint_2m"), call("velstand_gait")]


def test_renderer_evicts_old_sizes_and_closes_remaining_resources(monkeypatch):
    from types import ModuleType

    from jiuwensymbiosis.simulators.common.camera import OrbitCamera
    from jiuwensymbiosis.simulators.mujoco.mujoco_rendering import MujocoRenderer

    created = []

    def create_renderer(model, height, width):
        renderer = SimpleNamespace(
            update_scene=Mock(), render=lambda: np.zeros((height, width, 3), dtype=np.uint8), close=Mock()
        )
        created.append(renderer)
        return renderer

    sdk, torch = ModuleType("mujoco"), ModuleType("torch")
    sdk.Renderer = create_renderer
    sdk.MjvOption = lambda: SimpleNamespace(geomgroup=np.ones(6))
    sdk.MjvCamera = lambda: SimpleNamespace(lookat=np.zeros(3))
    sdk.mjtCamera = SimpleNamespace(mjCAMERA_FREE=0)
    torch.from_numpy = lambda value: value
    monkeypatch.setitem(sys.modules, "mujoco", sdk)
    monkeypatch.setitem(sys.modules, "torch", torch)
    size = [960, 540]
    runtime = SimpleNamespace(
        cameras={},
        frame_size=lambda _: tuple(size),
        model=object(),
        data=object(),
        hidden_geom_groups=(1,),
        pose=OrbitCamera().snapshot(),
    )
    renderer = MujocoRenderer(runtime)
    renderer.read_rgb()
    size[:] = (640, 480)
    renderer.read_rgb()
    size[:] = (960, 540)
    renderer.read_rgb()
    size[:] = (1280, 720)
    renderer.read_rgb()
    assert len(created) == 3 and len(renderer.renderers) == 2
    created[1].close.assert_called_once()
    renderer.close()
    assert not renderer.renderers
    assert all(item.close.call_count == 1 for item in created)
