"""Reset admission, measured stopping, and state restoration without a GPU."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jiuwensymbiosis.simulators.genesis.preview import GenesisPreview


@pytest.fixture
def world(monkeypatch):
    directory = Path(__file__).parents[4] / "jiuwensymbiosis/simulators/genesis"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("runtime_reset_under_test", directory / "scene_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sample = {"linear_velocity_m_s": [0, 0, 0], "angular_velocity_rad_s": [0, 0, 0]}
    motion = SimpleNamespace(active=False)
    monkeypatch.setitem(
        sys.modules, "base_motion", SimpleNamespace(measure=lambda _: sample, RelativeMotion=lambda _: motion)
    )
    runtime = module.RuntimeScene.__new__(module.RuntimeScene)
    runtime.info = {"instance_id": "old"}
    runtime.scene, runtime.renderer, runtime.robot = Mock(), object(), Mock()
    runtime.motion = SimpleNamespace(active=False)
    runtime.settings = SimpleNamespace(dt=0.01, planning_profile="")
    runtime._initial_state = object()
    runtime._initial_steps = 200
    runtime.steps, runtime.cycle = 300, 0
    runtime.selection = runtime.selected_entity = runtime.truth = object()
    return runtime, sample


def test_reset_retains_world_and_renderer_but_invalidates_observation_identity(world):
    runtime, _ = world
    physics, renderer = runtime.scene, runtime.renderer
    result = runtime.reset()
    assert runtime.scene is physics and runtime.renderer is renderer
    physics.reset.assert_called_once_with(runtime._initial_state)
    assert physics.step.call_count == 60
    runtime.robot.reset_controller.assert_called_once()
    assert runtime.steps == 200 and runtime.cycle == 1
    assert runtime.selection is runtime.selected_entity is runtime.truth is None
    assert result["scene_instance_id"] != "old" and result["stop_confirmation"] == "measured"


def test_reset_rejects_active_motion_without_teleporting(world):
    runtime, _ = world
    runtime.motion.active = True
    with pytest.raises(RuntimeError, match="任务"):
        runtime.reset()
    runtime.scene.reset.assert_not_called()


def test_unconfirmed_stop_never_restores_checkpoint(world):
    runtime, sample = world
    sample["linear_velocity_m_s"] = [0.1, 0, 0]
    with pytest.raises(RuntimeError, match="未确认停车"):
        runtime.reset()
    runtime.scene.reset.assert_not_called()
    assert runtime.info["instance_id"] == "old"


def test_shutdown_during_stopping_never_restores_checkpoint(world):
    runtime, _ = world
    with pytest.raises(RuntimeError, match="未确认停车"):
        runtime.reset(stopping=lambda: True)
    runtime.scene.reset.assert_not_called()


@pytest.mark.parametrize("busy", ["_motion_owner", "_capture_request", "_scene_query"])
def test_host_rejects_reset_while_work_is_owned(tmp_path, busy, monkeypatch):
    preview = GenesisPreview(state_dir=tmp_path)
    monkeypatch.setattr(preview, "status", lambda: {"state": "ready"})
    setattr(preview, busy, "active")
    with pytest.raises(RuntimeError, match="当前有任务"):
        preview.reset_scene()
    assert preview._reset_request is None


@pytest.fixture
def host(tmp_path):
    preview = GenesisPreview(state_dir=tmp_path)
    observed = {"state": "ready", "scene": {"instance_id": "old"}, "config_revision": 0}
    preview._mailbox = SimpleNamespace(read_status=lambda: observed.copy(), write_control=Mock())
    return preview, observed


@pytest.mark.parametrize("error", [None, "复位前未确认停车，保留当前世界。"])
def test_host_releases_completed_reset_and_preserves_result(host, error):
    preview, observed = host

    def respond(control):
        if control["reset"]:
            observed["reset_result"] = {"id": control["reset"]["id"], "complete": True, "error": error}

    preview._mailbox.write_control.side_effect = respond
    if error:
        with pytest.raises(RuntimeError, match="未确认停车"):
            preview.reset_scene()
    else:
        assert preview.reset_scene()["state"] == "ready"
    assert preview._reset_request is None
    assert preview._mailbox.write_control.call_args.args[0]["reset"] is None
    assert preview.status()["state"] == "ready"
    # A rejected reset keeps the world available for a later explicit retry.
    if error:
        with pytest.raises(RuntimeError, match="未确认停车"):
            preview.reset_scene()


def test_reset_timeout_stays_isolated_until_matching_completion(host, monkeypatch):
    from jiuwensymbiosis.simulators.genesis import preview as module

    preview, observed = host
    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", lambda _: clock.__setitem__(0, clock[0] + 16))
    with pytest.raises(TimeoutError, match="复位未确认"):
        preview.reset_scene()
    request = preview._reset_request
    assert preview.status()["state"] == "resetting"
    with pytest.raises(RuntimeError, match="不可用"):
        preview.acquire_motion()
    observed["reset_result"] = {"id": "another-request", "complete": True}
    assert preview.status()["state"] == "resetting"
    observed["reset_result"] = {"id": request["id"], "complete": True}
    assert preview.status()["state"] == "ready"
    assert preview._reset_request is None


def test_completed_reset_does_not_hide_a_failed_worker(host):
    preview, observed = host

    def respond(control):
        if control["reset"]:
            observed.update(state="failed", message="worker failed")
            observed["reset_result"] = {"id": control["reset"]["id"], "complete": True, "error": "worker failed"}

    preview._mailbox.write_control.side_effect = respond
    with pytest.raises(RuntimeError, match="worker failed"):
        preview.reset_scene()
    assert preview._reset_request is None
    assert preview.status()["state"] == "failed"


@pytest.mark.parametrize("mode, expected", [("robot_mobile", False), ("robot_preview", True), ("demo", True)])
def test_physics_preserves_existing_robot_self_collision_policy(world, monkeypatch, mode, expected):
    runtime, _ = world
    options = SimpleNamespace(SimOptions=lambda **kw: kw, RigidOptions=lambda **kw: kw)
    scene = Mock(side_effect=RuntimeError("options captured"))
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "genesis", SimpleNamespace(options=options, Scene=scene))
    monkeypatch.setitem(sys.modules, "gs_nyx", SimpleNamespace(nyx_py_sdk=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "gs_nyx.nyx_py_sdk", SimpleNamespace())
    runtime.source = None
    runtime.info = {"mode": mode}
    with pytest.raises(RuntimeError, match="options captured"):
        runtime._build_physics()
    assert scene.call_args.kwargs["rigid_options"]["enable_self_collision"] is expected


def test_reset_rebuilds_arm_evidence_but_never_bypasses_unknown_stop(world, monkeypatch):
    runtime, _ = world
    old = Mock(active=False, follower=SimpleNamespace(terminal=False))
    old.effectors.evidence = {}
    runtime.arm_motion = old
    runtime._planning_profile = {}
    runtime._evidence_directory = Path("/unused")
    replacement = Mock()
    factory = Mock(return_value=replacement)
    monkeypatch.setitem(sys.modules, "arm_motion", SimpleNamespace(ArmMotion=factory))
    factory.side_effect = ValueError("invalid model")
    with pytest.raises(ValueError, match="invalid model"):
        runtime.reset()
    runtime.scene.reset.assert_not_called()
    old.close.assert_not_called()
    assert runtime.arm_motion is old and runtime.info["instance_id"] == "old"
    factory.side_effect = None
    runtime.reset()
    old.close.assert_called_once()
    assert runtime.arm_motion is replacement
    assert runtime.info["instance_id"] in str(factory.call_args.args[2])
    runtime.arm_motion = old
    old.follower.terminal = True
    runtime.scene.reset.reset_mock()
    with pytest.raises(RuntimeError, match="stopping unconfirmed"):
        runtime.reset()
    runtime.scene.reset.assert_not_called()
