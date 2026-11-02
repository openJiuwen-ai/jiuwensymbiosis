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
    runtime.settings = SimpleNamespace(dt=0.01)
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
