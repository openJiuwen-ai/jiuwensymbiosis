"""Worker control/capture boundaries exercised without Genesis, CUDA or a process."""

import copy
import importlib.util
import sys
from dataclasses import asdict
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.simulators.common.camera import PreviewSettings
from jiuwensymbiosis.simulators.common.scheduling import PhysicsClock
from jiuwensymbiosis.simulators.common.startup_diagnostics import StartupDiagnostics


@pytest.fixture
def worker_loop(monkeypatch, tmp_path):
    state = SimpleNamespace(now=0.0, stopping=False, renders=0, encoders=0, commands=[], statuses=[], steps=[])

    def advance(seconds):
        state.now += seconds

    state.advance = advance
    state.command = {
        "settings": asdict(PreviewSettings(backend="mujoco")),
        "config_revision": 0,
        "stream_epoch": 0,
        "streaming": True,
        "camera": {"revision": 1},
        "keyframe_request": 0,
    }
    state.on_camera = state.on_render = state.on_encode_init = state.on_step = lambda: None
    state.on_ready = lambda: setattr(state, "stopping", True)
    state.current_base = None

    def stop_base():
        state.current_base = None

    def command_base(*args):
        state.current_base = args
        state.commands.append(args)

    def read_rgb(camera_name="observer"):
        state.renders += 1
        advance(0.020)
        state.on_render()
        return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.zeros((2, 2, 3), dtype=np.uint8)))

    runtime = SimpleNamespace(
        settings=SimpleNamespace(robot_mode="moz1_wheels"),
        info={"mode": "robot_mobile", "instance_id": "scene"},
        pose={"revision": 0},
        initial_pose={},
        motion=SimpleNamespace(active=False, result={}, accept_request=lambda *a, **kw: None),
        robot=SimpleNamespace(stop=stop_base, command=command_base, snapshot=lambda: {}),
        renderer=SimpleNamespace(read_rgb=read_rgb, fov_y=60, render_counts={}),
        cameras={},
        frame_size=lambda name: (1280, 720),
        bodies=[],
        indoor=None,
        steps=0,
        cycle=0,
        selection_snapshot=lambda: None,
        close=lambda: None,
    )

    def update_camera(camera):
        runtime.pose = camera.copy()
        state.on_camera()

    def step():
        runtime.steps += 1
        state.steps.append(state.current_base)
        advance(0.010)
        state.on_step()

    runtime.update_camera = update_camera
    runtime.step = step

    def write_status(status):
        state.statuses.append(status)
        if "steps" in status:
            state.on_ready()

    mailbox = SimpleNamespace(
        read_control=lambda: copy.deepcopy(state.command),
        write_status=write_status,
        write_frame=lambda *a: None,
        acknowledged=lambda: 100000,
        stopping=lambda: state.stopping,
    )

    def encoder(settings, **kwargs):
        state.encoders += 1
        state.on_encode_init()
        return SimpleNamespace(
            close=lambda: None,
            copy_ms=0,
            encode_ms=0,
            encode=lambda rgb, metadata, **kw: [{**metadata, "h264": b"frame"}],
        )

    def module(name, **attributes):
        result = ModuleType(name)
        result.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, result)
        return result

    module("base_motion", measure=lambda robot: {})
    module("camera", PreviewSettings=PreviewSettings)
    module("encoding", NvEncoder=encoder, SoftwareEncoder=encoder)
    module("ipc", Mailbox=object)
    module("scene_catalog", resolve_scene=lambda _: None, validate_material_bundle=lambda _: None)
    module("scene_runtime", create_runtime=lambda *a, **kw: runtime, reset_runtime=lambda *a, **kw: (runtime, {}))
    module("quadrants")
    module("mujoco", __version__="4.0.1")
    module("scheduling", PhysicsClock=lambda dt: PhysicsClock(dt, monotonic=lambda: state.now))
    module("startup_diagnostics", StartupDiagnostics=lambda **kw: StartupDiagnostics(**kw, clock=lambda: state.now))
    module("usd_scene", prepare_static_stage=lambda *a: None)
    module(
        "genesis",
        cuda="cuda",
        cpu="cpu",
        metal="metal",
        init=lambda **kw: setattr(state, "backend", kw["backend"]),
        destroy=lambda: None,
    )
    sdk = module("gs_nyx.nyx_py_sdk", startup=lambda: None, shutdown=lambda: None)
    module("gs_nyx", nyx_py_sdk=sdk)
    module("torch", cuda=SimpleNamespace(current_stream=lambda: SimpleNamespace(synchronize=lambda: None)))
    worker = importlib.import_module("jiuwensymbiosis.simulators.common.worker")
    monkeypatch.setattr(worker, "measure", lambda robot: {})
    from jiuwensymbiosis.simulators.common import encoding

    monkeypatch.setattr(encoding, "SoftwareEncoder", encoder)
    monkeypatch.setattr(worker, "NvEncoder", encoder)
    monkeypatch.setattr(worker, "create_runtime", lambda *a, **kw: runtime)
    monkeypatch.setattr(worker, "reset_runtime", lambda *a, **kw: (runtime, {}))
    monkeypatch.setattr(worker, "PhysicsClock", lambda dt: PhysicsClock(dt, monotonic=lambda: state.now))
    monkeypatch.setattr(worker, "StartupDiagnostics", lambda **kw: StartupDiagnostics(**kw, clock=lambda: state.now))
    monkeypatch.setattr(worker.sys, "platform", "linux")
    monkeypatch.setattr(worker, "time", SimpleNamespace(monotonic=lambda: state.now, sleep=advance))
    state.run = lambda: worker.run_preview(mailbox, tmp_path)
    state.worker = worker
    state.directory = tmp_path
    state.runtime = runtime
    return state


def capture_request():
    return {"id": "test", "config_revision": 0, "scene_instance_id": "scene", "camera_revision": 1, "expires_at": 10}


def test_invalid_capture_is_rejected_while_physics_blocks_render(worker_loop):
    state = worker_loop
    state.command["capture"] = {**capture_request(), "camera_revision": 0}
    state.on_camera = lambda: state.advance(0.100)

    state.run()

    # Invalid capture remains rejected, but it must not starve valid video.
    assert state.renders == 1
    assert state.statuses[-1]["capture_result"] == {"id": "test", "error": "capture_failed"}


def test_video_keeps_refreshing_when_physics_cannot_clear_its_debt(worker_loop):
    state = worker_loop
    rendered_at = []
    state.on_render = lambda: rendered_at.append(state.now)
    state.on_ready = lambda: setattr(state, "stopping", state.now >= 2.0)
    state.run()
    # 10 ms physics ticks cannot meet the unchanged 1/120 s target. Frames
    # still arrive under the 20% video budget, not the old 0.5 s fallback.
    assert len(rendered_at) > 12
    assert max(b - a for a, b in zip(rendered_at, rendered_at[1:], strict=False)) < 0.2
    assert state.statuses[-1]["physics_clock"]["actual_hz"] < 120
    assert state.statuses[-1]["video_schedule"]["recent_source_fps"] > 5


def test_stop_during_capture_prevents_an_extra_video_render(worker_loop):
    state = worker_loop
    state.command["capture"] = capture_request()
    state.on_render = lambda: setattr(state, "stopping", True)

    state.run()

    assert state.renders == 1
    assert not (state.directory / "capture-test.jpg").exists()
    assert all(status["state"] != "failed" for status in state.statuses)


def test_cancel_during_capture_removes_the_late_image(worker_loop):
    state = worker_loop
    state.command["capture"] = capture_request()
    state.on_render = lambda: state.command.update(capture=None)
    state.on_ready = lambda: None
    state.cleaned_before_exit = False

    def step():
        if state.renders:
            state.cleaned_before_exit = not (state.directory / "capture-test.jpg").exists()
            state.stopping = True

    state.on_step = step
    state.run()

    assert state.cleaned_before_exit
    assert all(status["state"] != "failed" for status in state.statuses)


def test_encoder_rebuild_cannot_apply_a_stale_motion_lease(worker_loop):
    state = worker_loop

    def ready():
        state.command.update(
            stream_epoch=1,
            base_command={
                "id": "stale",
                "scene_instance_id": "scene",
                "config_revision": 0,
                "expires_at": state.now + 0.100,
                "vx": 0.1,
                "vy": 0,
                "wz": 0,
            },
        )

    def encoder_init():
        if state.encoders == 2:
            state.advance(0.200)
            state.command["base_command"] = None

    state.on_ready = ready
    state.on_encode_init = encoder_init
    state.on_step = lambda: setattr(state, "stopping", True)

    state.run()

    assert state.encoders == 2
    assert state.commands == []
    assert state.steps == [None]
    assert all(status["state"] != "failed" for status in state.statuses)


@pytest.mark.parametrize(
    "mode,base_active,arm_active,accepted",
    [
        ("action", True, False, False),
        ("spatial_memory", True, False, True),
        ("spatial_memory", True, True, False),
        ("action", False, False, True),
    ],
)
def test_rgbd_motion_mode_preserves_action_and_arm_guards(
    worker_loop, monkeypatch, mode, base_active, arm_active, accepted
):
    from unittest.mock import Mock

    state = worker_loop
    runtime = state.runtime
    runtime.info["mode"] = "robot_mobile"
    runtime.motion = Mock(active=base_active, result={"id": "move"})
    runtime.arm_motion = Mock(active=arm_active)
    runtime.arm_motion.base_pose.return_value = np.eye(4)
    runtime.arm_motion.effectors.report.return_value = {}
    from jiuwensymbiosis.simulators.genesis import rgbd as rgbd_module

    monkeypatch.setattr(rgbd_module, "base_body_bounds", lambda *args: np.zeros((2, 3)))
    monkeypatch.setattr(rgbd_module, "base_from_camera", lambda *args: np.eye(4))
    runtime.robot.bounds = np.zeros((2, 3))
    runtime.renderer.last_camera = {"T_world_camera": np.eye(4).tolist()}

    def read_rgbd(name, **kwargs):
        kwargs["check"]()
        return np.zeros((8, 8, 3), np.uint8), np.ones((8, 8), np.float32)

    runtime.renderer.read_rgbd = read_rgbd
    state.command["capture"] = {
        **capture_request(),
        "include_depth": True,
        "camera_name": "head_cam",
        "capture_mode": mode,
    }
    state.run()
    assert not any(s["state"] == "failed" for s in state.statuses)
    result = state.statuses[-1]["capture_result"]
    assert ("error" not in result) is accepted
    if accepted and mode == "spatial_memory":
        assert result["motion_active"] is True
        assert result["synchronization"] == {
            "kind": "single_physics_state",
            "rgb_step": result["step"],
            "depth_step": result["step"],
            "pose_step": result["step"],
        }


def test_unconfigured_arm_request_does_not_destroy_scene(worker_loop):
    state = worker_loop
    state.command["base_motion"] = {"id": "arm-invalid", "kind": "arm"}
    state.on_ready = lambda: setattr(state, "stopping", bool(state.steps and state.renders))
    state.on_camera = lambda: state.advance(0.100)
    state.run()
    assert state.steps
    ready = [s for s in state.statuses if s.get("state") == "ready"]
    assert ready and not any(s.get("state") == "failed" for s in state.statuses)
    result = ready[-1]["base_motion"]
    assert result["id"] == "arm-invalid" and not result["ok"]
    assert result["stop_confirmation"] == "unknown"


def test_virtual_rgbd_has_no_fabricated_robot_evidence(worker_loop, monkeypatch):
    state = worker_loop
    runtime = state.runtime
    runtime.settings.robot_mode = "virtual_camera"
    runtime.info["mode"] = "static_background"
    runtime.motion = None
    runtime.robot = None
    from jiuwensymbiosis.simulators.genesis import rgbd as rgbd_module

    monkeypatch.setattr(rgbd_module, "base_body_bounds", lambda *a: pytest.fail("virtual capture has no robot bounds"))
    monkeypatch.setattr(
        rgbd_module, "base_from_camera", lambda *a: pytest.fail("virtual capture has no base transform")
    )
    runtime.renderer.last_camera = {"T_world_camera": np.eye(4).tolist()}
    runtime.renderer.read_rgbd = lambda *a, **kw: (np.zeros((8, 8, 3), np.uint8), np.ones((8, 8), np.float32))
    state.command["capture"] = {
        **capture_request(),
        "include_depth": True,
        "camera_name": "scan_cam",
        "capture_mode": "action",
    }
    state.run()
    result = state.statuses[-1]["capture_result"]
    assert "error" not in result
    assert result["pose_source"] == "virtual_camera_pose"
    assert result["robot_loaded"] is False
    assert "base_sample" not in result
    assert "T_base_camera_mm" not in result


def test_genesis_engine_uses_cpu_and_dispatches_nyx(worker_loop):
    from jiuwensymbiosis.simulators.genesis.engine import engine_session

    calls = []

    def dispatch_call(function):
        calls.append(function)
        return function()

    import logging

    startup = StartupDiagnostics(logger=logging.getLogger(__name__))
    with engine_session(startup, render_dispatch=SimpleNamespace(call=dispatch_call)):
        assert worker_loop.backend == "cpu"
        assert len(calls) == 1
    assert len(calls) == 2
