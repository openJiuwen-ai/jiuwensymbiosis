"""Worker control/capture boundaries exercised without Genesis, CUDA or a process."""

import copy
import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.simulators.genesis.camera import PreviewSettings
from jiuwensymbiosis.simulators.genesis.scheduling import PhysicsClock


@pytest.fixture
def worker_loop(monkeypatch, tmp_path):
    state = SimpleNamespace(now=0.0, stopping=False, renders=0, encoders=0, commands=[], statuses=[], steps=[])

    def advance(seconds):
        state.now += seconds

    state.advance = advance
    state.command = {
        "settings": asdict(PreviewSettings()),
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
    module("encoding", NvEncoder=encoder)
    module("ipc", Mailbox=object)
    module("scene_catalog", resolve_scene=lambda _: None, validate_material_bundle=lambda _: None)
    module("scene_runtime", RuntimeScene=lambda *a, **kw: runtime)
    from jiuwensymbiosis.simulators.genesis import startup_diagnostics

    monkeypatch.setitem(sys.modules, "startup_diagnostics", startup_diagnostics)
    module("quadrants")
    module("scheduling", PhysicsClock=lambda dt: PhysicsClock(dt, monotonic=lambda: state.now))
    module("usd_scene", prepare_static_stage=lambda *a: None)
    module("genesis", cuda=object(), init=lambda **kw: None, destroy=lambda: None)
    sdk = module("gs_nyx.nyx_py_sdk", startup=lambda: None, shutdown=lambda: None)
    module("gs_nyx", nyx_py_sdk=sdk)
    module("torch", cuda=SimpleNamespace(current_stream=lambda: SimpleNamespace(synchronize=lambda: None)))
    path = Path(__file__).parents[4] / "jiuwensymbiosis/simulators/genesis/worker.py"
    spec = importlib.util.spec_from_file_location("worker_scheduling_under_test", path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    monkeypatch.setattr(worker, "time", SimpleNamespace(monotonic=lambda: state.now, sleep=advance))
    state.run = lambda: worker.run_preview(mailbox, tmp_path)
    state.directory = tmp_path
    return state


def capture_request():
    return {"id": "test", "config_revision": 0, "scene_instance_id": "scene", "camera_revision": 1, "expires_at": 10}


def test_invalid_capture_is_rejected_while_physics_blocks_render(worker_loop):
    state = worker_loop
    state.command["capture"] = {**capture_request(), "camera_revision": 0}
    state.on_camera = lambda: state.advance(0.100)

    state.run()

    assert state.renders == 0
    assert state.statuses[-1]["capture_result"] == {"id": "test", "error": "capture_failed"}


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
