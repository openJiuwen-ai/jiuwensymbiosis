"""Worker control/capture boundaries exercised without Genesis, CUDA or a process."""

import copy
import importlib.util
import sys
from dataclasses import asdict, replace
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
        motion=SimpleNamespace(active=False, result={}, accept_request=lambda *a, **kw: None, cancel=lambda *a: None),
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
    state.runtime = runtime
    state.directory = tmp_path
    return state


@pytest.mark.parametrize("requested, expected", [(None, "cpu"), ("cpu", "cpu"), ("metal", "metal")])
def test_macos_physics_backend_keeps_the_worker_contract(worker_loop, monkeypatch, requested, expected):
    state = worker_loop
    monkeypatch.setattr(state.worker.sys, "platform", "darwin")
    if requested is None:
        monkeypatch.delenv("JIUWEN_GENESIS_MACOS_BACKEND", raising=False)
    else:
        monkeypatch.setenv("JIUWEN_GENESIS_MACOS_BACKEND", requested)
    state.run()
    assert state.backend == expected
    assert state.statuses[-1]["state"] == "ready"
    assert state.statuses[-1]["physics_backend"] == expected


def test_macos_invalid_backend_fails_before_scene_or_encoder(worker_loop, monkeypatch):
    state = worker_loop
    monkeypatch.setattr(state.worker.sys, "platform", "darwin")
    monkeypatch.setenv("JIUWEN_GENESIS_MACOS_BACKEND", "invalid")
    state.run()
    assert state.encoders == 0
    assert state.statuses[-1]["state"] == "failed"
    assert state.statuses[-1]["startup"]["failed_phase"] == "genesis_init"


def capture_request():
    return {"id": "test", "config_revision": 0, "scene_instance_id": "scene", "camera_revision": 1, "expires_at": 10}


def test_invalid_capture_is_rejected_while_physics_blocks_render(worker_loop):
    state = worker_loop
    state.command["capture"] = {**capture_request(), "camera_revision": 0}
    state.on_camera = lambda: state.advance(0.100)

    state.run()

    assert state.renders == 0
    waiting = [status for status in state.statuses if status.get("rendered_frames") == 0]
    assert waiting and all(status["state"] == "starting" for status in waiting)
    assert all(status["loading"]["state"] == "running" for status in waiting)
    assert all("已就绪" not in status["message"] for status in waiting)
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


def test_unconfigured_arm_request_does_not_destroy_scene(worker_loop):
    state = worker_loop
    state.command["base_motion"] = {
        "id": "arm-invalid",
        "kind": "arm",
        "scene_instance_id": "scene",
        "config_revision": 0,
        "lease_until": state.now + 10,
        "cancelled": False,
    }
    state.on_ready = lambda: setattr(state, "stopping", bool(state.steps and state.renders))
    state.on_camera = lambda: state.advance(0.100)
    state.run()
    assert state.steps
    ready = [s for s in state.statuses if s.get("state") == "ready"]
    assert ready and not any(s.get("state") == "failed" for s in state.statuses)
    result = ready[-1]["base_motion"]
    assert result["id"] == "arm-invalid" and not result["ok"]
    assert result["stop_confirmation"] == "unknown"


def test_configuration_switch_updates_settings_through_the_reconfigurator(worker_loop):
    state = worker_loop
    target = replace(PreviewSettings(), fps=30)

    def ready():
        current = state.statuses[-1]
        if state.command["config_revision"] == 0:
            state.command.update(config_revision=1, settings=asdict(target))
        elif current.get("config_revision") == 1:
            state.stopping = True

    state.on_ready = ready
    state.run()

    ready_statuses = [status for status in state.statuses if status.get("state") == "ready"]
    assert ready_statuses[-1]["config_revision"] == 1
    assert ready_statuses[-1]["settings"]["fps"] == 30
    assert ready_statuses[-1]["reconfigure_error"] is None
    assert state.encoders == 2
    assert all(status["state"] != "failed" for status in state.statuses)


def test_failed_configuration_switch_keeps_old_settings_and_reports_the_error(worker_loop, monkeypatch):
    state = worker_loop
    target = replace(PreviewSettings(), fps=30)
    original = state.worker._encoder_type

    def failing_factory(settings):
        if settings.fps == 30:
            return lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("nvenc unavailable"))
        return original(settings)

    monkeypatch.setattr(state.worker, "_encoder_type", failing_factory)

    def ready():
        if state.command["config_revision"] == 0:
            state.command.update(config_revision=1, settings=asdict(target))
        elif state.statuses[-1].get("config_revision") == 1:
            state.stopping = True

    state.on_ready = ready
    state.run()

    ready_statuses = [status for status in state.statuses if status.get("state") == "ready"]
    final = ready_statuses[-1]
    assert final["config_revision"] == 1
    assert final["settings"]["fps"] == 60
    assert "切换失败" in final["reconfigure_error"]
    assert all(status["state"] != "failed" for status in state.statuses)


def test_malformed_control_requests_stay_request_scoped(worker_loop):
    state = worker_loop
    state.command["capture"] = {**capture_request(), "include_depth": True}
    state.command["pick"] = "not-an-object"
    state.command["scene_query"] = {
        "id": "query-bad",
        "operation": "scene",
        "params": [],
        "scene_instance_id": "scene",
        "config_revision": 0,
        "expires_at": 10,
    }
    state.command["reset"] = {"id": "reset-bad"}
    state.command["base_command"] = {
        "id": "base-bad",
        "vx": "fast",
        "vy": 0,
        "wz": 0,
        "scene_instance_id": "scene",
        "config_revision": 0,
        "expires_at": 10,
    }
    state.command["base_motion"] = {"id": "motion-bad", "kind": "arm", "cancelled": 1}

    state.on_ready = lambda: setattr(state, "stopping", bool(state.steps and state.renders))
    state.on_camera = lambda: state.advance(0.100)
    state.run()

    ready = [status for status in state.statuses if status.get("state") == "ready"]
    final = state.statuses[-1]
    assert ready and final["state"] == "ready"
    assert final["capture_result"]["error"] == "capture_failed"
    assert final["pick_result"]["error"]
    assert final["scene_query_result"]["error"]
    assert final["reset_result"]["error"]
    assert final["base_command_result"]["applied"] is False
    assert final["base_command_result"]["error"]
    assert final["base_motion"]["reason"] == "invalid_request"
    assert final["base_motion"]["id"] == "motion-bad"
    assert state.commands == []


def test_rgbd_capture_reports_depth_and_the_base_sample(worker_loop):
    state = worker_loop
    runtime = state.runtime
    runtime.robot.bounds = np.array([[0.1, 0.1, 0.0], [0.3, 0.3, 0.2]])
    runtime.renderer.last_camera = {"T_world_camera": np.eye(4).tolist()}
    state.worker.measure = lambda robot: {"position_m": [0.0, 0.0, 0.0], "body_yaw_rad": 0.0}

    def read_rgbd(name, *, stride, check):
        check()
        return np.zeros((4, 6, 3), dtype=np.uint8), np.zeros((4, 6), dtype=np.float32)

    runtime.renderer.read_rgbd = read_rgbd
    state.command["capture"] = {**capture_request(), "camera_name": "wrist", "include_depth": True, "depth_stride": 8}
    seen = {}

    def on_ready():
        if state.renders:
            seen["depth_bytes"] = (state.directory / "capture-test.npz").stat().st_size
        state.stopping = bool(state.steps and state.renders)

    state.on_ready = on_ready
    state.run()

    final = state.statuses[-1]
    assert final["state"] == "ready"
    result = final["capture_result"]
    assert (result["width"], result["height"]) == (6, 4)
    assert result["base_sample"] == {"position_m": [0.0, 0.0, 0.0], "body_yaw_rad": 0.0}
    assert result["body_bounds_m"] == [[0.1, 0.1, 0.0], [0.3, 0.3, 0.2]]
    # The worker deletes the pending depth file on exit; it must exist while live.
    assert seen["depth_bytes"] > 0


def test_unconfirmed_cleanup_fails_the_worker_instead_of_reporting_ready(worker_loop, monkeypatch):
    state = worker_loop
    from jiuwensymbiosis.simulators.common import reconfigurator

    def cleanup_failure(self, *args, **kwargs):
        raise reconfigurator.ReconfigurationCleanupError("old world release unconfirmed", ())

    monkeypatch.setattr(reconfigurator.Reconfigurator, "apply", cleanup_failure)
    state.on_ready = lambda: state.command.update(config_revision=1) if state.steps else None
    state.run()

    ready = [status for status in state.statuses if status.get("state") == "ready"]
    final = state.statuses[-1]
    assert ready
    assert final["state"] == "failed"
    assert "unconfirmed" in final["message"]
    assert all(status.get("config_revision") != 1 for status in ready)
