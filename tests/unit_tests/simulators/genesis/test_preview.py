"""Camera boundaries and real process ownership, using a CPU-only child."""

import importlib.util
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from jiuwensymbiosis.simulators.genesis import preview
from jiuwensymbiosis.simulators.genesis.camera import OrbitCamera


def wait_for(predicate, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.02)
    raise AssertionError("Child did not reach expected state")


@pytest.fixture
def manager(tmp_path, monkeypatch):
    worker = tmp_path / "cpu_worker.py"
    worker.write_text(
        "import sys, time\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(preview._WORKER.parent)!r})\n"
        "from ipc import Mailbox\n"
        "assert not any(k.startswith('openjiuwen') for k in sys.modules)\n"
        "mailbox = Mailbox(Path(sys.argv[1]))\nstep = 0\n"
        "while not mailbox.stopping():\n"
        "    command = mailbox.read_control()\n    step += 1\n"
        "    mailbox.write_status({'state':'ready', 'steps':step, 'streaming': command['streaming'], 'revision':command['camera']['revision'], 'settings':command['settings'], 'config_revision':command.get('config_revision',0)})\n"
        "    time.sleep(.02)\n"
        "mailbox.close()\n"
    )
    monkeypatch.setattr(preview, "_WORKER", worker)
    original = importlib.util.find_spec
    monkeypatch.setattr(
        preview.importlib.util,
        "find_spec",
        lambda name: (
            object() if name in {"genesis", "gs_nyx_plugin", "aiortc", "av", "PyNvVideoCodec"} else original(name)
        ),
    )
    instance = preview.GenesisPreview(state_dir=tmp_path / "state")
    yield instance
    instance.stop()


def test_import_and_status_do_not_load_optional_sdks(tmp_path):
    code = "import sys; from jiuwensymbiosis.simulators.genesis.stream import PreviewStream; s=PreviewStream(); assert s.status()['state']=='stopped'; assert not ({'genesis','gs_nyx','gs_nyx_plugin','aiortc','av','PyNvVideoCodec','nicegui','jiuwensymbiosis_gui'} & sys.modules.keys())"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, timeout=30)
    instance = preview.GenesisPreview(state_dir=tmp_path / "unused")
    assert instance.status()["state"] == "stopped"
    assert not instance.state_dir.exists()


def test_worker_asset_import_keeps_sdk_lookup_and_agent_isolation(tmp_path):
    code = (
        "import sys, importlib.util\n"
        f"sys.path.insert(0, {str(preview._WORKER.parent)!r})\n"
        "before = importlib.util.find_spec('genesis')\n"
        "import scene_catalog\n"
        "after = importlib.util.find_spec('genesis')\n"
        "assert (before.origin if before else None) == (after.origin if after else None)\n"
        "assert not {'openjiuwen', 'jiuwensymbiosis', 'torch', 'genesis'} & sys.modules.keys()\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, check=True, timeout=30)


def test_concurrent_start_disconnect_and_stop_own_exactly_one_child(manager):
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: manager.start(), range(4)))
    assert len({r["pid"] for r in results}) == 1
    wait_for(lambda: manager.status()["state"] == "ready")
    pid = manager.status()["pid"]
    manager.set_streaming(True)
    wait_for(lambda: manager.status().get("streaming"))
    for _ in range(100):
        manager.camera_command("orbit", dx=0.01)
    manager.set_streaming(False)
    wait_for(lambda: manager.status().get("revision") == 100 and manager.status().get("streaming") is False)
    before = manager.status()["steps"]
    wait_for(lambda: manager.status()["steps"] > before)
    assert manager.status()["pid"] == pid
    assert manager.stop()["state"] == "stopped"
    assert manager.stop()["pid"] is None
    assert manager.latest_frame() is None
    assert manager.start()["pid"] != pid


def test_child_crash_is_reported_and_can_restart(manager):
    manager.start()
    wait_for(lambda: manager.status()["state"] == "ready")
    manager._process.terminate()
    manager._process.wait(timeout=5)
    assert manager.status()["state"] == "failed"
    assert manager.start()["state"] == "starting"
    wait_for(lambda: manager.status()["state"] == "ready")


def test_missing_dependency_never_spawns(manager, monkeypatch):
    monkeypatch.setattr(preview.importlib.util, "find_spec", lambda _: None)
    assert manager.start()["state"] == "failed"
    assert manager.status()["pid"] is None
    message = manager.status()["message"]
    assert ".[genesis]" in message
    assert "gui" not in message


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.01, 1.01, True, "1"])
def test_camera_rejects_invalid_input_without_changing_pose(value):
    camera = OrbitCamera()
    with pytest.raises(ValueError):
        camera.command("orbit", dx=value)
    assert camera.revision == 0


def test_camera_orbit_pan_zoom_and_reset_in_world_metres():
    camera = OrbitCamera()
    rotated = camera.command("orbit", dx=0.2, dy=0.1)
    assert rotated.snapshot()["pos"] != camera.snapshot()["pos"]
    assert rotated.target == camera.target
    panned = rotated.command("pan", dx=0.1, dy=0.1)
    assert panned.target != rotated.target
    zoomed = panned.command("zoom", delta=-0.3)
    assert zoomed.distance < panned.distance
    for _ in range(100):
        zoomed = zoomed.command("zoom", delta=-1).command("orbit", dy=1).command("pan", dy=1)
    assert zoomed.distance == 0.02
    assert zoomed.pitch == 1.48
    assert zoomed.snapshot()["pos"][2] <= 20
    reset = zoomed.command("reset")
    assert reset.snapshot()["pos"] == camera.snapshot()["pos"]
    assert reset.revision == zoomed.revision + 1


def test_fly_looks_from_fixed_eye_and_orbit_keeps_new_target():
    camera = OrbitCamera()
    looked = camera.command("fly", dx=0.25, dy=-0.6)
    assert looked.snapshot()["pos"] == pytest.approx(camera.snapshot()["pos"])
    assert looked.target != camera.target
    assert looked.pitch < 0  # Can look up, beyond the old orbit hemisphere.
    orbited = looked.command("orbit", dx=0.1)
    assert orbited.target == looked.target
    assert orbited.pitch == looked.pitch
    assert orbited.snapshot()["pos"] != looked.snapshot()["pos"]


def test_fly_translation_uses_camera_heading_and_world_up():
    camera = OrbitCamera(yaw=math.pi / 2, pitch=0, scene_scale=3)
    moved = camera.command("fly", forward=0.2, strafe=0.1, lift=0.05)
    expected = [-0.3, -0.6, 0.15]
    assert [moved.snapshot()["pos"][i] - camera.snapshot()["pos"][i] for i in range(3)] == pytest.approx(expected)
    assert [moved.target[i] - camera.target[i] for i in range(3)] == pytest.approx(expected)
    assert (moved.yaw, moved.pitch, moved.distance) == (camera.yaw, camera.pitch, camera.distance)
    combined = camera.command("fly", dx=0.5, forward=0.2)
    # After a quarter turn, forward follows -X instead of the old -Y heading.
    assert combined.snapshot()["pos"][0] == pytest.approx(camera.snapshot()["pos"][0] - 0.6)
    assert combined.snapshot()["pos"][1] == pytest.approx(camera.snapshot()["pos"][1])


@pytest.mark.parametrize("field", ["forward", "strafe", "lift"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.01, True, "1"])
def test_fly_rejects_non_finite_or_unbounded_distance(field, value):
    with pytest.raises(ValueError):
        OrbitCamera().command("fly", **{field: value})


def test_fly_limits_translation_and_pitch_without_a_pole_singularity():
    camera = OrbitCamera()
    for _ in range(100):
        camera = camera.command("fly", dy=-1, forward=1, strafe=1, lift=1)
    assert camera.pitch == -1.48
    assert all(abs(v) <= 20 + 1e-10 for v in camera.snapshot()["pos"])
    assert all(math.isfinite(v) for v in camera.snapshot()["pos"])
    assert camera.command("fly", dx=0.1).snapshot()["pos"] == pytest.approx(camera.snapshot()["pos"])


def test_orbit_eye_outside_flight_bounds_can_return_without_snapping():
    camera = OrbitCamera(yaw=0, pitch=0, target=(20, 0, 0))
    assert camera.snapshot()["pos"][0] == 23
    assert camera.command("fly", lift=0.1).snapshot()["pos"][0] == 23
    assert camera.command("fly", forward=0.1).snapshot()["pos"][0] == pytest.approx(22.9)
    assert camera.command("fly", forward=-0.1).snapshot()["pos"][0] == 23


def test_failed_spawn_and_stop_before_ready_release_mailbox(manager, monkeypatch):
    from unittest.mock import Mock

    monkeypatch.setattr(preview.subprocess, "Popen", Mock(side_effect=OSError("fixture spawn failure")))
    assert manager.start()["state"] == "failed"
    assert "fixture spawn failure" in manager.status()["message"]
    assert not list(manager.state_dir.glob("*.mmap"))
    assert manager.stop()["state"] == "stopped"


def test_stop_while_starting_leaves_no_process_or_shared_buffer(manager):
    manager.start()
    pid = manager.status()["pid"]
    assert manager.stop()["state"] == "stopped"
    assert not preview.Path(f"/proc/{pid}").exists()
    assert not list(manager.state_dir.glob("*.mmap"))


def test_packet_mailbox_has_variable_payload_and_explicit_ack(tmp_path):
    from jiuwensymbiosis.simulators.genesis.ipc import Mailbox

    path = tmp_path / "packets.mmap"
    path.touch()
    producer = Mailbox(path, frame_bytes=64)
    consumer = Mailbox(path)
    try:
        producer.write_frame({"sequence": 1}, b"A" * 64)
        assert consumer.read_frame()["h264"] == b"A" * 64
        assert producer.acknowledged() == 0  # reading status/metadata is not delivery
        consumer.acknowledge(1)
        assert producer.acknowledged() == 1
        producer.write_frame({"sequence": 2}, b"B" * 3)
        assert consumer.read_frame(1)["h264"] == b"BBB"
        assert consumer.read_frame(2) is None
        with pytest.raises(ValueError, match="capacity"):
            producer.write_frame({"sequence": 3}, b"C" * 65)
        assert consumer.read_frame()["sequence"] == 2
    finally:
        consumer.close()
        producer.close()


def test_running_preview_reconfigures_without_restarting_child(manager, monkeypatch):
    manager.start()
    wait_for(lambda: manager.status()["state"] == "ready")
    pid = manager.status()["pid"]
    before = manager.status()["steps"]
    manager.configure(width=1280, height=720, fps=60)
    send = manager._send_control
    monkeypatch.setattr(manager, "_send_control", lambda: None)
    pending = manager.configure(width=1920, height=1080, fps=30)
    assert pending["state"] == "reconfiguring"
    assert pending["settings"]["width"] == 1280
    assert pending["requested_settings"]["width"] == 1920
    with pytest.raises(RuntimeError, match="切换"):
        manager.configure(width=1280, height=720, fps=60)
    with pytest.raises(RuntimeError, match="就绪"):
        manager.camera_command("orbit", dx=0.1)
    monkeypatch.setattr(manager, "_send_control", send)
    send()
    wait_for(lambda: manager.status()["state"] == "ready")
    assert manager.status()["settings"]["width"] == 1920
    assert manager.status()["settings"]["fps"] == 30
    assert manager.status()["pid"] == pid
    assert manager.status()["steps"] >= before


def test_old_configuration_frame_is_not_delivered_after_switch(manager, monkeypatch):
    manager.start()
    wait_for(lambda: manager.status()["state"] == "ready")
    manager.set_streaming(True)
    monkeypatch.setattr(manager, "_send_control", lambda: None)
    manager.configure(width=1280, height=720, fps=30)
    manager._mailbox.write_frame({"sequence": 1, "stream_epoch": manager._stream_epoch, "config_revision": 0}, b"old")
    assert manager.latest_frame() is None
    manager._mailbox.write_frame({"sequence": 2, "stream_epoch": manager._stream_epoch, "config_revision": 1}, b"new")
    assert manager.latest_frame()["h264"] == b"new"


def test_reference_quality_switch_preserves_worker_and_physics_budget(manager):
    manager.start()
    wait_for(lambda: manager.status()["state"] == "ready")
    pid = manager.status()["pid"]
    dt = manager.settings.dt
    manager.configure(width=1280, height=720, fps=30, render_quality="reference")
    wait_for(lambda: manager.status()["state"] == "ready")
    assert manager.status()["pid"] == pid
    assert manager.settings.render_mode == "reference"
    assert manager.status()["renderer"] == "Nyx RefPathTracer"
    assert manager.settings.spp == 64
    assert manager.settings.dt == dt
    for quality in ("balanced", "high", "fast"):
        with pytest.raises(ValueError, match="仅支持 reference"):
            manager.configure(width=1280, height=720, fps=30, render_quality=quality)
    assert manager.settings.render_mode == "reference"
    assert manager.status()["renderer"] == "Nyx RefPathTracer"
    assert manager.status()["pid"] == pid


@pytest.mark.parametrize(
    "values",
    [
        {"fps": 0},
        {"fps": True},
        {"fps": 1000},
        {"width": 8192},
        {"spp": 0},
        {"dt": float("nan")},
        {"render_mode": "invalid"},
    ],
)
def test_preview_settings_reject_invalid_budgets(values):
    from jiuwensymbiosis.simulators.genesis.camera import PreviewSettings

    with pytest.raises(ValueError):
        PreviewSettings(**values)


@pytest.mark.parametrize("world_changed", [False, True])
def test_camera_catalog_validation_uses_the_requested_world(tmp_path, monkeypatch, world_changed):
    manager = preview.GenesisPreview(state_dir=tmp_path)
    manager.settings = replace(manager.settings, scene_id="room")
    manager._state = {"state": "ready", "cameras": []}
    monkeypatch.setattr(preview, "scene_path", lambda _: tmp_path)
    options = {"width": 1280, "height": 720, "fps": 30, "scene_id": "room", "camera_name": "head_cam"}
    if world_changed:
        result = manager.configure(**options, robot_id="moz1", robot_mode="moz1_wheels")
        assert result["requested_settings"]["camera_name"] == "head_cam"
    else:
        with pytest.raises(ValueError, match="没有相机"):
            manager.configure(**options)


def test_worker_environment_keeps_gpu_paths_without_host_credentials(manager, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "private-key")
    monkeypatch.setenv("UNRELATED_HOST_SECRET", "private-custom-secret")
    monkeypatch.setenv("HTTP_PROXY", "http://user:password@proxy")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/test/cuda/lib")
    monkeypatch.setenv("GS_CACHE_FILE_PATH", "/test/genesis/cache")
    monkeypatch.setenv("QD_NUM_THREADS", "2")
    original = preview.subprocess.Popen
    captured = {}

    def spawn(*args, **kwargs):
        captured.update(kwargs["env"])
        return original(*args, **kwargs)

    monkeypatch.setattr(preview.subprocess, "Popen", spawn)
    manager.start()
    wait_for(lambda: manager.status()["state"] == "ready")
    assert not {"OPENAI_API_KEY", "UNRELATED_HOST_SECRET", "HTTP_PROXY", "PYTHONPATH", "PYTHONHOME"} & captured.keys()
    assert captured["CUDA_VISIBLE_DEVICES"] == "0"
    assert captured["LD_LIBRARY_PATH"] == "/test/cuda/lib"
    assert captured["GS_CACHE_FILE_PATH"] == "/test/genesis/cache"
    assert captured["QD_NUM_THREADS"] == "2"
    assert preview.Path(captured["JIUWENSYMBIOSIS_SIMULATOR_HOME"]).is_absolute()


def test_stop_retains_unconfirmed_process_and_can_retry(tmp_path):
    from unittest.mock import Mock

    manager = preview.GenesisPreview(state_dir=tmp_path)
    process = Mock()
    process.wait.side_effect = [subprocess.TimeoutExpired("worker", 1)] * 3 + [0]
    mailbox = Mock()
    manager._process, manager._mailbox = process, mailbox
    with pytest.raises(subprocess.TimeoutExpired):
        manager.stop()
    assert manager._process is process and manager._mailbox is mailbox
    mailbox.close.assert_not_called()
    assert manager.stop()["state"] == "stopped"
    assert manager._process is None and manager._mailbox is None
    mailbox.close.assert_called_once()


def test_oversized_status_preserves_previous_snapshot_and_allows_failure_report(tmp_path):
    from jiuwensymbiosis.simulators.genesis.ipc import Mailbox

    path = tmp_path / "status.mmap"
    path.touch()
    mailbox = Mailbox(path, frame_bytes=64)
    try:
        mailbox.write_status({"state": "ready"})
        with pytest.raises(ValueError):
            mailbox.write_status({"payload": "x" * 16384})
        assert mailbox.read_status() == {"state": "ready"}
        mailbox.write_status({"state": "failed", "message": "status capacity exceeded"})
        assert mailbox.read_status()["state"] == "failed"
    finally:
        mailbox.close()


def test_invalid_worker_data_root_is_rejected_before_allocating_resources(manager, monkeypatch):
    monkeypatch.setenv("JIUWENSYMBIOSIS_SIMULATOR_HOME", "relative/path")
    with pytest.raises(ValueError, match="absolute path"):
        manager.start()
    assert manager._process is None and manager._mailbox is None
    assert not manager.state_dir.exists()


@pytest.mark.parametrize("size", ["", "0", "-1", "2147483648", "1.0", "1e6", " 1", "1 ", "+1", "１", "9" * 5000])
def test_invalid_quadrants_cache_size_is_rejected_before_allocating_resources(manager, monkeypatch, size):
    monkeypatch.setenv("QD_OFFLINE_CACHE_MAX_SIZE_OF_FILES", size)
    with pytest.raises(ValueError, match="1..2147483647"):
        manager.start()
    assert manager._process is None and manager._mailbox is None
    assert not manager.state_dir.exists()


@pytest.mark.parametrize("size", ["1", "104857600", "2147483647"])
def test_quadrants_cache_size_preserves_explicit_boundary_values(monkeypatch, size):
    monkeypatch.setenv("QD_OFFLINE_CACHE_MAX_SIZE_OF_FILES", size)
    assert preview._worker_environment()["QD_OFFLINE_CACHE_MAX_SIZE_OF_FILES"] == size


def test_unspecified_quadrants_cache_keeps_sdk_defaults(monkeypatch):
    monkeypatch.delenv("QD_OFFLINE_CACHE_MAX_SIZE_OF_FILES", raising=False)
    monkeypatch.delenv("QD_OFFLINE_CACHE_FILE_PATH", raising=False)
    assert (
        not {"QD_OFFLINE_CACHE_FILE_PATH", "QD_OFFLINE_CACHE_MAX_SIZE_OF_FILES"} & preview._worker_environment().keys()
    )


def test_quadrants_cache_path_is_resolved_before_worker_changes_cwd(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    monkeypatch.chdir(parent)
    monkeypatch.setenv("QD_OFFLINE_CACHE_FILE_PATH", "../cache")
    environment = preview._worker_environment()
    assert environment["QD_OFFLINE_CACHE_FILE_PATH"] == str(tmp_path / "cache")
    assert not (tmp_path / "cache").exists()  # Validation never populates or clears it.
    monkeypatch.setenv("HOME", str(parent))
    monkeypatch.setenv("QD_OFFLINE_CACHE_FILE_PATH", "~/cache")
    assert preview._worker_environment()["QD_OFFLINE_CACHE_FILE_PATH"] == str(parent / "cache")


@pytest.mark.parametrize("path", ["", " ", "cache\npath", "cache\rpath"])
def test_invalid_quadrants_cache_path_is_rejected(manager, monkeypatch, path):
    monkeypatch.setenv("QD_OFFLINE_CACHE_FILE_PATH", path)
    with pytest.raises(ValueError, match="QD_OFFLINE_CACHE_FILE_PATH"):
        manager.start()
    assert manager._process is None and manager._mailbox is None
    assert not manager.state_dir.exists()


def test_quadrants_cache_path_rejects_an_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "file"
    path.write_text("keep")
    monkeypatch.setenv("QD_OFFLINE_CACHE_FILE_PATH", str(path))
    with pytest.raises(ValueError, match="must name a directory"):
        preview._worker_environment()
    assert path.read_text() == "keep"
