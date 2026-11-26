# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Parent-side one-shot exchanges: send, wait by id, clean only our own slot."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.simulators.common import preview as preview_module
from jiuwensymbiosis.simulators.common.preview import SimulationPreview
from jiuwensymbiosis.simulators.common.protocol import (
    build_base_command_request,
    build_capture_reply,
    build_pick_reply,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0
        self.step = 0.02

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(seconds, self.step)


class _Harness:
    """A ready status whose replies are produced by the test callback."""

    def __init__(self, preview: SimulationPreview, clock: _Clock) -> None:
        self.preview = preview
        self.clock = clock
        self.state = {
            "state": "ready",
            "scene": {"instance_id": "scene"},
            "cameras": [{"name": "wrist"}],
            "camera": {"pos": [0, 0, 0]},
            "selection": None,
            "camera_fov_y": 60.0,
        }
        self.on_request = lambda request: None

    def status(self) -> dict:
        request = self.preview._capture_request or self.preview._pick_request or self.preview._scene_query
        if request is not None:
            self.on_request(request)
        self.state["measured_at"] = self.clock.now
        return {**self.state}


def _make_preview(tmp_path: Path, monkeypatch) -> tuple[SimulationPreview, _Harness]:
    preview = SimulationPreview(state_dir=tmp_path)
    clock = _Clock()
    monkeypatch.setattr(preview_module, "time", clock)
    preview._camera = SimpleNamespace(revision=5, snapshot=lambda: {"pos": [0, 0, 0]})
    harness = _Harness(preview, clock)
    monkeypatch.setattr(preview, "status", harness.status)
    return preview, harness


def test_capture_success_returns_the_reply_and_cleans_its_files(tmp_path, monkeypatch):
    preview, harness = _make_preview(tmp_path, monkeypatch)

    def reply(request):
        path = preview.state_dir / f"capture-{request['id']}.jpg"
        path.write_bytes(b"jpeg-bytes")
        harness.state["capture_result"] = build_capture_reply(
            request["id"],
            captured_at=harness.clock.now,
            captured_utc="2026-01-01T00:00:00.000+00:00",
            scene_instance_id="scene",
            config_revision=0,
            camera_revision=5,
            camera_name="observer",
            camera={"pos": [0, 0, 0]},
            step=1,
            width=2,
            height=2,
            capture_duration_ms=1.0,
        )
        harness.on_request = lambda _request: None

    harness.on_request = reply
    observation = preview.capture_image()

    assert observation["jpeg"] == b"jpeg-bytes"
    assert preview._capture_request is None
    assert not (preview.state_dir / f"capture-{observation['id']}.jpg").exists()


def test_capture_failures_stay_request_scoped(tmp_path, monkeypatch):
    preview, harness = _make_preview(tmp_path, monkeypatch)

    def fail(request):
        harness.state["capture_result"] = {"id": request["id"], "error": "capture_failed"}
        harness.on_request = lambda _request: None

    harness.on_request = fail
    with pytest.raises(RuntimeError, match="相机采样失败"):
        preview.capture_image()
    assert preview._capture_request is None

    # A reply for another id never completes this call.
    preview, harness = _make_preview(tmp_path, monkeypatch)
    harness.clock.step = 13.0
    harness.on_request = lambda request: harness.state.update(
        capture_result={"id": "another-request", "captured_at": harness.clock.now}
    )
    with pytest.raises(RuntimeError, match="相机采样超时"):
        preview.capture_image()
    assert preview._capture_request is None

    # A stopped worker aborts the wait through the same admission check.
    preview, harness = _make_preview(tmp_path, monkeypatch)
    harness.on_request = lambda _request: harness.state.update(state="stopped")
    with pytest.raises(RuntimeError, match="场景或视角已变化"):
        preview.capture_image()


def test_capture_cancel_scene_change_and_observer_view_abort_the_wait(tmp_path, monkeypatch):
    preview, harness = _make_preview(tmp_path, monkeypatch)
    cancelled = threading.Event()
    harness.on_request = lambda _request: cancelled.set()
    with pytest.raises(RuntimeError, match="已取消"):
        preview.capture_image(cancelled=cancelled)

    preview, harness = _make_preview(tmp_path, monkeypatch)
    harness.on_request = lambda _request: harness.state.update(scene={"instance_id": "other"})
    with pytest.raises(RuntimeError, match="场景或视角已变化"):
        preview.capture_image()

    # The observer view is bound to the camera revision; a named camera is not.
    preview, harness = _make_preview(tmp_path, monkeypatch)
    harness.on_request = lambda _request: setattr(preview._camera, "revision", 6)
    with pytest.raises(RuntimeError, match="场景或视角已变化"):
        preview.capture_image()

    preview, harness = _make_preview(tmp_path, monkeypatch)

    def reply(request):
        (preview.state_dir / f"capture-{request['id']}.jpg").write_bytes(b"wrist")
        preview._camera.revision = 6
        harness.state["capture_result"] = build_capture_reply(
            request["id"],
            captured_at=harness.clock.now,
            captured_utc="2026-01-01T00:00:00.000+00:00",
            scene_instance_id="scene",
            config_revision=0,
            camera_revision=5,
            camera_name="wrist",
            camera={"pos": [0, 0, 0]},
            step=1,
            width=2,
            height=2,
            capture_duration_ms=1.0,
        )
        harness.on_request = lambda _request: None

    harness.on_request = reply
    assert preview.capture_image("wrist")["jpeg"] == b"wrist"


def test_capture_busy_and_a_later_request_are_not_touched(tmp_path, monkeypatch):
    preview, harness = _make_preview(tmp_path, monkeypatch)
    preview._capture_request = {"id": "busy"}
    with pytest.raises(RuntimeError, match="另一张画面正在采样"):
        preview.capture_image()
    assert preview._capture_request == {"id": "busy"}

    preview, harness = _make_preview(tmp_path, monkeypatch)
    harness.clock.step = 13.0
    newer = {"id": "newer"}

    def replace(request):
        preview._capture_request = newer
        harness.state["capture_result"] = {"id": "another-request"}

    harness.on_request = replace
    with pytest.raises(RuntimeError, match="相机采样超时"):
        preview.capture_image()
    assert preview._capture_request is newer


def test_pick_success_and_stale_binding_errors(tmp_path, monkeypatch):
    preview, harness = _make_preview(tmp_path, monkeypatch)

    def reply(request):
        harness.state["pick_result"] = build_pick_reply(request["id"], duration_ms=2.0)
        harness.state["selection"] = {"name": "cup"}
        harness.state["camera"] = {"pos": [1, 2, 3]}
        harness.on_request = lambda _request: None

    harness.on_request = reply
    result = preview.select(
        action="pick", u=0.5, v=0.5, scene_instance_id="scene", config_revision=0, camera_revision=5
    )
    assert result == {"selection": {"name": "cup"}, "camera": {"pos": [1, 2, 3]}, "camera_fov_y": 60.0}
    assert preview._pick_request is None

    with pytest.raises(ValueError, match="场景已经切换"):
        preview.select(action="pick", u=0.5, v=0.5, scene_instance_id="other", config_revision=0, camera_revision=5)
    with pytest.raises(ValueError, match="视角已经变化"):
        preview.select(action="pick", u=0.5, v=0.5, scene_instance_id="scene", config_revision=0, camera_revision=4)

    preview, harness = _make_preview(tmp_path, monkeypatch)

    def clear_reply(request):
        harness.state["pick_result"] = build_pick_reply(request["id"], duration_ms=1.0)
        harness.state["selection"] = None
        harness.on_request = lambda _request: None

    harness.on_request = clear_reply
    result = preview.select(
        action="clear", u=0.5, v=0.5, scene_instance_id="scene", config_revision=0, camera_revision=4
    )
    assert result["selection"] is None


def test_pick_worker_error_timeout_and_later_request(tmp_path, monkeypatch):
    preview, harness = _make_preview(tmp_path, monkeypatch)

    def fail(request):
        harness.state["pick_result"] = {**build_pick_reply(request["id"], duration_ms=1.0), "error": "no surface"}
        harness.on_request = lambda _request: None

    harness.on_request = fail
    with pytest.raises(ValueError, match="no surface"):
        preview.select(action="pick", u=0.5, v=0.5, scene_instance_id="scene", config_revision=0, camera_revision=5)

    preview, harness = _make_preview(tmp_path, monkeypatch)
    harness.clock.step = 11.0
    newer = {"id": "newer-pick"}

    def replace(request):
        preview._pick_request = newer

    harness.on_request = replace
    with pytest.raises(RuntimeError, match="点选未在时限内确认"):
        preview.select(action="pick", u=0.5, v=0.5, scene_instance_id="scene", config_revision=0, camera_revision=5)
    assert preview._pick_request is newer


def test_query_scene_returns_data_and_handles_errors_timeouts_and_busy(tmp_path, monkeypatch):
    preview, harness = _make_preview(tmp_path, monkeypatch)

    def reply(request):
        path = preview.state_dir / f"query-{request['id']}.json"
        path.write_text(json.dumps({"scene_instance_id": "scene", "objects": []}), encoding="utf-8")
        harness.state["scene_query_result"] = {"id": request["id"]}
        harness.on_request = lambda _request: None

    harness.on_request = reply
    assert preview.query_scene("objects", {"kind": "cup"}) == {"scene_instance_id": "scene", "objects": []}
    assert preview._scene_query is None
    assert not list(preview.state_dir.glob("query-*.json"))

    preview, harness = _make_preview(tmp_path, monkeypatch)
    harness.on_request = lambda _request: harness.state.update(scene={"instance_id": "other"})
    with pytest.raises(RuntimeError, match="Scene changed during query"):
        preview.query_scene("objects")

    preview, harness = _make_preview(tmp_path, monkeypatch)

    def fail(request):
        harness.state["scene_query_result"] = {"id": request["id"], "error": "unknown operation"}
        harness.on_request = lambda _request: None

    harness.on_request = fail
    with pytest.raises(ValueError, match="unknown operation"):
        preview.query_scene("objects")

    preview, harness = _make_preview(tmp_path, monkeypatch)
    harness.clock.step = 61.0
    newer = {"id": "newer-query"}
    harness.on_request = lambda _request: setattr(preview, "_scene_query", newer)
    with pytest.raises(TimeoutError):
        preview.query_scene("objects")
    assert preview._scene_query is newer

    preview, harness = _make_preview(tmp_path, monkeypatch)
    preview._scene_query = {"id": "busy"}
    with pytest.raises(RuntimeError, match="already running"):
        preview.query_scene("objects")
    with pytest.raises(ValueError, match="bounded request budget"):
        preview.query_scene("objects", {"payload": "x" * 3000})


def test_base_command_uses_the_shared_request_and_control_slots(tmp_path, monkeypatch):
    preview, harness = _make_preview(tmp_path, monkeypatch)
    preview._streaming = True
    harness.state["robot"] = {"control_mode": "wheel_velocity_physics", "command_limits": {}}

    accepted = preview.base_command(vx=0.1, vy=0.0, wz=0.0, scene_instance_id="scene", config_revision=0)

    assert accepted["queued_id"] == preview._base_request["id"]
    assert preview._base_request == build_base_command_request(
        preview._base_request["id"],
        vx=0.1,
        vy=0.0,
        wz=0.0,
        scene_instance_id="scene",
        config_revision=0,
        expires_at=preview._base_request["expires_at"],
    )

    class _Mailbox:
        control = None

        def write_control(self, value):
            self.control = value

    preview._mailbox = _Mailbox()
    preview._send_control()
    assert tuple(preview._mailbox.control) == (
        "streaming",
        "camera",
        "settings",
        "config_revision",
        "stream_epoch",
        "keyframe_request",
        "pick",
        "base_command",
        "base_motion",
        "capture",
        "scene_query",
        "reset",
    )
    assert preview._mailbox.control["base_command"] is preview._base_request


def test_capture_rgbd_reads_depth_arrays_and_cleans_both_files(tmp_path, monkeypatch):
    preview, harness = _make_preview(tmp_path, monkeypatch)

    def reply(request):
        (preview.state_dir / f"capture-{request['id']}.jpg").write_bytes(b"jpeg")
        np.savez(
            preview.state_dir / f"capture-{request['id']}.npz",
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            depth_m=np.ones((2, 2), dtype=np.float32),
        )
        harness.state["capture_result"] = build_capture_reply(
            request["id"],
            captured_at=harness.clock.now,
            captured_utc="2026-01-01T00:00:00.000+00:00",
            scene_instance_id="scene",
            config_revision=0,
            camera_revision=5,
            camera_name="wrist",
            camera={"pos": [0, 0, 0]},
            step=1,
            width=2,
            height=2,
            capture_duration_ms=1.0,
            base_sample={"position_m": [0, 0, 0]},
            body_bounds_m=[[0, 0, 0], [1, 1, 1]],
            motion_id="motion-1",
        )
        harness.on_request = lambda _request: None

    harness.on_request = reply
    observation = preview.capture_image("wrist", include_depth=True, depth_stride=8)

    assert observation["rgb"].shape == (2, 2, 3)
    assert observation["depth_m"].dtype == np.float32
    assert not list(preview.state_dir.glob("capture-*"))


def test_failed_control_write_releases_the_reserved_slot(tmp_path, monkeypatch):
    preview, harness = _make_preview(tmp_path, monkeypatch)
    harness.state["robot"] = {"control_mode": "wheel_velocity_physics", "command_limits": {}}

    class _Boom:
        def write_control(self, value):
            raise RuntimeError("mailbox budget exceeded")

    preview._mailbox = _Boom()
    with pytest.raises(RuntimeError, match="mailbox budget exceeded"):
        preview.capture_image()
    assert preview._capture_request is None
