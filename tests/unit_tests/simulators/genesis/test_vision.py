"""Pixels must reach the model without scene truth, stale frames or motion tools."""

import asyncio
import base64
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from jiuwensymbiosis.agent.config import ModelSpec
from jiuwensymbiosis.simulators.genesis import vision
from jiuwensymbiosis.simulators.genesis.chat import ChatError, GenesisChat
from jiuwensymbiosis.simulators.genesis.preview import GenesisPreview


@pytest.fixture
def capture():
    return {
        "jpeg": b"jpeg-pixels",
        "captured_utc": "2026-10-17T12:00:00+00:00",
        "captured_at": time.monotonic(),
        "scene_instance_id": "private-scene-identity",
        "config_revision": 2,
        "camera_revision": 3,
        "width": 1280,
        "height": 720,
    }


@pytest.fixture
def setup_vision(monkeypatch, tmp_path, capture):
    monkeypatch.setattr(vision, "OUTPUT_DIR", tmp_path)
    invoke = AsyncMock(return_value=SimpleNamespace(content="桌子与椅子", tool_calls=[]))
    preview = SimpleNamespace(capture_image=Mock(return_value=capture))
    chat = GenesisChat(
        lambda: {"object_positions": [[999, 999, 999]], "scene": "private-scene-identity"},
        model_factory=lambda _: SimpleNamespace(invoke=invoke),
        preview=preview,
    )
    return chat, invoke


async def test_image_only_request_even_with_motion_permission(setup_vision, capture, tmp_path):
    chat, invoke = setup_vision
    chat.actions_enabled = True
    result = await chat.recognize(ModelSpec())
    messages = invoke.call_args.args[0]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "private-scene-identity" not in json.dumps(messages)
    assert "object_positions" not in json.dumps(messages)
    assert "tools" not in invoke.call_args.kwargs
    uri = messages[-1]["content"][1]["image_url"]["url"]
    assert base64.b64decode(uri.split(",", 1)[1]) == capture["jpeg"]
    assert (tmp_path / f"{result['id']}.jpg").read_bytes() == capture["jpeg"]
    assert vision.latest_result()["reply"] == "桌子与椅子"
    assert not chat.busy and not chat._history and chat.last_execution is None


async def test_capture_failure_does_not_call_model(setup_vision, tmp_path):
    chat, invoke = setup_vision
    chat.preview.capture_image.side_effect = RuntimeError("stale")
    with pytest.raises(ChatError, match="采样失败"):
        await chat.recognize(ModelSpec())
    invoke.assert_not_awaited()
    assert not chat.busy and vision.latest_result() is None


async def test_cancel_and_concurrent_request_do_not_publish_late_result(setup_vision):
    chat, invoke = setup_vision
    entered = asyncio.Event()

    async def slow(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return SimpleNamespace(content="late", tool_calls=[])

    invoke.side_effect = slow
    task = asyncio.create_task(chat.recognize(ModelSpec()))
    await entered.wait()
    with pytest.raises(ChatError, match="正在生成"):
        await chat.recognize(ModelSpec())
    chat.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert vision.latest_result() is None and not chat.busy


async def test_credentials_are_not_saved_and_bad_image_digest_is_rejected(setup_vision, tmp_path):
    chat, invoke = setup_vision
    invoke.return_value.content = "credential-secret"
    result = await chat.recognize(ModelSpec(api_key="credential-secret"))
    assert "credential-secret" not in (tmp_path / "latest.json").read_text()
    (tmp_path / f"{result['id']}.jpg").write_bytes(b"different")
    assert vision.latest_result() is None


@pytest.mark.parametrize("camera_name", ["observer", "head_cam"])
@pytest.mark.parametrize("changed", [None, "camera", "scene", "config", "stopped", "name", "expired"])
def test_capture_requires_matching_fresh_camera_without_video(monkeypatch, tmp_path, changed, camera_name):
    preview = GenesisPreview(state_dir=tmp_path)
    preview._config_revision = 2
    state = {
        "state": "ready",
        "measured_at": time.monotonic(),
        "scene": {"instance_id": "scene"},
        "cameras": [{"name": "head_cam"}],
    }
    monkeypatch.setattr(preview, "status", lambda: state)

    def deliver():
        request = preview._capture_request
        if request is None:
            return
        result = {**request, "captured_at": time.monotonic()}
        if changed == "camera":
            result["camera_revision"] += 1
        elif changed == "scene":
            state["scene"] = {"instance_id": "other"}
        elif changed == "config":
            preview._config_revision += 1
        elif changed == "stopped":
            state["state"] = "stopped"
        elif changed == "name":
            result["camera_name"] = "left_gripper_cam"
        elif changed == "expired":
            result["captured_at"] -= 10
        state["capture_result"] = result
        (tmp_path / f"capture-{request['id']}.jpg").write_bytes(b"fresh-pixels")

    monkeypatch.setattr(preview, "_send_control", deliver)
    if changed:
        with pytest.raises(RuntimeError):
            preview.capture_image(camera_name)
    else:
        assert preview.capture_image(camera_name)["jpeg"] == b"fresh-pixels"
    assert preview._capture_request is None
    assert not list(tmp_path.glob("capture-*.jpg"))
    assert not preview._streaming


def test_named_capture_is_independent_of_observer_navigation(monkeypatch, tmp_path):
    from dataclasses import replace

    preview = GenesisPreview(state_dir=tmp_path)
    state = {
        "state": "ready",
        "measured_at": time.monotonic(),
        "scene": {"instance_id": "scene"},
        "cameras": [{"name": "head_cam"}],
    }
    monkeypatch.setattr(preview, "status", lambda: state)

    def deliver():
        request = preview._capture_request
        if request is not None:
            assert request["camera_name"] == "head_cam"
            preview._camera = replace(preview._camera, revision=preview._camera.revision + 1)
            state["capture_result"] = {**request, "captured_at": time.monotonic()}
            (tmp_path / f"capture-{request['id']}.jpg").write_bytes(b"head-pixels")

    monkeypatch.setattr(preview, "_send_control", deliver)
    assert preview.capture_image("head_cam")["jpeg"] == b"head-pixels"
    with pytest.raises(ValueError, match="没有相机"):
        preview.capture_image("missing_cam")
    assert preview._capture_request is None


@pytest.mark.parametrize("cancel", [False, True])
def test_stop_releases_capture_and_concurrent_names_are_rejected(monkeypatch, tmp_path, cancel):
    preview = GenesisPreview(state_dir=tmp_path)
    state = {
        "state": "ready",
        "measured_at": time.monotonic(),
        "scene": {"instance_id": "scene"},
        "cameras": [{"name": "head_cam"}, {"name": "left_gripper_cam"}],
    }
    monkeypatch.setattr(preview, "status", lambda: state)
    requested = threading.Event()
    cancelled = threading.Event()
    monkeypatch.setattr(preview, "_send_control", requested.set)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(preview.capture_image, "head_cam", cancelled=cancelled)
        try:
            assert requested.wait(timeout=2)
            with pytest.raises(RuntimeError, match="另一张画面"):
                preview.capture_image("left_gripper_cam")
        finally:
            if cancel:
                cancelled.set()
            else:
                with preview._lock:
                    state["state"] = "stopped"
        with pytest.raises(RuntimeError, match="取消" if cancel else "采样期间"):
            pending.result(timeout=2)
    assert preview._capture_request is None
    assert not list(tmp_path.glob("capture-*.jpg"))
