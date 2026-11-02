"""Named camera delivery, permissions and real SDK multimodal serialization."""

import asyncio
import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from jiuwensymbiosis.agent.config import ModelSpec
from jiuwensymbiosis.simulators.genesis.agent import make_session, run_agent_query
from jiuwensymbiosis.simulators.genesis.agent_camera import CameraObservation
from jiuwensymbiosis.simulators.genesis.chat import GenesisChat
from jiuwensymbiosis.tools.builder import list_tool_meta


@pytest.fixture
def preview():
    state = {
        "state": "ready",
        "scene": {"instance_id": "PRIVATE_SCENE_42", "id": "PRIVATE_ASSET_HOME"},
        "config_revision": 1,
        "cameras": [{"name": "head_cam"}, {"name": "left_gripper_cam"}],
    }

    def capture(name, **kwargs):
        return {
            "jpeg": b"exact-camera-jpeg",
            "camera_name": name,
            "captured_at": time.monotonic(),
            "captured_utc": "2026-10-19T00:00:00Z",
            "scene_instance_id": "PRIVATE_SCENE_42",
            "config_revision": 1,
            "step": 120,
            "width": 640,
            "height": 480,
            "camera": {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "T_world_camera": "PRIVATE_EXTRINSICS"},
        }

    return SimpleNamespace(
        status=lambda: {**state, "measured_at": time.monotonic()},
        state=state,
        capture_image=Mock(side_effect=capture),
        acquire_motion=Mock(),
        release_motion=Mock(),
    )


def test_readonly_session_has_no_base_capability_or_motion_lease(preview):
    with make_session(preview, actions_enabled=False) as session:
        assert not session.api.capabilities
        assert {m["name"] for m in list_tool_meta(session.api, env=session.env)} == {"home"}
        assert session.env.get_observation().extra == {}
    preview.acquire_motion.assert_not_called()
    preview.release_motion.assert_not_called()


@pytest.mark.parametrize("changed", [{"camera_name": "observer"}, {"captured_at": 0}])
async def test_mismatched_or_stale_frame_is_not_saved(preview, tmp_path, changed):
    original = preview.capture_image.side_effect
    preview.capture_image.side_effect = lambda name, **kwargs: {**original(name), **changed}
    camera = CameraObservation(preview, threading.Event(), tmp_path, "run", Mock())
    with pytest.raises(RuntimeError, match="不匹配或已过期"):
        await camera.capture("head_cam")
    assert not camera.frames and not camera.pending and not list(tmp_path.iterdir())


async def test_exact_pixels_staged_until_tool_reply_and_bounded(preview, tmp_path):
    shown = Mock()
    camera = CameraObservation(preview, threading.Event(), tmp_path, "run", shown)
    ctx = SimpleNamespace(context=SimpleNamespace(add_messages=AsyncMock()), request_force_finish=Mock())
    result = await camera.capture("head_cam")
    assert set(result) == {"camera_name", "id", "image_sha256", "image_delivery"}
    shown.assert_not_called()
    ctx.context.add_messages.assert_not_awaited()
    await camera.before_model_call(ctx)
    message = ctx.context.add_messages.call_args.args[0]
    url = message.content[1]["image_url"]["url"]
    assert base64.b64decode(url.split(",", 1)[1]) == b"exact-camera-jpeg"
    assert shown.call_args.args[0]["image_url"] == url
    assert camera.frames[0]["injected"]
    assert (tmp_path / "run-1.jpg").read_bytes() == b"exact-camera-jpeg"
    for _ in range(3):
        await camera.capture("head_cam")
    assert "error" in await camera.capture("head_cam")
    assert preview.capture_image.call_count == 4


@pytest.mark.parametrize("change", ["cancel", "scene", "revision", "stopped"])
async def test_late_images_are_not_delivered_after_invalidation(preview, tmp_path, change):
    cancelled, shown = threading.Event(), Mock()
    camera = CameraObservation(preview, cancelled, tmp_path, "run", shown)
    await camera.capture("head_cam")
    if change == "cancel":
        cancelled.set()
    elif change == "scene":
        preview.state["scene"] = {"instance_id": "another"}
    elif change == "revision":
        preview.state["config_revision"] += 1
    else:
        preview.state["state"] = "stopped"
    ctx = SimpleNamespace(context=SimpleNamespace(add_messages=AsyncMock()), request_force_finish=Mock())
    await camera.before_model_call(ctx)
    ctx.context.add_messages.assert_not_awaited()
    shown.assert_not_called()
    ctx.request_force_finish.assert_called_once()
    assert not camera.pending


async def test_failed_delivery_stops_instead_of_claiming_vision(preview, tmp_path):
    shown = Mock()
    camera = CameraObservation(preview, threading.Event(), tmp_path, "run", shown)
    with pytest.raises(ValueError):
        await camera.capture("not-a-camera")
    preview.capture_image.assert_not_called()
    await camera.capture("head_cam")
    ctx = SimpleNamespace(
        context=SimpleNamespace(add_messages=AsyncMock(side_effect=RuntimeError())), request_force_finish=Mock()
    )
    await camera.before_model_call(ctx)
    assert camera.error and not camera.frames[0]["injected"]
    shown.assert_not_called()
    ctx.request_force_finish.assert_called_once()


async def test_readonly_chat_reuses_agent_and_discards_late_ui_callback(monkeypatch, preview):
    captured, shown = {}, Mock()

    def run(*args, **kwargs):
        captured.update(kwargs)
        kwargs["on_frame"]({"id": "first"})
        return {"reply": "看到了图片", "frames": [{"id": "first"}]}

    monkeypatch.setattr("jiuwensymbiosis.simulators.genesis.agent.run_agent_query", run)
    chat = GenesisChat(preview.status, preview=preview)
    assert await chat.send("看看头部画面", ModelSpec(), on_frame=shown) == "看到了图片"
    assert captured["actions_enabled"] is False
    shown.assert_called_once_with({"id": "first"})
    chat.reset()
    captured["on_frame"]({"id": "late"})
    await asyncio.sleep(0)
    assert shown.call_count == 1


@pytest.mark.parametrize("default_output", [False, True])
def test_shared_agent_sends_camera_pixels_to_model_without_motion(preview, tmp_path, monkeypatch, default_output):
    """Local HTTP model drives a real tool call, then checks the next wire request."""
    monkeypatch.setenv("JIUWENSYMBIOSIS_SIMULATOR_HOME", str(tmp_path))
    expected_output = tmp_path / "outputs/genesis-agent" if default_output else tmp_path
    requests, tool_requests, shown = [], [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            # The SDK may concurrently probe image support without any tools.
            # Respond independently so that the probe cannot consume the camera call.
            if request.get("tools"):
                tool_requests.append(request)
            call_camera = bool(request.get("tools")) and not any(
                item.get("role") == "tool" for item in request.get("messages", [])
            )
            if call_camera:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "capture-1",
                            "type": "function",
                            "function": {
                                "name": "capture_camera",
                                "arguments": json.dumps({"camera_name": "head_cam"}),
                            },
                        }
                    ],
                }
            else:
                message = {
                    "role": "assistant",
                    "content": "已根据头部相机图片回答。" if request.get("tools") else "red",
                }
            payload = {
                "id": "test",
                "object": "chat.completion",
                "created": 1,
                "model": "camera-test",
                "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if call_camera else "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = run_agent_query(
                preview,
                ModelSpec(api_base=f"http://127.0.0.1:{server.server_port}/v1", model_name="camera-test"),
                "看看头部画面",
                actions_enabled=False,
                output_dir=None if default_output else tmp_path,
                on_frame=shown.append,
            )
        finally:
            server.shutdown()
            thread.join(2)
    assert (expected_output / f"{result['id']}.json").is_file()
    assert len(tool_requests) >= 2
    assert all({t["function"]["name"] for t in request["tools"]} == {"capture_camera"} for request in tool_requests)
    messages = tool_requests[-1]["messages"]
    image_message = next(
        m
        for m in messages
        if isinstance(m.get("content"), list) and any(c.get("type") == "image_url" for c in m["content"])
    )
    image_index = messages.index(image_message)
    assert messages[image_index - 1]["role"] == "tool"
    image = next(c["image_url"]["url"] for c in image_message["content"] if c.get("type") == "image_url")
    assert base64.b64decode(image.split(",", 1)[1]) == b"exact-camera-jpeg"
    wire = json.dumps(requests)
    for forbidden in (
        "PRIVATE_SCENE_42",
        "PRIVATE_ASSET_HOME",
        "PRIVATE_EXTRINSICS",
        "scene_instance_id",
        "config_revision",
        "T_world_camera",
        "object_positions",
    ):
        assert forbidden not in wire
    assert tool_requests[0]["tools"][0]["function"]["parameters"]["properties"]["camera_name"]["enum"] == ["head_cam"]
    assert "PRIVATE_EXTRINSICS" in json.dumps(result["frames"])  # private audit stays available
    assert shown[0]["image_url"] == image
    assert result["frames"][0]["injected"] and not result["actions"]
    assert preview.capture_image.call_count == 1
    preview.acquire_motion.assert_not_called()


@pytest.mark.parametrize("name", ["observer", "left_gripper_cam", "right_gripper_cam"])
async def test_non_head_camera_cannot_be_requested(preview, tmp_path, name):
    camera = CameraObservation(preview, threading.Event(), tmp_path, "run", None)
    with pytest.raises(ValueError, match="仅允许 head_cam"):
        await camera.capture(name)
    preview.capture_image.assert_not_called()


async def test_missing_head_camera_does_not_fall_back(preview, tmp_path):
    preview.state["cameras"] = [{"name": "left_gripper_cam"}]
    camera = CameraObservation(preview, threading.Event(), tmp_path, "run", None)
    with pytest.raises(RuntimeError, match="头部相机不可用"):
        await camera.capture("head_cam")
    preview.capture_image.assert_not_called()


async def test_camera_error_does_not_expose_backend_diagnostics(preview, tmp_path):
    preview.capture_image.side_effect = RuntimeError("PRIVATE_ASSET_HOME at (123, 456)")
    camera = CameraObservation(preview, threading.Event(), tmp_path, "run", None)
    with pytest.raises(RuntimeError, match="^头部相机取图失败，请稍后重新取图。$"):
        await camera.capture("head_cam")


@pytest.mark.parametrize("stage", ["capture", "delivery"])
async def test_failure_diagnostics_are_actionable_without_private_payload(preview, tmp_path, caplog, stage):
    secret = "PRIVATE_KEY=secret data:image/jpeg;base64,private-pixels"
    camera = CameraObservation(preview, threading.Event(), tmp_path, "run", None)
    if stage == "capture":
        preview.capture_image.side_effect = TimeoutError(secret)
        with pytest.raises(RuntimeError) as raised:
            await camera.capture("head_cam")
        assert secret not in str(raised.value)
        code = "genesis.camera.capture_failed"
    else:
        await camera.capture("head_cam")
        ctx = SimpleNamespace(
            context=SimpleNamespace(add_messages=AsyncMock(side_effect=TimeoutError(secret))),
            request_force_finish=Mock(),
        )
        await camera.before_model_call(ctx)
        assert camera.error and not camera.pending
        assert not camera.frames[0]["injected"]
        ctx.request_force_finish.assert_called_once()
        code = "genesis.camera.delivery_failed"
    assert code in caplog.text and "TimeoutError" in caplog.text
    assert "PRIVATE_KEY" not in caplog.text and "private-pixels" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


async def test_vendor_exception_class_name_cannot_leak_credentials(preview, tmp_path, caplog):
    private_error = type("PRIVATE_API_KEY_and_image_payload", (Exception,), {})
    preview.capture_image.side_effect = private_error("PRIVATE_RAW_BODY")
    camera = CameraObservation(preview, threading.Event(), tmp_path, "run", None)
    with pytest.raises(RuntimeError):
        await camera.capture("head_cam")
    assert "genesis.camera.capture_failed (Exception)" in caplog.text
    assert "PRIVATE" not in caplog.text
