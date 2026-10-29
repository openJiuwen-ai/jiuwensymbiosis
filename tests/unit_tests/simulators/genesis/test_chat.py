"""Read-only state grounding, bounded context and async conversation lifecycle."""

import asyncio
import json
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from jiuwensymbiosis.agent.config import ModelSpec
from jiuwensymbiosis.simulators.genesis.chat import ChatError, GenesisChat, state_snapshot
from tests.helpers import genesis_ready_state as ready


def test_snapshot_has_units_and_freshness_but_no_logs_or_session_secrets():
    raw = ready()
    snapshot = state_snapshot(raw)
    assert snapshot["fresh"] and snapshot["sample_age_s"] < 1
    assert snapshot["coordinate_frame"] == "world; metres; Z-up"
    assert snapshot["robot_actions_available"] is False
    assert not {"log_path", "message", "session_id", "measured_at"} & snapshot.keys()
    raw["object_positions"][0][0] = 9
    assert snapshot["object_positions"] == [[1, 2, 3]]


def test_static_background_is_distinguished_from_empty_dynamic_scene():
    raw = {
        **ready(),
        "object_positions": [],
        "scene": {
            "id": "library://geniesim/background/kitchen/kitchen_1/background",
            "mode": "static_background",
            "collision": False,
            "visible_meshes": 22,
            "source_path": "/private/asset",
            "missing_assets": ["/private/texture"],
        },
    }
    snapshot = state_snapshot(raw)
    assert snapshot["scene"] == {
        "id": "library://geniesim/background/kitchen/kitchen_1/background",
        "mode": "static_background",
        "collision": False,
        "visible_meshes": 22,
    }
    assert snapshot["object_positions"] == []
    assert "scene" not in state_snapshot({**raw, "state": "stopped"})


@pytest.mark.parametrize("state,age", [("ready", 5), ("starting", 0), ("failed", 0), ("stopped", 0)])
def test_old_or_unavailable_measurements_are_not_presented_as_current(state, age):
    snapshot = state_snapshot({**ready(), "state": state, "measured_at": time.monotonic() - age})
    assert not snapshot["fresh"]
    assert "object_positions" not in snapshot and "camera" not in snapshot


async def test_operator_snapshot_is_current_but_model_only_receives_visual_history():
    raw = ready()
    invoke = AsyncMock(return_value=SimpleNamespace(content="收到", tool_calls=[]))
    chat = GenesisChat(lambda: raw, model_factory=lambda _: SimpleNamespace(invoke=invoke))
    spec = ModelSpec()
    await chat.send("第一问", spec)
    raw["steps"] = 24
    await chat.send("第二问", spec)
    messages = invoke.call_args.args[0]
    assert len(messages) == 4
    assert "steps" not in json.dumps(messages)
    assert "object_positions" not in json.dumps(messages)
    assert chat.last_snapshot["steps"] == 24
    assert "第一问" == messages[1]["content"]
    assert "tools" not in invoke.call_args.kwargs
    await chat.send("新模型", replace(spec, model_name="other"))
    assert len(invoke.call_args.args[0]) == 2
    assert "第一问" not in json.dumps(invoke.call_args.args[0], ensure_ascii=False)


async def test_failure_is_safe_and_not_committed_to_history(caplog):
    invoke = AsyncMock(side_effect=RuntimeError("provider echoed private-key"))
    chat = GenesisChat(ready, model_factory=lambda _: SimpleNamespace(invoke=invoke))
    with pytest.raises(ChatError) as exc:
        await chat.send("失败的消息", ModelSpec(api_key="private-key"))
    assert "private-key" not in str(exc.value) + caplog.text
    assert not chat.busy
    invoke.side_effect = None
    invoke.return_value = SimpleNamespace(content="恢复", tool_calls=[])
    await chat.send("重试", ModelSpec(api_key="private-key"))
    assert len(invoke.call_args.args[0]) == 2


async def test_cancel_excludes_concurrent_send_and_keeps_chat_reusable():
    entered = asyncio.Event()

    async def slow(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    invoke = AsyncMock(side_effect=slow)
    chat = GenesisChat(ready, model_factory=lambda _: SimpleNamespace(invoke=invoke))
    task = asyncio.create_task(chat.send("慢请求", ModelSpec()))
    await entered.wait()
    with pytest.raises(ChatError, match="正在生成"):
        await chat.send("重复请求", ModelSpec())
    chat.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not chat.busy
    invoke.side_effect = None
    invoke.return_value = SimpleNamespace(content="新回复", tool_calls=[])
    await chat.send("新请求", ModelSpec())
    assert len(invoke.call_args.args[0]) == 2


async def test_reset_rejects_late_reply_even_if_transport_swallows_cancellation():
    entered = asyncio.Event()

    async def slow(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return SimpleNamespace(content="迟到回复", tool_calls=[])

    chat = GenesisChat(ready, model_factory=lambda _: SimpleNamespace(invoke=slow))
    task = asyncio.create_task(chat.send("旧消息", ModelSpec()))
    await entered.wait()
    chat.reset()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not chat.busy and chat.last_snapshot is None


async def test_timeout_and_tool_request_fail_without_actions():
    async def slow(*args, **kwargs):
        await asyncio.Event().wait()

    chat = GenesisChat(ready, model_factory=lambda _: SimpleNamespace(invoke=slow), timeout=0.01)
    with pytest.raises(ChatError, match="超时"):
        await chat.send("问题", ModelSpec())
    assert not chat.busy
    invoke = AsyncMock(return_value=SimpleNamespace(content="", tool_calls=[{"name": "move"}]))
    chat = GenesisChat(ready, model_factory=lambda _: SimpleNamespace(invoke=invoke))
    with pytest.raises(ChatError, match="未开放"):
        await chat.send("执行动作", ModelSpec())


async def test_history_is_bounded_and_reply_redacts_key():
    invoke = AsyncMock(return_value=SimpleNamespace(content="private-key " + "字" * 16000, tool_calls=[]))
    chat = GenesisChat(ready, model_factory=lambda _: SimpleNamespace(invoke=invoke))
    for _ in range(10):
        reply = await chat.send("问题", ModelSpec(api_key="private-key"))
    assert len(reply) == 16000 and "private-key" not in reply
    messages = invoke.call_args.args[0]
    assert len(messages) <= 14
    assert sum(len(m["content"]) for m in messages[1:-1]) <= 48000


@pytest.mark.parametrize("changes", [{"api_base": "file:///tmp/private"}, {"model_name": ""}, {"api_key": ""}])
def test_incomplete_model_cannot_start(changes):
    with pytest.raises(ChatError):
        GenesisChat(ready).configure(replace(ModelSpec(), **changes))


@pytest.mark.parametrize("with_image", [False, True])
async def test_real_sdk_serializes_state_to_local_openai_compatible_endpoint(with_image, monkeypatch, tmp_path):
    """Exercise build_model and SDK parsing against HTTP, without a paid model/key."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            requests.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            body = json.dumps(
                {
                    "id": "chat-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "状态读取链路正常"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                },
                ensure_ascii=False,
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            from jiuwensymbiosis.simulators.genesis import vision

            monkeypatch.setattr(vision, "OUTPUT_DIR", tmp_path)
            preview = SimpleNamespace(capture_image=lambda: {"jpeg": b"exact-image-bytes"})
            chat = GenesisChat(ready, timeout=10, preview=preview if with_image else None)
            spec = ModelSpec(api_base=f"http://127.0.0.1:{server.server_port}/v1", model_name="test-model")
            if with_image:
                assert (await chat.recognize(spec))["reply"] == "状态读取链路正常"
            else:
                assert await chat.send("当前坐标？", spec) == "状态读取链路正常"
            path, request = requests[0]
            assert path == "/v1/chat/completions"
            assert request["model"] == "test-model"
            if with_image:
                import base64

                parts = request["messages"][-1]["content"]
                assert parts[1]["type"] == "image_url"
                assert base64.b64decode(parts[1]["image_url"]["url"].split(",", 1)[1]) == b"exact-image-bytes"
                assert "object_positions" not in json.dumps(request)
            else:
                assert "object_positions" not in json.dumps(request)
            assert not request.get("tools")
        finally:
            server.shutdown()
            thread.join(timeout=2)


def test_indoor_contact_evidence_is_fresh_and_does_not_enable_readonly_actions():
    raw = {
        **ready(),
        "base_observation": {"position_m": [0, 0, 0.24]},
        "scene": {
            "mode": "robot_mobile",
            "background_id": "library://rooms/home0",
            "collision": True,
            "spawn_support_verified": True,
            "source_path": "/private/room.usda",
        },
        "scene_contacts": {"count": 1, "surfaces": [{"path": "/World/floor", "force_n": 100}]},
    }
    snapshot = state_snapshot(raw)
    assert snapshot["scene"]["spawn_support_verified"]
    assert snapshot["scene_contacts"]["count"] == 1
    assert not snapshot["robot_actions_available"]
    assert "source_path" not in snapshot["scene"]
    assert "scene_contacts" not in state_snapshot({**raw, "measured_at": time.monotonic() - 5})


async def test_legacy_state_grounded_history_is_not_replayed():
    invoke = AsyncMock(return_value=SimpleNamespace(content="收到", tool_calls=[]))
    chat = GenesisChat(ready, model_factory=lambda _: SimpleNamespace(invoke=invoke))
    spec = ModelSpec()
    chat.configure(spec)
    chat._history = [{"role": "assistant", "content": "LEGACY_WORLD_POSE"}]
    await chat.send("看前方", spec)
    assert "LEGACY_WORLD_POSE" not in json.dumps(invoke.call_args.args[0])
