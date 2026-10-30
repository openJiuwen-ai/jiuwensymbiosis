"""Action-mode chat delegates to the existing Agent, preserving cancellation ownership."""

import asyncio
import threading
from unittest.mock import Mock

import pytest

from jiuwensymbiosis.agent.config import ModelSpec
from jiuwensymbiosis.simulators.genesis.chat import ChatError, GenesisChat
from tests.unit_tests.simulators.genesis.test_chat import ready


def mobile_ready():
    return {**ready(), "base_observation": {"position_m": [0, 0, 0]}}


async def test_action_mode_uses_shared_runner_and_selected_model(monkeypatch):
    runner = Mock(return_value={"reply": "已停稳", "actions": [{"ok": True, "settled": True}]})
    monkeypatch.setattr("jiuwensymbiosis.simulators.genesis.agent.run_agent_query", runner)
    preview = object()
    direct_model = Mock(side_effect=AssertionError("must use shared agent"))
    chat = GenesisChat(mobile_ready, preview=preview, model_factory=direct_model)
    chat.actions_enabled = True
    model = ModelSpec(model_name="selected-model")
    assert await chat.send("左移 20 厘米", model) == "已停稳"
    assert runner.call_args.args == (preview, model, "左移 20 厘米")
    assert chat.last_execution["actions"][0]["settled"]
    direct_model.assert_not_called()


async def test_cancel_does_not_forget_running_thread_or_allow_second_turn(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    captured = {}

    def run(*args, cancelled, **kwargs):
        captured["cancelled"] = cancelled
        entered.set()
        assert release.wait(5)
        return {"reply": "已取消", "actions": []}

    monkeypatch.setattr("jiuwensymbiosis.simulators.genesis.agent.run_agent_query", run)
    chat = GenesisChat(mobile_ready, preview=object())
    chat.actions_enabled = True
    task = asyncio.create_task(chat.send("前进", ModelSpec()))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        chat.actions_enabled = False
        with pytest.raises(asyncio.CancelledError):
            await task
        assert captured["cancelled"].is_set() and chat.busy
        with pytest.raises(ChatError, match="等待"):
            chat.actions_enabled = True
        assert not chat.actions_enabled
        with pytest.raises(ChatError, match="正在"):
            chat.configure(ModelSpec())
    finally:
        release.set()
        if chat._action_task:
            await chat._action_task
    assert not chat.busy


async def test_inspection_mode_gives_actionable_error_without_calling_model(monkeypatch):
    runner = Mock()
    monkeypatch.setattr("jiuwensymbiosis.simulators.genesis.agent.run_agent_query", runner)
    chat = GenesisChat(ready, preview=object())
    chat.actions_enabled = True
    with pytest.raises(ChatError, match="MOZ1 底盘物理"):
        await chat.send("前进", ModelSpec())
    runner.assert_not_called()


def test_control_cannot_be_enabled_without_preview():
    chat = GenesisChat(ready)
    with pytest.raises(ChatError, match="控制接口"):
        chat.actions_enabled = True
    assert not chat.actions_enabled
