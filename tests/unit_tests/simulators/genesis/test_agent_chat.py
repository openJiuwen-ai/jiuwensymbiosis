"""Action-mode chat delegates to the existing Agent, preserving cancellation ownership."""

import asyncio
import errno
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jiuwensymbiosis.agent.config import ModelSpec
from jiuwensymbiosis.simulators.common.chat import AgentIdleTimeout, ChatError, SimulationChat
from tests.unit_tests.simulators.genesis.test_chat import ready


def mobile_ready():
    return {**ready(), "base_observation": {"position_m": [0, 0, 0]}}


async def test_action_mode_uses_shared_runner_and_selected_model(monkeypatch):
    runner = Mock(return_value={"reply": "已停稳", "actions": [{"ok": True, "settled": True}]})
    monkeypatch.setattr("jiuwensymbiosis.simulators.common.agent.run_agent_query", runner)
    preview = object()
    direct_model = Mock(side_effect=AssertionError("must use shared agent"))
    chat = SimulationChat(mobile_ready, preview=preview, model_factory=direct_model)
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

    monkeypatch.setattr("jiuwensymbiosis.simulators.common.agent.run_agent_query", run)
    chat = SimulationChat(mobile_ready, preview=object())
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
    monkeypatch.setattr("jiuwensymbiosis.simulators.common.agent.run_agent_query", runner)
    chat = SimulationChat(ready, preview=object())
    chat.actions_enabled = True
    with pytest.raises(ChatError, match="MOZ1 底盘物理"):
        await chat.send("前进", ModelSpec())
    runner.assert_not_called()


async def test_workbench_reports_local_file_limit_when_the_resident_transport_fails(monkeypatch):
    def run(*args, **kwargs):
        raise RuntimeError("仿真连接中断") from OSError(errno.EMFILE, "private path")

    monkeypatch.setattr("jiuwensymbiosis.simulators.common.agent.run_agent_query", run)
    chat = SimulationChat(mobile_ready, preview=object())
    chat.actions_enabled = True
    with pytest.raises(ChatError, match="工作台打开文件数已达上限") as caught:
        await chat.send("前进", ModelSpec())
    assert "private path" not in str(caught.value)
    assert not chat.busy


def test_control_cannot_be_enabled_without_preview():
    chat = SimulationChat(ready)
    with pytest.raises(ChatError, match="控制接口"):
        chat.actions_enabled = True
    assert not chat.actions_enabled


async def test_progress_allows_an_agent_turn_to_run_past_the_idle_window(monkeypatch):
    def run(*args, on_progress, cancelled, **kwargs):
        for tool in ("capture_camera", None, "navigate_relative", None):
            on_progress(tool)
            time.sleep(0.07)
            assert not cancelled.is_set()
        return {"reply": "已完成"}

    monkeypatch.setattr("jiuwensymbiosis.simulators.common.agent.run_agent_query", run)
    chat = SimulationChat(mobile_ready, preview=object(), timeout=0.2)
    assert await chat.send("继续调整位置", ModelSpec()) == "已完成"


async def test_watchdog_allows_a_long_tool_then_expires_on_a_stalled_model(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def run(*args, on_progress, cancelled, **kwargs):
        on_progress("dance_crab")
        time.sleep(0.25)
        assert not cancelled.is_set()
        on_progress(None)
        entered.set()
        assert cancelled.wait(5)
        assert release.wait(5)
        return {"reply": "已取消", "control_released": True}

    monkeypatch.setattr("jiuwensymbiosis.simulators.common.agent.run_agent_query", run)
    chat = SimulationChat(mobile_ready, preview=object(), timeout=0.18)
    try:
        with pytest.raises(AgentIdleTimeout, match="模型或 Agent"):
            await chat.send("跳舞", ModelSpec())
        assert entered.is_set() and chat.busy
        with pytest.raises(ChatError, match="正在"):
            await chat.send("重复", ModelSpec())
    finally:
        release.set()
        assert (await chat.wait_for_action_exit())["control_released"]
    assert not chat.busy


async def test_stopping_simulation_model_cancels_an_inflight_request(monkeypatch):
    from jiuwensymbiosis.simulators.common.agent import _simulation_model

    entered, cleaned_up = asyncio.Event(), asyncio.Event()
    cancelled = threading.Event()

    async def invoke(**kwargs):
        try:
            entered.set()
            await asyncio.Event().wait()
        finally:
            cleaned_up.set()

    monkeypatch.setattr(
        "jiuwensymbiosis.simulators.common.agent.build_model", lambda *args, **kwargs: SimpleNamespace(invoke=invoke)
    )
    model = _simulation_model(ModelSpec(), cancelled)
    request = asyncio.create_task(model.invoke())
    await asyncio.wait_for(entered.wait(), 1)
    cancelled.set()
    assert (await asyncio.wait_for(request, 1)).content == ""
    assert cleaned_up.is_set()
