"""The simulation page shares model selection and gates Agent control explicitly."""

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from nicegui import ui

from jiuwensymbiosis.agent.config import ModelSpec
from jiuwensymbiosis_gui.workbench.pages.genesis_view import GenesisView
from tests.helpers import genesis_ready_state as ready


def mobile_ready():
    return {
        **ready(),
        "robot": {"control_mode": "wheel_velocity_physics"},
        "base_observation": {"position_m": [0, 0, 0]},
    }


@pytest.fixture
def page_slot():
    return ui.column()


@pytest.fixture
async def panel(monkeypatch, page_slot):
    monkeypatch.setattr(ui, "run_javascript", Mock())
    with page_slot:
        view = GenesisView(status=ready, model=lambda: ModelSpec(), open_models=lambda: None)
    invoke = AsyncMock(return_value=SimpleNamespace(content="**状态已读取**", tool_calls=[]))
    view._chat._model_factory = lambda _: SimpleNamespace(invoke=invoke)
    return view, invoke


async def test_diagnostic_snapshot_stays_out_of_model_and_clear_resets_context(panel):
    view, invoke = panel
    view._input.set_value("现在状态如何？")
    await view._send()
    assert "回复完成" in view._chat_status.text
    assert json.loads(view._result.content)["object_positions"] == [[1, 2, 3]]
    assert view._send_button.enabled and not view._cancel_button.enabled
    assert len(invoke.call_args.args[0]) == 2
    assert "object_positions" not in json.dumps(invoke.call_args.args[0])
    assert "纯视觉" in view._chat_status.text or "未获取新图像" in view._chat_status.text
    view._clear()
    view._input.set_value("新的对话")
    await view._send()
    assert len(invoke.call_args.args[0]) == 2


async def test_action_switch_uses_selected_model_and_shared_execution(monkeypatch, page_slot):
    runner = Mock(return_value={"reply": "左移完成并停稳", "actions": [{"ok": True, "settled": True}]})
    monkeypatch.setattr("jiuwensymbiosis.simulators.genesis.agent.run_agent_query", runner)
    preview = object()
    model = ModelSpec(model_name="shared-model")
    with page_slot:
        view = GenesisView(status=mobile_ready, model=lambda: model, open_models=lambda: None, preview=preview)
    assert view._action_switch.enabled
    view._action_switch.set_value(True)
    view._input.set_value("左移 20 厘米")
    await view._send()
    assert runner.call_args.args == (preview, model, "左移 20 厘米")
    assert "动作与相机观测" in view._chat_status.text
    assert view._action_switch.enabled


async def test_switch_off_revokes_running_agent_and_returns_to_read_only(monkeypatch, page_slot):
    entered, release = threading.Event(), threading.Event()
    captured = {}

    def run(*args, cancelled, **kwargs):
        captured["cancelled"] = cancelled
        entered.set()
        assert release.wait(5)
        return {"reply": "已取消"}

    runner = Mock(side_effect=run)
    monkeypatch.setattr("jiuwensymbiosis.simulators.genesis.agent.run_agent_query", runner)
    with page_slot:
        view = GenesisView(status=mobile_ready, model=ModelSpec, open_models=lambda: None, preview=object())
    invoke = AsyncMock(return_value=SimpleNamespace(content="只读，未执行移动", tool_calls=[]))
    view._chat._model_factory = lambda _: SimpleNamespace(invoke=invoke)
    assert not view._action_switch.value
    view._action_switch.set_value(True)
    view._input.set_value("前进 20 厘米")
    task = asyncio.create_task(view._send())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert view._action_switch.enabled  # OFF remains available during execution.
        view._action_switch.set_value(False)
        assert captured["cancelled"].is_set()
        await task
        assert not view._input.enabled and not view._action_switch.enabled
        assert "取消" in view._chat_status.text
    finally:
        view._cancel()
        release.set()
        await task
        if view._chat._action_task:
            await view._chat._action_task
    view._refresh_badge()
    assert view._input.enabled and view._action_switch.enabled
    assert not view._chat.actions_enabled
    view._input.set_value("继续前进 20 厘米")
    await view._send()
    assert runner.call_count == 2
    assert runner.call_args.kwargs["actions_enabled"] is False
    invoke.assert_not_awaited()
    assert not view._chat.last_snapshot["robot_actions_available"]


def test_assistant_is_named_symbiosis_sim(panel):
    view, _ = panel
    assert any(isinstance(e, ui.label) and e.text == "Symbiosis-sim" for e in view._workspace.descendants())


def test_camera_images_use_exact_model_input_and_clear_releases_them(panel):
    view, _ = panel
    image = "data:image/jpeg;base64,ZXhhY3Q="
    for step in range(15):
        view._append_frame(
            {"camera_name": "head_cam", "captured_utc": "2026-10-19T01:02:03Z", "step": step, "image_url": image}
        )
    images = [e for e in view._messages.descendants() if isinstance(e, ui.image)]
    assert len(images) == 12
    assert all(e.source == image for e in images)
    assert "模型正在理解" in view._chat_status.text
    view._clear()
    assert not view._image_cards
    assert not any(isinstance(e, ui.image) for e in view._messages.descendants())


async def test_manual_queries_work_without_any_model(panel):
    view, invoke = panel
    view._model = Mock(side_effect=RuntimeError("not configured"))
    view._query_objects()
    result = json.loads(view._result.content)
    assert result["fresh"] and result["object_positions"] == [[1, 2, 3]]
    invoke.assert_not_awaited()
    view._input.set_value("消息")
    await view._send()
    assert "模型设置尚未就绪" in view._chat_status.text
    assert view._input.value == "消息"
    invoke.assert_not_awaited()


async def test_cancel_stops_only_reply_and_reenables_input(panel):
    view, invoke = panel
    entered = asyncio.Event()

    async def slow(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    invoke.side_effect = slow
    view._input.set_value("慢请求")
    task = asyncio.create_task(view._send())
    await entered.wait()
    assert not view._send_button.enabled and view._cancel_button.enabled
    await view._send()
    assert invoke.await_count == 1
    view._cancel()
    await task
    assert "已停止" in view._chat_status.text
    assert view._input.enabled and view._send_button.enabled


def test_stale_query_does_not_show_old_positions(panel):
    view, _ = panel
    view._status = lambda: {**ready(), "measured_at": 0}
    view._query_status()
    assert "object_positions" not in json.loads(view._result.content)
    view._refresh_badge()
    assert "已过期" in view._backend_badge.text


def test_model_html_and_unsafe_links_are_not_rendered_as_active_content(panel):
    view, _ = panel
    view._append("<script>alert(1)</script> [link](javascript:alert(1))\n\n| A | B |\n|---|---|\n| 1 | 2 |")
    html = [e.content for e in view._messages.descendants() if isinstance(e, ui.html)][-1]
    assert "<script>" not in html and 'href="javascript:' not in html
    assert "<table>" in html and "table-scroll" in html


def test_panel_toggles_preserve_iframe_and_unsent_message(panel):
    view, _ = panel
    frame = next(e for e in view._workspace.descendants() if e.tag == "iframe")
    view._input.set_value("尚未发送的指令")
    for _ in range(2):
        view._toggle_chat()
        view._toggle_focus()
    assert next(e for e in view._workspace.descendants() if e.tag == "iframe") is frame
    assert view._input.value == "尚未发送的指令"
    assert not view._chat_collapsed and not view._focused
    assert not view._chat.actions_enabled


def test_state_summary_does_not_retain_stale_collision_information(panel):
    view, _ = panel
    view._status = lambda: {**ready(), "scene": {"mode": "robot_mobile", "collision": True, "background_id": "home0"}}
    view._query_status()
    labels = [e.text for e in view._state_summary.descendants() if isinstance(e, ui.label)]
    assert "室内场景与机器人" in labels and "已启用" in labels
    view._status = lambda: {**ready(), "measured_at": 0, "scene": {"collision": True}}
    view._query_status()
    labels = [e.text for e in view._state_summary.descendants() if isinstance(e, ui.label)]
    assert "已启用" not in labels and "未知" in labels


def test_unreachable_status_is_unknown_and_does_not_reuse_last_positions(panel):
    view, _ = panel
    view._status = Mock(side_effect=OSError("service unavailable"))
    view._refresh_badge()
    assert view._backend_badge.text == "服务不可达 · 状态未知"
    view._query_status()
    result = json.loads(view._result.content)
    assert not result["fresh"] and "object_positions" not in result


@pytest.mark.parametrize("actions_enabled", [False, True])
@pytest.mark.parametrize("sample_age", [0, 5])
async def test_manual_stop_remains_available_without_agent_permission_or_fresh_sample(
    page_slot, actions_enabled, sample_age
):
    preview = SimpleNamespace(base_command=Mock(return_value={"queued_id": "cancel"}))
    status = {**mobile_ready(), "scene": {"instance_id": "room-a"}, "config_revision": 4}
    status["measured_at"] -= sample_age
    with page_slot:
        view = GenesisView(status=lambda: status, model=ModelSpec, open_models=lambda: None, preview=preview)
    view._action_switch.set_value(actions_enabled)
    view._refresh_badge()
    assert view._robot_stop.visible and view._robot_stop.enabled
    await view._stop_robot()
    preview.base_command.assert_called_once_with(vx=0, vy=0, wz=0, scene_instance_id="room-a", config_revision=4)
    assert "已请求停止" in view._chat_status.text and "等待" in view._chat_status.text


async def test_stop_failure_is_reported_without_claiming_physical_success(page_slot):
    preview = SimpleNamespace(base_command=Mock(side_effect=RuntimeError("场景已切换")))
    with page_slot:
        view = GenesisView(status=mobile_ready, model=ModelSpec, open_models=lambda: None, preview=preview)
    view._action_switch.set_value(True)
    await view._stop_robot()
    assert "停止请求未确认" in view._chat_status.text
    view._action_switch.set_value(False)
    await view._stop_robot()
    assert preview.base_command.call_count == 2
    assert "停止请求未确认" in view._chat_status.text
