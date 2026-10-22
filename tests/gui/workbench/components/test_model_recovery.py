"""Invalid prefill and credential restoration must not remove recovery controls."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from nicegui import context, ui

from jiuwensymbiosis.agent.config import ModelSpec
from jiuwensymbiosis_gui.workbench import layout as layout_module
from jiuwensymbiosis_gui.workbench import model_preferences, registry
from jiuwensymbiosis_gui.workbench.app_state import AppState
from jiuwensymbiosis_gui.workbench.pages import model_view
from tests.gui.workbench.unit.test_model_preferences import MemoryKeyring

_VALID_YAML = "env:\n  cfg:\n    low_level:\n      port: /dev/test-only\n"
_NICEGUI_IO_BOUND = model_view.run.io_bound


@pytest.fixture
def recovery_page(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "configs_dir", lambda: tmp_path / "configs")
    monkeypatch.setenv("JIUWENSYMBIOSIS_RUNTIME_DIR", str(tmp_path / "locks"))
    monkeypatch.setattr(ui, "timer", Mock())
    monkeypatch.setattr(ui, "notify", Mock())
    store = model_preferences.ModelPreferences(backend=MemoryKeyring())
    monkeypatch.setattr(model_view, "ModelPreferences", lambda: store)

    async def immediate(function, *args):
        return function(*args)

    monkeypatch.setattr(model_view.run, "io_bound", immediate)
    slot = context.client.content

    def build(yaml_text=_VALID_YAML):
        body = registry.list_bodies()[0]
        path = body.config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml_text, encoding="utf-8")
        state = AppState(workspace=str(tmp_path / "workspace"))
        with slot:
            page = layout_module.Layout(state)
        return page, path

    return build


@pytest.mark.parametrize("invalid", ["agent: [invalid]\n", "model:\n  api_base: null\n"])
async def test_bad_prefill_keeps_drop_recovery_and_strict_start_validation(recovery_page, monkeypatch, invalid):
    original = _VALID_YAML + invalid
    page, source = recovery_page(original)
    await page._models._restore_model()
    assert page._models.ready
    assert "任务配置" in page._models._config_status.text
    assert page._models._active is None
    assert source.read_text(encoding="utf-8") == original
    assert any(
        "drop" in listener.type
        for element in page._tabs.client.elements.values()
        for listener in element._event_listeners.values()
    )
    engine = Mock()
    monkeypatch.setattr(layout_module, "RunEngine", engine)
    page._start_run(page._state.current_task)
    engine.assert_not_called()
    assert ui.notify.call_args.kwargs["type"] == "negative"

    finished = SimpleNamespace(
        body_key=page._state.current_body,
        task_key=page._state.current_task,
        is_running=lambda: False,
        rerun_with=Mock(),
    )
    page._state.engine = finished
    page._rerun()
    finished.rerun_with.assert_not_called()
    assert ui.notify.call_args.kwargs["type"] == "negative"
    page._state.engine = None

    page._on_yaml_dropped(SimpleNamespace(args=[{"name": "fixed.yaml", "text": _VALID_YAML}]))
    assert page._drop_dialog.value
    page._confirm_drop()
    page._models._import_model()
    assert page._models._config_status.text == ""
    assert page._state.current_config().get("env.cfg.low_level.port") == "/dev/test-only"
    monkeypatch.setattr(page._tools, "release_hardware", lambda: True)
    monkeypatch.setattr(page._run, "attach", Mock())
    page._start_run(page._state.current_task)
    engine.assert_called_once()
    assert source.read_text(encoding="utf-8") == original


async def test_failed_manual_prefill_keeps_draft_and_active_model(recovery_page):
    page, _ = recovery_page()
    panel = page._models
    await panel._restore_model()
    panel._apply_model()
    previous = panel.selected()
    draft = panel._draft_model()
    page._state.current_config().data["model"] = {"api_base": None}
    panel._import_model()
    assert panel._draft_model() == draft
    assert panel.selected() == previous
    assert "任务配置" in panel._config_status.text


async def test_kwallet_auto_restore_skips_read_but_user_can_restore(recovery_page, monkeypatch):
    import keyring

    saved = replace(ModelSpec(), model_name="saved-wallet-model")
    memory = MemoryKeyring()
    model_preferences.ModelPreferences(backend=memory).save(saved)
    read = Mock(side_effect=memory.get_password)
    backend = type("Keyring", (), {"__module__": "keyring.backends.kwallet", "priority": 5, "get_password": read})()
    monkeypatch.setattr(keyring, "get_keyring", lambda: backend)
    page, _ = recovery_page()
    panel = page._models
    panel._preferences = model_preferences.ModelPreferences()
    await panel._restore_model()
    read.assert_not_called()
    assert panel.ready and panel._task_model_button.enabled and panel._apply_model_button.enabled
    panel._apply_model()
    assert panel._active is not None
    panel._use_task_model()
    assert panel._active is None
    assert panel._restore_model_button.enabled
    engine = Mock()
    monkeypatch.setattr(layout_module, "RunEngine", engine)
    monkeypatch.setattr(page._tools, "release_hardware", lambda: True)
    monkeypatch.setattr(page._run, "attach", Mock())
    page._start_run(page._state.current_task)
    engine.assert_called_once()
    finished = SimpleNamespace(
        body_key=page._state.current_body,
        task_key=page._state.current_task,
        is_running=lambda: False,
        rerun_with=Mock(),
    )
    page._state.engine = finished
    page._rerun()
    finished.rerun_with.assert_called_once()
    await panel._restore_model_interactively()
    read.assert_called_once()
    assert panel.selected().model_name == "saved-wallet-model"


async def test_timed_out_restore_cannot_overwrite_a_new_selection(recovery_page, monkeypatch):
    page, _ = recovery_page()
    panel = page._models
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def delayed_load(default):
        started.set()
        try:
            assert release.wait(5), "test failed to release credential reader"
            return replace(default, model_name="late-saved-model"), {}
        finally:
            finished.set()

    monkeypatch.setattr(panel._preferences, "load_profiles", delayed_load)
    monkeypatch.setattr(model_view.run, "io_bound", _NICEGUI_IO_BOUND)
    monkeypatch.setattr(model_view, "_AUTO_RESTORE_TIMEOUT_S", 0.05)
    try:
        await panel._restore_model()
        assert started.is_set()
        assert panel.ready and panel._apply_model_button.enabled
        assert "超时" in panel._memory_status.text
        panel._name.set_options(["user-model"], value="user-model")
        panel._apply_model()
        assert panel.selected().model_name == "user-model"
        release.set()
        assert await asyncio.to_thread(finished.wait, 1)
        await asyncio.sleep(0)
        assert panel.selected().model_name == "user-model"
        assert "未保存" in panel._memory_status.text
    finally:
        release.set()
