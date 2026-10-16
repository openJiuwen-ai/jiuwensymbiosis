"""Discovery and presets update the same ModelSpec the runtime receives."""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from jiuwensymbiosis_gui.workbench.model_preferences import ModelPreferenceError, ModelPreferences
from jiuwensymbiosis_gui.workbench.pages import model_view as model_view
from tests.gui.workbench.unit.test_model_preferences import MemoryKeyring


@pytest.fixture
def page_slot():
    return model_view.ui.column()


@pytest.fixture
async def view(tmp_path, monkeypatch, page_slot):
    config = {"max_tokens": 8192, "temperature": 0.5, "api_key": "old-service-key"}
    monkeypatch.setattr(model_view.ui, "notify", Mock())
    store = ModelPreferences(backend=MemoryKeyring())
    monkeypatch.setattr(model_view, "ModelPreferences", lambda: store)

    async def call(function, *args):
        return function(*args)

    monkeypatch.setattr(model_view.run, "io_bound", call)
    with page_slot:
        panel = model_view.ModelView(
            model_config=lambda: dict(config),
        )
    await panel._restore_model()
    yield panel, config


@pytest.fixture
def immediate_io(monkeypatch):
    async def call(function, *args):
        return function(*args)

    monkeypatch.setattr(model_view.run, "io_bound", call)


def test_official_preset_sets_actual_agent_configuration_and_does_not_save_key(view):
    panel, config = view
    panel._preset.set_value("deepseek")
    spec = panel._draft_model()
    assert spec.api_base == "https://api.deepseek.com"
    assert spec.api_key == ""
    assert spec.provider == "OpenAI"
    assert spec.model_name == "deepseek-flash"
    assert spec.verify_ssl
    assert spec.max_tokens == 8192
    assert config["api_key"] == "old-service-key"


async def test_fetched_display_name_selects_api_id_and_keeps_credentials_local(view, immediate_io, monkeypatch):
    panel, _ = view
    panel._preset.set_value("deepseek")
    panel._key.set_value("deepseek-test-key")
    fetch = Mock(return_value={"deepseek-flash": "DeepSeek-V4.1-Flash", "other": "Other model"})
    monkeypatch.setattr(model_view, "fetch_model_options", fetch)
    await panel._fetch_models()
    panel._name.set_value("other")
    spec = panel._draft_model()
    assert spec.model_name == "other"
    assert spec.api_key == "deepseek-test-key"
    assert fetch.call_args.args[0].model_name == "deepseek-flash"
    assert "2 个模型" in panel._catalog_status.text
    assert panel._fetch.enabled and panel._save_model_button.enabled


async def test_failed_fetch_preserves_manual_model_and_recovers_controls(view, immediate_io, monkeypatch):
    panel, _ = view
    before = panel._draft_model()
    monkeypatch.setattr(model_view, "fetch_model_options", Mock(side_effect=ValueError("鉴权失败")))
    await panel._fetch_models()
    assert panel._draft_model() == before
    assert panel._catalog_status.text == "鉴权失败"
    assert panel._fetch.enabled and panel._name.enabled


async def test_changed_endpoint_invalidates_catalog_without_losing_manual_choice(view, immediate_io, monkeypatch):
    panel, _ = view
    monkeypatch.setattr(model_view, "fetch_model_options", Mock(return_value={"another-model": "Another model"}))
    await panel._fetch_models()
    panel._name.set_value("another-model")
    panel._base.set_value("https://custom.example/v1")
    assert panel._name.options == {"another-model": "another-model"}
    assert panel._preset.value == "custom"
    assert panel._draft_model().model_name == "another-model"
    panel._import_model()
    assert panel._draft_model().api_key == "old-service-key"
    assert panel._draft_model().max_tokens == 8192


async def test_discovery_prevents_concurrent_send_or_second_fetch(view, monkeypatch):
    panel, _ = view
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_io(*args):
        started.set()
        await release.wait()
        return {"model": "model"}

    monkeypatch.setattr(model_view.run, "io_bound", slow_io)
    task = asyncio.create_task(panel._fetch_models())
    await started.wait()
    await panel._fetch_models()
    assert not panel._save_model_button.enabled
    release.set()
    await task
    assert panel._save_model_button.enabled


async def test_explicit_save_restores_model_and_key_when_page_opens_again(view):
    panel, config = view
    panel._preset.set_value("deepseek")
    panel._key.set_value("remembered-test-key")
    await panel._save_model()
    panel._import_model()
    await panel._restore_model()
    assert panel._draft_model().model_name == "deepseek-flash"
    assert panel._draft_model().api_key == "remembered-test-key"
    assert panel._draft_model().verify_ssl
    assert config["api_key"] == "old-service-key"


async def test_discovery_does_not_apply_or_save_a_draft(view, monkeypatch):
    panel, _ = view
    panel._preset.set_value("deepseek")
    panel._key.set_value("catalog-test-key")
    monkeypatch.setattr(model_view, "fetch_model_options", Mock(return_value={"deepseek-flash": "Flash"}))
    await panel._fetch_models()
    remembered = panel._preferences.load(panel._model)
    assert remembered is None
    assert panel.selected().api_key == "old-service-key"


async def test_forget_deletes_cache_and_clears_current_key_and_conversation(view):
    panel, _ = view
    panel._preset.set_value("deepseek")
    panel._key.set_value("forget-test-key")
    await panel._save_model()
    await panel._forget_model()
    assert panel._preferences.load(panel._model) is None
    assert panel._draft_model().api_key == ""
    assert panel._model.api_key == ""
    assert "已忘记" in panel._memory_status.text


async def test_cache_failure_keeps_chat_usable_and_does_not_claim_saved(view, monkeypatch):
    panel, _ = view
    monkeypatch.setattr(panel._preferences, "save", Mock(side_effect=ModelPreferenceError("钥匙串已锁定")))
    await panel._save_model()
    assert "钥匙串已锁定" in panel._memory_status.text
    assert panel.selected().api_key == "old-service-key"
    assert panel._save_model_button.enabled and panel._key.enabled


def test_changing_api_origin_clears_key_but_version_path_changes_keep_it(view):
    panel, _ = view
    panel._preset.set_value("deepseek")
    panel._key.set_value("origin-bound-test-key")
    panel._base.set_value("https://api.deepseek.com/v1")
    assert panel._key.value == "origin-bound-test-key"
    panel._base.set_value("https://other.example/v1")
    assert panel._key.value == ""


async def test_restore_failure_leaves_default_model_editable(view, monkeypatch):
    panel, _ = view
    before = panel._draft_model()
    monkeypatch.setattr(panel._preferences, "load", Mock(side_effect=ModelPreferenceError("无法读取钥匙串")))
    await panel._restore_model()
    assert panel._draft_model() == before
    assert panel._key.enabled and panel._fetch.enabled
    assert panel._memory_status.text == "无法读取钥匙串"


def test_draft_is_not_applied_and_selected_snapshot_keeps_task_parameters(view):
    from jiuwensymbiosis.agent.config import ModelSpec

    panel, _ = view
    panel._name.set_value("unsaved-model")
    chosen = panel.selected(ModelSpec(max_tokens=777, temperature=0.25))
    assert chosen.model_name != "unsaved-model"
    assert chosen.max_tokens == 777 and chosen.temperature == 0.25
    chosen.api_key = "mutated-snapshot"
    assert panel.selected().api_key == "old-service-key"


async def test_invalid_draft_cannot_replace_applied_model(view):
    panel, _ = view
    before = panel.selected()
    panel._base.set_value("invalid-url")
    await panel._save_model()
    assert panel.selected() == before


def test_selection_is_blocked_during_restoration(view):
    panel, _ = view
    panel._restoring_model = True
    with pytest.raises(RuntimeError, match="正在恢复"):
        panel.selected()


def test_without_an_applied_profile_each_task_keeps_its_connection(view):
    from jiuwensymbiosis.agent.config import ModelSpec

    panel, _ = view
    task_model = ModelSpec(api_base="https://task.example/v1", api_key="task-key", model_name="task-model")
    assert panel.selected(task_model) == task_model


async def test_failed_save_keeps_previously_applied_connection(view, monkeypatch):
    panel, _ = view
    await panel._save_model()
    previous = panel.selected()
    panel._base.set_value("https://other.example/v1")
    panel._key.set_value("other-test-key")
    panel._name.set_value("other-model")
    monkeypatch.setattr(panel._preferences, "save", Mock(side_effect=ModelPreferenceError("保存失败")))
    await panel._save_model()
    assert panel.selected() == previous


def test_tls_choice_is_part_of_the_connection_profile(view):
    panel, _ = view
    panel._verify_ssl.set_value(True)
    assert panel._draft_model().verify_ssl is True
    panel._verify_ssl.set_value(False)
    panel._base.set_value("https://other.example/v1")
    assert panel._draft_model().verify_ssl is True


async def test_forget_does_not_apply_an_unsaved_endpoint(view):
    panel, _ = view
    await panel._save_model()
    previous = panel.selected()
    panel._base.set_value("https://unsaved.example/v1")
    await panel._forget_model()
    assert panel.selected().api_base == previous.api_base
    assert panel.selected().api_key == ""


def test_session_apply_works_without_keyring_and_does_not_modify_task(view, monkeypatch):
    panel, config = view
    original = dict(config)
    backend = Mock(side_effect=AssertionError("temporary selection must not access keyring"))
    monkeypatch.setattr(panel._preferences, "_backend", backend)
    panel._base.set_value("https://temporary.example/v1")
    panel._key.set_value("temporary-test-key")
    panel._name.set_options(["temporary-model"], value="temporary-model")
    panel._apply_model()
    selected = panel.selected()
    assert selected.api_base == "https://temporary.example/v1"
    assert selected.api_key == "temporary-test-key"
    assert selected.max_tokens == 8192 and selected.temperature == 0.5
    panel._key.set_value("later-draft-key")
    assert panel.selected().api_key == "temporary-test-key"
    assert config == original
    backend.assert_not_called()
    assert "未保存" in panel._memory_status.text


async def test_temporary_selection_and_task_mode_leave_saved_profile_for_next_page(view, page_slot):
    panel, config = view
    await panel._save_model()
    saved = panel.selected()
    panel._base.set_value("https://temporary.example/v1")
    panel._key.set_value("temporary-test-key")
    panel._apply_model()
    assert panel.selected().api_base != saved.api_base
    panel._use_task_model()
    assert panel._preferences.load(saved) == saved
    with page_slot:
        reopened = model_view.ModelView(model_config=lambda: dict(config))
    await reopened._restore_model()
    assert reopened.selected() == saved


@pytest.mark.parametrize("field,value", [("_base", "invalid-url"), ("_provider", " "), ("_name", None)])
def test_invalid_temporary_connection_keeps_active_selection(view, field, value):
    panel, _ = view
    panel._apply_model()
    previous = panel.selected()
    getattr(panel, field).set_value(value)
    panel._apply_model()
    assert panel.selected() == previous
    assert "当前生效模型未改变" in panel._memory_status.text


async def test_save_failure_can_be_followed_by_session_apply(view, monkeypatch):
    panel, _ = view
    panel._base.set_value("https://temporary.example/v1")
    monkeypatch.setattr(panel._preferences, "save", Mock(side_effect=ModelPreferenceError("钥匙串已锁定")))
    await panel._save_model()
    assert panel.selected().api_base != "https://temporary.example/v1"
    panel._apply_model()
    assert panel.selected().api_base == "https://temporary.example/v1"


def test_task_mode_uses_each_tasks_live_defaults_without_applying_draft(view, monkeypatch):
    from jiuwensymbiosis.agent.config import ModelSpec

    panel, config = view
    panel._apply_model()
    frozen = panel.selected()
    panel._base.set_value("https://unsaved.example/v1")
    backend = Mock(side_effect=AssertionError("task mode must not access keyring"))
    monkeypatch.setattr(panel._preferences, "_backend", backend)
    panel._use_task_model()
    for name in ("task-a", "task-b"):
        default = ModelSpec(api_base=f"https://{name}.example/v1", api_key=f"{name}-key", model_name=name)
        assert panel.selected(default) == default
    config["model_name"] = "changed-task-model"
    assert panel.selected().model_name == "changed-task-model"
    assert panel._draft_model().api_base == "https://unsaved.example/v1"
    assert frozen.api_key == "old-service-key"
    backend.assert_not_called()


@pytest.mark.parametrize("busy", ["_restoring_model", "_saving_model", "_fetching_models"])
def test_new_selection_actions_cannot_override_pending_operation(view, busy):
    panel, _ = view
    panel._apply_model()
    previous = panel.selected()
    panel._base.set_value("https://unsaved.example/v1")
    setattr(panel, busy, True)
    panel._apply_model()
    panel._use_task_model()
    setattr(panel, busy, False)
    assert panel.selected() == previous


async def test_forgetting_saved_profile_preserves_task_mode(view):
    panel, _ = view
    await panel._save_model()
    panel._use_task_model()
    before = panel.selected()
    await panel._forget_model()
    assert panel.selected() == before
    assert panel._preferences.load(before) is None
    assert panel._key.value == ""
