# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""config_view:原始 YAML 的「下载」与「存储为可选配置」。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jiuwensymbiosis_gui.workbench import registry
from jiuwensymbiosis_gui.workbench.config_model import ConfigModel
from jiuwensymbiosis_gui.workbench.pages import config_view as config_view_module
from jiuwensymbiosis_gui.workbench.pages.config_view import ConfigView

_YAML = "env:\n  cfg:\n    low_level:\n      port: /dev/ttyACM7\n"


@pytest.fixture
def page(tmp_path, monkeypatch):
    """配置目录引到临时目录后建一个已载入的配置页;``saved`` 记录主页刷新回调。"""
    monkeypatch.setattr(registry, "configs_dir", lambda: tmp_path)
    body_key = registry.list_bodies()[0].key
    registry.get_body(body_key).config_path().parent.mkdir(parents=True, exist_ok=True)
    saved: list[bool] = []
    view = ConfigView(on_run=lambda: None, on_back=lambda: None, on_config_saved=lambda: saved.append(True))
    view.load("拾取盒子", ConfigModel.from_yaml_text(_YAML), body_key=body_key)
    return SimpleNamespace(view=view, body_key=body_key, saved=saved)


def test_reveal_field_switches_to_the_group_that_holds_it(page):
    """别的页面把「这项没填」做成可点跳转,落点得是那一栏所在的分组,不是表单第一页。"""
    page.view._form_tabs.set_value("基础")

    assert page.view.reveal_field("env.cfg.low_level.camera_serial")
    assert page.view._form_tabs.value == "机器人参数"


def test_reveal_field_reports_paths_the_form_does_not_expose(page):
    """只在「原始 YAML」里能改的字段没有输入框,调用方据此改口而不是静默跳错地方。"""
    assert not page.view.reveal_field("env.cfg.low_level.not_a_form_field")


def test_switching_detector_mode_rebuilds_only_relevant_fields(page):
    page.view._set("detector.mode", "remote")
    assert {path for path in page.view._controls if path.startswith("detector.endpoint.")} == {
        "detector.endpoint.url",
        "detector.endpoint.request_timeout_s",
    }
    assert not any("model_id" in path for path in page.view._controls)
    page.view._set("detector.mode", "local")
    assert "detector.local.gdino_model_id" in page.view._controls
    assert "detector.local.startup_timeout_s" in page.view._controls
    assert "detector.endpoint.url" not in page.view._controls


def test_save_dialog_prefills_local_name_and_warns_on_overwrite(page):
    page.view._open_save()
    assert page.view._save_name.value == f"{page.body_key}.local"
    assert page.view._save_hint.text == ""  # 尚不存在 → 无覆盖提示

    registry.save_body_config(page.body_key, page.view._save_name.value, _YAML)
    page.view._refresh_overwrite_hint()
    assert "覆盖" in page.view._save_hint.text


def test_save_as_option_writes_into_body_config_dir(page):
    page.view._open_save()
    page.view._save_name.set_value("bench")
    page.view._yaml.value = _YAML
    page.view._save_as_option()

    assert registry.body_config_path(page.body_key, "bench").read_text(encoding="utf-8") == _YAML
    assert page.saved == [True]  # 通知主页刷新「配置文件」下拉


def test_save_skips_invalid_yaml(page):
    page.view._open_save()
    page.view._save_name.set_value("broken")
    page.view._yaml.value = "a: ["
    page.view._save_as_option()

    assert not registry.body_config_path(page.body_key, "broken").exists()
    assert page.saved == []


def test_download_sends_editor_text_and_skips_invalid_yaml(page, monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(
        config_view_module.ui, "download", SimpleNamespace(content=lambda *args, **kwargs: calls.append(args))
    )

    page.view._yaml.value = _YAML
    page.view._download_yaml()
    assert calls == [(_YAML, f"{page.body_key}.yaml")]

    page.view._yaml.value = "a: ["
    page.view._download_yaml()
    assert len(calls) == 1  # 非法 YAML 不下载
