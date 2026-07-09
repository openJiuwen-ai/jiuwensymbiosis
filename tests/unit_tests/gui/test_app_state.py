# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""app_state:配置装载/默认填充/一键修复沉淀(纯逻辑,无 UI 框架)。"""

from __future__ import annotations

from jiuwensymbiosis.gui.app_state import AppState
from jiuwensymbiosis.gui.config_model import ConfigModel


def test_config_for_task_applies_agent_defaults_and_tracing():
    state = AppState()
    model = state.config_for_task("pick_box")
    assert model.get("agent.enable_tracing") is True  # 默认开启轨迹记录


def test_config_for_task_is_cached():
    state = AppState()
    first = state.config_for_task("pick_box")
    assert state.config_for_task("pick_box") is first  # 同一实例(带缓存)


def test_apply_fix_patches_detector_server():
    state = AppState()
    state.current_task = "pick_box"
    state.set_config("pick_box", ConfigModel.from_dict({"api_servers": [{"_target_": "x.grounding_dino.Detector"}]}))
    state.apply_fix({"hf_endpoint": "https://hf-mirror.com"})
    servers = state.config_for_task("pick_box").data["api_servers"]
    assert servers[0]["hf_endpoint"] == "https://hf-mirror.com"


def test_apply_fix_noop_without_current_task():
    state = AppState()
    state.apply_fix({"hf_endpoint": "x"})  # 不应抛异常
    assert state.current_task is None


def test_not_busy_before_any_run():
    assert AppState().is_busy() is False
