# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""app_state:配置装载/默认填充/一键修复沉淀(纯逻辑,无 UI 框架)。"""

from __future__ import annotations

import os

from jiuwensymbiosis.gui.app_state import AppState
from jiuwensymbiosis.gui.config_model import ConfigModel


def test_config_for_task_applies_agent_defaults_and_tracing():
    state = AppState()
    model = state.config_for_task("pick_box")
    assert model.get("agent.enable_tracing") is True  # 默认开启轨迹记录


def test_config_for_task_prefills_prompt_from_default_query():
    from jiuwensymbiosis.gui import registry

    state = AppState()
    model = state.config_for_task("pick_box")
    # piper.yaml 任务无关化后不含 prompt → 用任务默认指令预填「配置 → 任务指令」框
    default_query = registry.get_task("pick_box").default_query
    assert default_query and model.get("env.cfg.prompt") == default_query


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


def _detector_config(use_sam2=True):
    return ConfigModel.from_dict(
        {"api_servers": [{"_target_": "x.grounding_dino_sam2_server.main", "use_sam2": use_sam2}]}
    )


def _make_gdino(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"x")


def test_prime_sets_env_for_found_model_and_reports_missing(tmp_path, monkeypatch):
    from jiuwensymbiosis.gui import local_models

    monkeypatch.setattr(local_models, "HF_HUB", tmp_path / "hf")
    monkeypatch.setattr(local_models, "MODELSCOPE", tmp_path / "ms")
    monkeypatch.delenv("GDINO_MODEL_ID", raising=False)
    monkeypatch.delenv("SAM2_MODEL_ID", raising=False)
    snap = tmp_path / "hf" / "models--IDEA-Research--grounding-dino-base" / "snapshots" / "abc"
    _make_gdino(snap)  # gdino present locally, sam2 absent

    state = AppState()
    state.set_config("pick_box", _detector_config(use_sam2=True))
    missing = state.prime_detector_models("pick_box")

    # prime 直接写 os.environ,monkeypatch 不追踪也就不会还原;立刻 pop 回收,避免泄漏污染后续测试。
    gdino_env = os.environ.pop("GDINO_MODEL_ID", None)
    assert gdino_env == str(snap)  # 本地目录喂给检测器,离线加载
    assert missing == ["SAM2"]


def test_prime_respects_user_env(monkeypatch):
    monkeypatch.setenv("GDINO_MODEL_ID", "/my/own/gdino")
    monkeypatch.setenv("SAM2_MODEL_ID", "/my/own/sam2")
    state = AppState()
    state.set_config("pick_box", _detector_config())
    assert state.prime_detector_models("pick_box") == []  # 已设则不干预
    assert os.environ["GDINO_MODEL_ID"] == "/my/own/gdino"


def test_prime_noop_without_detector():
    state = AppState()
    state.set_config("pick_box", ConfigModel.from_dict({"env": {"cfg": {"prompt": "hi"}}}))
    assert state.prime_detector_models("pick_box") == []
