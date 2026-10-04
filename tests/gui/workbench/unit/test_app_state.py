# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""app_state:配置装载/默认填充/一键修复沉淀(纯逻辑,无 UI 框架)。"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from jiuwensymbiosis_gui.workbench.app_state import AppState
from jiuwensymbiosis_gui.workbench.config_model import ConfigModel


def test_config_for_task_applies_agent_defaults_and_tracing():
    state = AppState()
    model = state.config_for("piper", "pick_box")
    assert model.get("agent.enable_tracing") is True  # 默认开启轨迹记录
    assert model.get("agent.exec_mode") == "fastagent"  # 默认快速模式


def test_config_for_task_prefills_prompt_from_default_query():
    from jiuwensymbiosis_gui.workbench import registry

    state = AppState()
    model = state.config_for("piper", "pick_box")
    # piper.yaml 任务无关化后不含 prompt → 用任务默认指令预填「配置 → 任务指令」框
    default_query = registry.get_task("pick_box").default_query
    assert default_query and model.get("env.cfg.prompt") == default_query


def test_config_for_task_is_cached():
    state = AppState()
    first = state.config_for("piper", "pick_box")
    assert state.config_for("piper", "pick_box") is first  # 同一实例(带缓存)


def test_config_for_uses_selected_config_file(tmp_path):
    alt = tmp_path / "so101.alt.yaml"
    alt.write_text("env:\n  cfg:\n    low_level:\n      port: /dev/ttyACM9\n", encoding="utf-8")

    state = AppState()
    default_model = state.config_for("so101", "pick_box")
    state.current_config_file = str(alt)
    alt_model = state.config_for("so101", "pick_box")

    assert alt_model is not default_model  # 换配置文件 = 另一份独立可编辑配置
    assert alt_model.get("env.cfg.low_level.port") == "/dev/ttyACM9"
    assert alt_model.get("agent.enable_tracing") is True  # 任务/GUI 默认照样叠上
    state.current_config_file = None
    assert state.config_for("so101", "pick_box") is default_model  # 切回默认取回原缓存


def test_apply_fix_patches_detector_server():
    state = AppState()
    state.current_body = "piper"
    state.current_task = "pick_box"
    detector_cfg = ConfigModel.from_dict({"api_servers": [{"_target_": "x.grounding_dino.Detector"}]})
    state.set_config("piper", "pick_box", detector_cfg)
    state.apply_fix({"hf_endpoint": "https://hf-mirror.com"})
    servers = state.config_for("piper", "pick_box").data["api_servers"]
    assert servers[0]["hf_endpoint"] == "https://hf-mirror.com"


def test_apply_fix_noop_without_current_task():
    state = AppState()
    state.apply_fix({"hf_endpoint": "x"})  # 不应抛异常
    assert state.current_task is None


def test_start_pose_survives_within_the_same_body_and_config():
    state = AppState()
    state.current_body = "piper"
    state.remember_start_pose("piper", [1.0, 2.0])

    assert state.start_pose_joints() == [1.0, 2.0]


def test_start_pose_is_dropped_after_switching_body():
    """关节数与限位都可能对不上,拿 A 本体的姿态喂 B 本体是往硬件里塞垃圾。"""
    state = AppState()
    state.current_body = "piper"
    state.remember_start_pose("piper", [1.0, 2.0])

    state.current_body = "so101"

    assert state.start_pose_joints() is None


def test_start_pose_is_dropped_after_switching_config_file():
    state = AppState()
    state.current_body = "piper"
    state.remember_start_pose("piper", [1.0, 2.0])

    state.current_config_file = "/tmp/other.yaml"

    assert state.start_pose_joints() is None


def test_no_start_pose_before_any_run():
    state = AppState()
    state.current_body = "piper"

    assert state.start_pose_joints() is None


def test_start_pose_keeps_original_job_config_when_client_selected_another(tmp_path):
    state = AppState()
    original = tmp_path / "original.yaml"
    state.current_body = "piper"
    state.current_config_file = str(tmp_path / "other.yaml")
    state.remember_start_pose("piper", [1.0], config_source=original)
    assert state.start_pose_joints() is None
    state.current_config_file = str(original)
    assert state.start_pose_joints() == [1.0]


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


def test_prime_writes_config_for_found_model_and_reports_missing(tmp_path, monkeypatch):
    from jiuwensymbiosis_gui.workbench import local_models

    monkeypatch.setattr(local_models, "HF_HUB", tmp_path / "hf")
    monkeypatch.setattr(local_models, "MODELSCOPE", tmp_path / "ms")
    monkeypatch.delenv("GDINO_MODEL_ID", raising=False)
    monkeypatch.delenv("SAM2_MODEL_ID", raising=False)
    snap = tmp_path / "hf" / "models--IDEA-Research--grounding-dino-base" / "snapshots" / "abc"
    _make_gdino(snap)  # gdino present locally, sam2 absent

    state = AppState()
    cfg = _detector_config(use_sam2=True)
    state.set_config("piper", "pick_box", cfg)
    missing = state.prime_detector_models("piper", "pick_box")

    # 本地目录写进检测器配置项(离线加载),不再污染进程环境变量。
    assert cfg.data["api_servers"][0]["gdino_model_id"] == str(snap)
    assert "GDINO_MODEL_ID" not in os.environ
    assert missing == ["SAM2"]


def test_prime_respects_user_env(monkeypatch):
    # 环境变量显式指定(CLI export 场景)时 prime 不干预,与 config.py 的 env>yaml 一致。
    monkeypatch.setenv("GDINO_MODEL_ID", "/my/own/gdino")
    monkeypatch.setenv("SAM2_MODEL_ID", "/my/own/sam2")
    state = AppState()
    state.set_config("piper", "pick_box", _detector_config())
    assert state.prime_detector_models("piper", "pick_box") == []  # 已设则不干预
    assert os.environ["GDINO_MODEL_ID"] == "/my/own/gdino"


def test_prime_canonical_freezes_env_models_into_actual_sidecar_arguments(monkeypatch):
    from jiuwensymbiosis.adapters._common.builder import make_detector_sidecar
    from jiuwensymbiosis.adapters.piper.config import PiperConfig
    from jiuwensymbiosis_gui.workbench import local_models

    monkeypatch.setenv("GDINO_MODEL_ID", "/my/own/gdino")
    monkeypatch.setenv("SAM2_MODEL_ID", "/my/own/sam2")

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit model paths need no cache search")

    monkeypatch.setattr(local_models, "detect_local_model", forbidden)
    cfg = ConfigModel.from_dict({"detector": {"mode": "disabled"}})
    cfg.set("detector.mode", "local")
    state = AppState()
    state.set_config("piper", "pick_box", cfg)
    assert state.prime_detector_models("piper", "pick_box") == []

    # The GUI export and the real sidecar factory must agree after env changes.
    restored = ConfigModel.from_yaml_text(cfg.to_yaml())
    monkeypatch.setenv("GDINO_MODEL_ID", "/later/gdino")
    monkeypatch.setenv("SAM2_MODEL_ID", "/later/sam2")
    assert state.prime_detector_models("piper", "pick_box") == []
    command = make_detector_sidecar()(PiperConfig.from_dict(restored.data))().command
    assert command[command.index("--gdino-model-id") + 1] == "/my/own/gdino"
    assert command[command.index("--sam2-model-id") + 1] == "/my/own/sam2"
    assert cfg.get("detector.local.gdino_model_id") == "/my/own/gdino"
    assert cfg.get("detector.local.sam2_model_id") == "/my/own/sam2"


@pytest.mark.parametrize("model_id", ["/configured/gdino", "IDEA-Research/grounding-dino-base"])
def test_prime_canonical_explicit_model_takes_precedence_over_env(model_id, monkeypatch):
    from jiuwensymbiosis.perception.config import parse_detector_config
    from jiuwensymbiosis_gui.workbench import local_models

    monkeypatch.setenv("GDINO_MODEL_ID", "/from/env/gdino")
    monkeypatch.setattr(local_models, "detect_local_model", lambda *args: None)
    cfg = ConfigModel.from_dict(
        {"detector": {"mode": "local", "local": {"gdino_model_id": model_id, "use_sam2": False}}}
    )
    state = AppState()
    state.set_config("piper", "pick_box", cfg)
    missing = state.prime_detector_models("piper", "pick_box")
    assert missing == (["GroundingDINO"] if model_id == local_models.GDINO_REPO else [])
    assert parse_detector_config(cfg.data).local.gdino_model_id == model_id


def test_prime_canonical_blank_env_does_not_claim_models_are_ready(monkeypatch):
    from jiuwensymbiosis_gui.workbench import local_models

    monkeypatch.setenv("GDINO_MODEL_ID", "  \t")
    monkeypatch.setenv("SAM2_MODEL_ID", "  \n")
    monkeypatch.setattr(local_models, "detect_local_model", lambda *args: None)
    cfg = ConfigModel.from_dict({"detector": {"mode": "local", "local": {}}})
    state = AppState()
    state.set_config("piper", "pick_box", cfg)
    assert state.prime_detector_models("piper", "pick_box") == ["GroundingDINO", "SAM2"]
    assert cfg.local_detector() == {}


def test_prime_respects_explicit_config_model_id(tmp_path, monkeypatch):
    from jiuwensymbiosis_gui.workbench import local_models

    # 配置里显式指定了非默认(本地)路径 → 即使本地缓存不存在也尊重:不覆盖、不报 missing。
    monkeypatch.setattr(local_models, "HF_HUB", tmp_path / "hf")
    monkeypatch.setattr(local_models, "MODELSCOPE", tmp_path / "ms")
    monkeypatch.delenv("GDINO_MODEL_ID", raising=False)
    monkeypatch.delenv("SAM2_MODEL_ID", raising=False)
    cfg = ConfigModel.from_dict(
        {
            "api_servers": [
                {
                    "_target_": "x.grounding_dino_sam2_server.main",
                    "gdino_model_id": "/my/local/gdino",
                    "sam2_model_id": "/my/local/sam2",
                }
            ]
        }
    )
    state = AppState()
    state.set_config("piper", "pick_box", cfg)
    assert state.prime_detector_models("piper", "pick_box") == []
    detector = cfg.data["api_servers"][0]
    assert detector["gdino_model_id"] == "/my/local/gdino"  # 未被自动探测覆盖
    assert detector["sam2_model_id"] == "/my/local/sam2"


def test_prime_treats_default_placeholder_as_not_explicit(tmp_path, monkeypatch):
    from jiuwensymbiosis_gui.workbench import local_models

    # 出厂占位符 == 默认 repo id,不算显式定制:仍应被探测到的本地快照替换。
    monkeypatch.setattr(local_models, "HF_HUB", tmp_path / "hf")
    monkeypatch.setattr(local_models, "MODELSCOPE", tmp_path / "ms")
    monkeypatch.delenv("GDINO_MODEL_ID", raising=False)
    monkeypatch.delenv("SAM2_MODEL_ID", raising=False)
    snap = tmp_path / "hf" / "models--IDEA-Research--grounding-dino-base" / "snapshots" / "abc"
    _make_gdino(snap)
    cfg = ConfigModel.from_dict(
        {
            "api_servers": [
                {
                    "_target_": "x.grounding_dino_sam2_server.main",
                    "gdino_model_id": local_models.GDINO_REPO,
                    "use_sam2": False,
                }
            ]
        }
    )
    state = AppState()
    state.set_config("piper", "pick_box", cfg)
    assert state.prime_detector_models("piper", "pick_box") == []
    assert cfg.data["api_servers"][0]["gdino_model_id"] == str(snap)


def test_prime_noop_without_detector():
    state = AppState()
    state.set_config("piper", "pick_box", ConfigModel.from_dict({"env": {"cfg": {"prompt": "hi"}}}))
    assert state.prime_detector_models("piper", "pick_box") == []


def test_prime_remote_never_searches_model_files(monkeypatch):
    from jiuwensymbiosis_gui.workbench import app_state, local_models

    state = AppState()
    state.set_config(
        "piper",
        "pick_box",
        ConfigModel.from_dict(
            {
                "detector": {"mode": "remote", "endpoint": {"url": "https://inference.invalid"}},
            }
        ),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("remote mode cannot probe local model files")

    monkeypatch.setattr(local_models, "detect_local_model", forbidden)
    monkeypatch.setattr(app_state, "os", SimpleNamespace(environ=SimpleNamespace(get=forbidden)))
    assert state.prime_detector_models("piper", "pick_box") == []


def test_prime_noop_when_vision_disabled():
    state = AppState()
    cfg = _detector_config()
    cfg.set("gui.disable_vision", True)  # 「禁用视觉服务」开关打开
    state.set_config("piper", "pick_box", cfg)
    assert state.prime_detector_models("piper", "pick_box") == []  # 不去喂检测器模型
