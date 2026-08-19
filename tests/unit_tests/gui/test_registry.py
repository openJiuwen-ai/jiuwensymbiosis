# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""registry:本体/任务注册项完整、路径可解析、真机会话可构造。"""

from __future__ import annotations

from jiuwensymbiosis.agent.session import RobotSession
from jiuwensymbiosis.gui import registry


def test_piper_body_registered():
    body = registry.get_body("piper")
    assert body.key == "piper"
    assert body.display_name


def test_so101_body_registered_with_config():
    body = registry.get_body("so101")
    assert body.key == "so101"
    path = body.config_path()
    assert path.name == "so101.yaml"
    assert path.is_file()  # 随包配置存在


def test_pick_box_task_is_body_agnostic():
    task = registry.get_task("pick_box")
    assert task.bodies == ()  # 本体无关:适用所有本体
    assert task.default_query


def test_pick_banana_task_is_body_agnostic_with_fast_defaults():
    task = registry.get_task("pick_banana")
    assert task.bodies == ()  # 一张卡片,任意本体都能执行
    assert task.default_query
    assert task.agent_defaults.get("exec_mode") == "fastagent"
    assert task.agent_defaults.get("enable_skill") is True


def test_body_agnostic_tasks_appear_for_every_body():
    for body_key in ("piper", "so101", "cruzr"):
        keys = {t.key for t in registry.tasks_for_body(body_key)}
        assert {"pick_box", "pick_banana"} <= keys


def test_body_config_path_points_into_configs_dir():
    path = registry.get_body("piper").config_path()
    assert path.name == "piper.yaml"
    assert "piper" in str(path)


def test_so101_body_builds_real_session_from_shipped_config():
    import yaml

    body = registry.get_body("so101")
    data = yaml.safe_load(body.config_path().read_text(encoding="utf-8"))
    session = body.build_real_session(data)  # 约定式惰性构建,不因缺 SDK 在 import 期失败
    assert isinstance(session, RobotSession)
    # 随包真实配置带相机 → 声明运动 + 夹爪 + 视觉能力。
    assert "grasp.parallel" in session.env.capabilities
    assert any(c.startswith("vision.") for c in session.env.capabilities)


def test_load_tasks_merges_user_tasks(tmp_path, monkeypatch):
    user_dir = tmp_path / "gui"
    user_dir.mkdir()
    (user_dir / "tasks.yaml").write_text(
        "tasks:\n  - key: my_task\n    display_name: 我的任务\n    default_query: 做点什么\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(registry, "user_data_dir", lambda: user_dir)
    tasks = registry._load_tasks()
    assert "pick_box" in tasks  # 内置仍在
    assert tasks["my_task"].display_name == "我的任务"  # 用户任务合并进来
    assert tasks["my_task"].bodies == ()  # 本体无关


def test_load_tasks_falls_back_when_all_sources_missing(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(registry, "_data_dir", lambda: empty)
    monkeypatch.setattr(registry, "user_data_dir", lambda: empty)
    assert "pick_box" in registry._load_tasks()  # 回落到内置最小默认


def test_body_from_unknown_adapter_is_skipped():
    assert registry._body_from_dict({"key": "x", "adapter": "does-not-exist"}) is None


def test_add_user_task_persists_and_registers(tmp_path, monkeypatch):
    import yaml

    udir = tmp_path / "gui"
    monkeypatch.setattr(registry, "user_data_dir", lambda: udir)
    task = registry.add_user_task(display_name="我的新任务", description="演示", default_query="抓起杯子")
    try:
        assert registry.get_task(task.key).display_name == "我的新任务"  # 内存注册
        assert task.bodies == ()  # 本体无关意图
        assert task.default_query == "抓起杯子"
        saved = yaml.safe_load((udir / "tasks.yaml").read_text(encoding="utf-8"))
        assert any(t["display_name"] == "我的新任务" for t in saved["tasks"])  # 追加到用户 tasks.yaml
    finally:
        registry._TASKS.pop(task.key, None)  # 避免污染其它用例的全局注册表
