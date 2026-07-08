# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""registry:本体/任务注册项完整、路径可解析、模拟会话可构造。"""

from __future__ import annotations

from jiuwensymbiosis.agent.session import RobotSession
from jiuwensymbiosis.gui import registry


def test_piper_body_registered():
    body = registry.get_body("piper")
    assert body.key == "piper"
    assert body.capability_badges  # 非空徽章


def test_pick_box_task_registered_and_bound_to_piper():
    task = registry.get_task("pick_box")
    assert task.body_key == "piper"
    assert task.mock_script  # 有脚本序列
    assert task.default_query


def test_tasks_for_body_filters():
    tasks = registry.tasks_for_body("piper")
    assert any(t.key == "pick_box" for t in tasks)


def test_task_config_path_points_into_configs_dir():
    task = registry.get_task("pick_box")
    path = task.config_path()
    assert path.name == "piper.yaml"
    assert "piper" in str(path)


def test_body_builds_mock_session():
    body = registry.get_body("piper")
    session = body.build_mock_session()
    assert isinstance(session, RobotSession)
    assert "vision.camera" in session.env.capabilities


def test_load_tasks_merges_user_tasks(tmp_path, monkeypatch):
    user_dir = tmp_path / "gui"
    user_dir.mkdir()
    (user_dir / "tasks.yaml").write_text(
        "tasks:\n  - key: my_task\n    body: piper\n    display_name: 我的任务\n    config_relpath: piper/piper.yaml\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(registry, "user_data_dir", lambda: user_dir)
    tasks = registry._load_tasks()
    assert "pick_box" in tasks  # 内置仍在
    assert tasks["my_task"].display_name == "我的任务"  # 用户任务合并进来


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
    task = registry.add_user_task(
        display_name="我的新任务",
        description="演示",
        body_key="piper",
        config_yaml="env:\n  cfg:\n    prompt: hi\n",
    )
    try:
        assert registry.get_task(task.key).display_name == "我的新任务"  # 内存注册
        assert task.config_path().is_file()  # 配置 yaml 落盘(绝对路径)
        saved = yaml.safe_load((udir / "tasks.yaml").read_text(encoding="utf-8"))
        assert any(t["display_name"] == "我的新任务" for t in saved["tasks"])  # 追加到用户 tasks.yaml
    finally:
        registry._TASKS.pop(task.key, None)  # 避免污染其它用例的全局注册表
