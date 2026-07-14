# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""app:重开自检的端口让出等待(区分健康实例 vs 正在关停的旧实例)。"""

from __future__ import annotations

from jiuwensymbiosis.gui import app


def test_wait_for_port_release_returns_true_once_freed(monkeypatch):
    states = iter([True, True, False])  # 旧实例关停中,几拍后让出端口
    monkeypatch.setattr(app, "_server_already_running", lambda host, port: next(states))
    assert app._wait_for_port_release("127.0.0.1", 8770, timeout=2.0) is True


def test_wait_for_port_release_times_out_while_still_held(monkeypatch):
    monkeypatch.setattr(app, "_server_already_running", lambda host, port: True)  # 健康实例一直占着
    assert app._wait_for_port_release("127.0.0.1", 8770, timeout=0.2) is False


def test_instance_marker_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "_INSTANCE_MARKER", None)
    monkeypatch.setattr(app, "_marker_path", lambda port: tmp_path / f"gui-{port}.lock")

    assert app._healthy_instance_marked(8770) is False  # 未起服务器前无标记
    app._mark_instance_healthy(8770)
    assert app._healthy_instance_marked(8770) is True  # 起服务器后落标记 → 新进程秒开
    app.clear_instance_marker()
    assert app._healthy_instance_marked(8770) is False  # 退出前撤标记 → 新进程改为等端口让出


def test_clear_instance_marker_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "_INSTANCE_MARKER", None)
    monkeypatch.setattr(app, "_marker_path", lambda port: tmp_path / f"gui-{port}.lock")
    app._mark_instance_healthy(8770)
    app.clear_instance_marker()
    app.clear_instance_marker()  # 关停时 + ui.run 收尾都会调,重复调用不报错
    assert app._healthy_instance_marked(8770) is False


def test_resolve_trace_returns_path_for_existing_safe_stem(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "gui-abc123.json").write_text("{}", encoding="utf-8")
    assert app._resolve_trace("gui-abc123", str(tmp_path)) == (traces / "gui-abc123.json").resolve()


def test_resolve_trace_rejects_separators_and_traversal(tmp_path):
    (tmp_path / "traces").mkdir()
    assert app._resolve_trace("../secret", str(tmp_path)) is None  # 含分隔符 → 拒
    assert app._resolve_trace("a/b", str(tmp_path)) is None
    assert app._resolve_trace("bad name", str(tmp_path)) is None  # 空格非安全字符


def test_resolve_trace_returns_none_for_missing_or_no_workspace(tmp_path):
    (tmp_path / "traces").mkdir()
    assert app._resolve_trace("nope", str(tmp_path)) is None  # 文件不存在
    assert app._resolve_trace("gui-abc", None) is None  # 无工作区


def test_spawn_replacement_clears_marker_and_launches_entry(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setattr(app, "_INSTANCE_MARKER", None)
    monkeypatch.setattr(app, "_marker_path", lambda port: tmp_path / f"gui-{port}.lock")
    app._mark_instance_healthy(8770)
    calls: list = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: calls.append((a, k)))

    app.spawn_replacement()

    assert app._healthy_instance_marked(8770) is False  # 先撤标记 → 接替进程会等端口让出后自己起
    assert calls and "jiuwensymbiosis.gui" in calls[0][0][0]  # 拉起同一入口
    assert calls[0][1].get("start_new_session") is True  # 分离式,不随本进程退出被带走
