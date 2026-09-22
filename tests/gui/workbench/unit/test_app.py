# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""app:重开自检的端口让出等待(区分健康实例 vs 正在关停的旧实例)。"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

from jiuwensymbiosis_gui.plugin import LaunchOptions
from jiuwensymbiosis_gui.workbench import app


def test_instance_identity_requires_matching_live_response(tmp_path, monkeypatch):
    import urllib.request

    options = LaunchOptions(port=8770)
    monkeypatch.setattr(app, "_LAUNCH_OPTIONS", options)
    monkeypatch.setattr(app, "_marker_path", lambda port: tmp_path / "instance.json")
    record = {"token": "actual-server", "launch": options.identity()}
    app._marker_path(8770).write_text(json.dumps(record))
    response = dict(record)
    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *args: SimpleNamespace(open=lambda *a, **kw: io.BytesIO(json.dumps(response).encode())),
    )
    assert app._healthy_instance_marked(8770)
    response["token"] = "different-server"
    assert not app._healthy_instance_marked(8770)
    response["token"] = "actual-server"
    monkeypatch.setattr(app, "_LAUNCH_OPTIONS", LaunchOptions(gui="another"))
    assert not app._healthy_instance_marked(8770)


def test_wait_for_port_release_returns_true_once_freed(monkeypatch):
    states = iter([True, True, False])  # 旧实例关停中,几拍后让出端口
    monkeypatch.setattr(app, "_server_already_running", lambda host, port: next(states))
    assert app._wait_for_port_release("127.0.0.1", 8770, timeout=2.0) is True


def test_wait_for_port_release_times_out_while_still_held(monkeypatch):
    monkeypatch.setattr(app, "_server_already_running", lambda host, port: True)  # 健康实例一直占着
    assert app._wait_for_port_release("127.0.0.1", 8770, timeout=0.2) is False


def _occupied(monkeypatch, *, marked: bool, released: bool) -> None:
    monkeypatch.setattr(app, "_server_already_running", lambda host, port: True)
    monkeypatch.setattr(app, "_healthy_instance_marked", lambda port: marked)
    monkeypatch.setattr(app, "_wait_for_port_release", lambda host, port, timeout: released)


def test_port_takeover_starts_when_port_is_free(monkeypatch):
    monkeypatch.setattr(app, "_server_already_running", lambda host, port: False)
    assert app._port_takeover("127.0.0.1", 8770, show=True) == "start"


def test_port_takeover_points_browser_at_healthy_instance(monkeypatch):
    _occupied(monkeypatch, marked=True, released=False)
    assert app._port_takeover("127.0.0.1", 8770, show=True) == "open"


def test_port_takeover_waits_out_a_closing_instance(monkeypatch):
    _occupied(monkeypatch, marked=False, released=True)
    assert app._port_takeover("127.0.0.1", 8770, show=True) == "start"
    assert app._port_takeover("127.0.0.1", 8770, show=False) == "start"  # 重启的接替进程同样接手


def test_port_takeover_aborts_headless_when_port_never_freed(monkeypatch):
    _occupied(monkeypatch, marked=False, released=False)
    assert app._port_takeover("127.0.0.1", 8770, show=True) == "abort"  # 未证实身份，不指向任意监听者
    assert app._port_takeover("127.0.0.1", 8770, show=False) == "abort"  # 无浏览器可指,只能放弃


def test_replacement_startup_failure_shows_error_without_opening_browser(monkeypatch):
    from jiuwensymbiosis_gui import startup

    _occupied(monkeypatch, marked=False, released=False)
    messages = []
    monkeypatch.setattr(startup, "_show_error_dialog", lambda title, message: messages.append(message))
    monkeypatch.setattr(app, "_LAUNCH_OPTIONS", None)
    assert app.run(show=False, port=9876) == 1
    assert len(messages) == 1
    assert "9876" in messages[0] and "手动重开" in messages[0]


def test_instance_marker_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "_INSTANCE_MARKER", None)
    monkeypatch.setattr(app, "_marker_path", lambda port: tmp_path / f"gui-{port}.lock")

    assert app._healthy_instance_marked(8770) is False  # 未起服务器前无标记
    app._mark_instance_healthy(8770)
    assert app._marker_path(8770).exists()
    assert app._healthy_instance_marked(8770) is False  # 文件本身不是服务身份的证据
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
    assert calls and "jiuwensymbiosis_gui" in calls[0][0][0]
    assert calls[0][0][0][3:5] == ["--gui", "workbench"]
    assert "--no-browser" in calls[0][0][0]  # 接替进程不再开新标签页,原页面自行重连刷新
    assert calls[0][1].get("start_new_session") is True  # 分离式,不随本进程退出被带走
