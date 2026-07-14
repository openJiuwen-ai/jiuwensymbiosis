# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""run_engine:后台线程 + 事件队列驱动一次模拟运行(纯逻辑,无 Qt / 无 nicegui)。"""

from __future__ import annotations

import logging
import queue

from jiuwensymbiosis.gui import registry
from jiuwensymbiosis.gui.run_engine import QueueLogHandler, RunEngine, default_workspace


def _tags(events):
    return [tag for tag, _ in events]


def test_mock_run_emits_ordered_event_stream(tmp_path):
    task = registry.get_task("pick_box")
    config = {
        "env": {"cfg": {"prompt": "把黑盒放到白盒上"}},
        "agent": {"mode": "tool", "max_iterations": 20, "enable_visual_feedback": False, "enable_tracing": False},
    }
    engine = RunEngine(task, config, mock=True, workspace=str(tmp_path))
    engine.start()
    engine.join(timeout=60)
    assert not engine.is_running()

    events = engine.drain()
    tags = _tags(events)
    assert tags[0] == "run_started"
    assert tags[-1] == "run_finished"

    meta = events[0][1]
    assert meta["body"] == "piper"
    assert meta["mock"] is True

    finished = [payload for tag, payload in events if tag == "step_finished"]
    tools = [f["tool"] for f in finished]
    assert tools == [step["tool"] for step in task.mock_script]  # 忠实回放该任务的脚本序列
    assert all(f["ok"] for f in finished)

    result = events[-1][1]
    assert result["ok"] is True


def test_mock_run_frames_are_encoded_data_uris(tmp_path):
    task = registry.get_task("pick_box")
    config = {"agent": {"mode": "tool", "max_iterations": 20, "enable_visual_feedback": False, "enable_tracing": False}}
    engine = RunEngine(task, config, mock=True, workspace=str(tmp_path))
    engine.start()
    engine.join(timeout=60)

    frames = [payload for tag, payload in engine.drain() if tag == "frame"]
    assert frames  # 初始帧 + 运动/抓取后各刷新
    assert all(isinstance(uri, str) and uri.startswith("data:image/jpeg;base64,") for uri in frames)


def test_clone_reuses_same_params_with_independent_config(tmp_path):
    task = registry.get_task("pick_box")
    config = {"env": {"cfg": {"prompt": "把黑盒放到白盒上"}}, "agent": {"mode": "tool"}}
    engine = RunEngine(task, config, mock=True, workspace=str(tmp_path))

    twin = engine.clone()

    assert twin is not engine
    assert twin._task is task and twin._mock is True and twin._workspace == str(tmp_path)
    assert twin._config.data == engine._config.data
    twin._config.set("env.cfg.prompt", "改了")  # 深拷贝:动克隆不影响原引擎
    assert engine._config.get("env.cfg.prompt") == "把黑盒放到白盒上"


def test_drain_is_empty_before_start(tmp_path):
    engine = RunEngine(registry.get_task("pick_box"), {}, mock=True, workspace=str(tmp_path))
    assert engine.drain() == []
    assert engine.is_running() is False


def test_default_workspace_under_home():
    assert default_workspace().endswith("gui_workspace")


def test_queue_log_handler_enqueues_and_keeps_tail():
    events: queue.Queue = queue.Queue()
    handler = QueueLogHandler(events)
    record = logging.LogRecord("jiuwensymbiosis", logging.WARNING, __file__, 1, "视觉检测未就绪", None, None)
    handler.emit(record)
    tag, payload = events.get_nowait()
    assert tag == "log"
    assert payload["level"] == "WARNING"
    assert "视觉检测未就绪" in payload["msg"]
    assert "视觉检测未就绪" in handler.log_tail()
