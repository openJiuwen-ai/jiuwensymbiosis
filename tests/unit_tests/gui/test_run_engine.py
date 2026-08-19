# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""run_engine:后台线程 + 事件队列驱动一次模拟运行(纯逻辑,无 Qt / 无 nicegui)。"""

from __future__ import annotations

import asyncio
import logging
import queue
from types import SimpleNamespace

import jiuwensymbiosis.gui.run_engine as run_engine_module
from jiuwensymbiosis.gui import registry
from jiuwensymbiosis.gui.bridge import UIBridgeRail
from jiuwensymbiosis.gui.run_engine import (
    QueueLogHandler,
    RunEngine,
    default_workspace,
    strip_vision_services,
)
from tests.mocks.mock_session import make_mock_session

# One full identify→grasp→lift chain using only tools the hardware-free session exposes.
_SCRIPT = [
    {"tool": "home", "args": {}},
    {"tool": "get_grasp_info_simple", "args": {"object_name": "black box"}},
    {"tool": "goto_xyzr", "args": {"x": 230, "y": 0, "z": 90}},
    {"tool": "close_gripper", "args": {}},
    {"tool": "goto_xyzr", "args": {"x": 230, "y": 0, "z": 250}},
]


def _tags(events):
    return [tag for tag, _ in events]


def _use_hardware_free_session(monkeypatch, tmp_path):
    """Make ``RunEngine._build`` construct a MockArmEnv-backed session instead of a real robot."""
    stub_body = SimpleNamespace(
        build_real_session=lambda _cfg: make_mock_session(),
        config_path=lambda: tmp_path / "body.yaml",
    )
    monkeypatch.setattr(run_engine_module, "get_body", lambda _key: stub_body)


def _patch_script_runner(monkeypatch, script):
    """Drive the UI rail directly; RunEngine unit tests do not need DeepAgent."""

    def fake_run_robot_task(_session, _query, cfg, *, conversation_id, cancel_token=None):
        del conversation_id, cancel_token
        bridge = next(rail for rail in cfg.extra_rails if isinstance(rail, UIBridgeRail))

        async def drive_script():
            for step in script:
                inputs = SimpleNamespace(
                    tool_name=step["tool"],
                    tool_args=step.get("args", {}),
                    tool_result={"ok": True},
                )
                ctx = SimpleNamespace(inputs=inputs, extra={})
                await bridge.before_tool_call(ctx)
                await bridge.after_tool_call(ctx)

        asyncio.run(drive_script())
        return {"output": "模拟任务完成", "result_type": "answer"}

    monkeypatch.setattr(run_engine_module, "run_robot_task", fake_run_robot_task)


def test_run_emits_ordered_event_stream(tmp_path, monkeypatch):
    _use_hardware_free_session(monkeypatch, tmp_path)
    _patch_script_runner(monkeypatch, _SCRIPT)
    task = registry.get_task("pick_box")
    config = {
        "env": {"cfg": {"prompt": "把黑盒放到白盒上"}},
        "agent": {
            "mode": "tool",
            "exec_mode": "stepagent",
            "max_iterations": 20,
            "enable_visual_feedback": False,
            "enable_tracing": False,
        },
    }
    engine = RunEngine(task, config, workspace=str(tmp_path), body_key="piper")
    engine.start()
    engine.join(timeout=5)
    assert not engine.is_running()

    events = engine.drain()
    tags = _tags(events)
    assert tags[0] == "run_started"
    assert tags[-1] == "run_finished"

    meta = events[0][1]
    assert meta["body"] == "piper"

    finished = [payload for tag, payload in events if tag == "step_finished"]
    tools = [f["tool"] for f in finished]
    assert tools == [step["tool"] for step in _SCRIPT]  # 忠实回放脚本序列
    assert all(f["ok"] for f in finished)

    result = events[-1][1]
    assert result["ok"] is True


def test_run_frames_are_encoded_data_uris(tmp_path, monkeypatch):
    _use_hardware_free_session(monkeypatch, tmp_path)
    _patch_script_runner(monkeypatch, _SCRIPT)
    task = registry.get_task("pick_box")
    config = {
        "agent": {
            "mode": "tool",
            "exec_mode": "stepagent",
            "max_iterations": 20,
            "enable_visual_feedback": False,
            "enable_tracing": False,
        }
    }
    engine = RunEngine(task, config, workspace=str(tmp_path), body_key="piper")
    engine.start()
    engine.join(timeout=5)
    assert not engine.is_running()

    frames = [payload for tag, payload in engine.drain() if tag == "frame"]
    assert frames  # 初始帧 + 运动/抓取后各刷新
    assert all(isinstance(uri, str) and uri.startswith("data:image/jpeg;base64,") for uri in frames)


def test_clone_reuses_same_params_with_independent_config(tmp_path):
    task = registry.get_task("pick_box")
    config = {"env": {"cfg": {"prompt": "把黑盒放到白盒上"}}, "agent": {"mode": "tool"}}
    engine = RunEngine(task, config, workspace=str(tmp_path), body_key="piper")

    twin = engine.clone()

    assert twin is not engine
    assert twin._task is task and twin._workspace == str(tmp_path)
    assert twin._body_key == "piper"
    assert twin._config.data == engine._config.data
    twin._config.set("env.cfg.prompt", "改了")  # 深拷贝:动克隆不影响原引擎
    assert engine._config.get("env.cfg.prompt") == "把黑盒放到白盒上"


def test_drain_is_empty_before_start(tmp_path):
    engine = RunEngine(registry.get_task("pick_box"), {}, workspace=str(tmp_path), body_key="piper")
    assert engine.drain() == []
    assert engine.is_running() is False


def test_strip_vision_services_removes_detector_and_camera():
    cfg = {
        "api_servers": [{"_target_": "x.grounding_dino_sam2_server.main"}],
        "env": {"cfg": {"low_level": {"camera_serial": "123", "port": "/dev/ttyUSB0"}}},
    }
    out = strip_vision_services(cfg)
    assert "api_servers" not in out  # 检测器 sidecar 不再 spawn
    assert "camera_serial" not in out["env"]["cfg"]["low_level"]  # 相机不打开
    assert out["env"]["cfg"]["low_level"]["port"] == "/dev/ttyUSB0"  # 非视觉字段保留
    assert "api_servers" in cfg  # 深拷贝:原配置不动


def test_disable_vision_strips_real_session_config(tmp_path):
    task = registry.get_task("pick_banana")
    config = {
        "gui": {"disable_vision": True},
        "api_servers": [{"_target_": "x.grounding_dino_sam2_server.main"}],
        "env": {"cfg": {"low_level": {"camera_serial": "123"}, "prompt": "抓香蕉"}},
    }
    engine = RunEngine(task, config, workspace=str(tmp_path), body_key="so101")
    real = engine._real_session_config()
    assert "api_servers" not in real
    assert "camera_serial" not in real.get("env", {}).get("cfg", {}).get("low_level", {})


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
