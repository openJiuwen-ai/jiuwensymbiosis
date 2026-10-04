"""Workbench maps shared runtime facts to labels and image presentation."""

import asyncio
import logging
import queue
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.runtime import Runtime, worker
from jiuwensymbiosis.runtime.events import TaskEventRail
from jiuwensymbiosis_gui.workbench import registry
from jiuwensymbiosis_gui.workbench.run_engine import (
    QueueLogHandler,
    RunEngine,
    default_workspace,
    strip_vision_services,
)
from tests.mocks.mock_session import make_mock_session


@pytest.fixture(autouse=True)
def temporary_resource_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("JIUWENSYMBIOSIS_RUNTIME_DIR", str(tmp_path / "locks"))


@dataclass
class Binding:
    workspace: str
    session: object
    binding_id: str = "binding"
    fingerprint: str = "fingerprint"
    adapter: str = "piper"
    resources: tuple = ("device:fake",)

    def config_data(self):
        return {"agent": {"exec_mode": "stepagent", "enable_tracing": False}}

    def build_session(self):
        return self.session

    def without_sidecars(self):
        return self


@pytest.fixture
def running_engine(tmp_path, monkeypatch):
    runtime = Runtime(tmp_path, resource_directory=tmp_path / "locks")
    binding = Binding(str(tmp_path), make_mock_session())

    def prepare(*args, **kwargs):
        runtime._bindings[binding.binding_id] = binding
        return binding

    monkeypatch.setattr(runtime, "prepare_binding", prepare)

    def run(session, query, config, **kwargs):
        rail = next(rail for rail in config.extra_rails if isinstance(rail, TaskEventRail))

        async def steps():
            for tool in ("home", "get_grasp_info_simple", "goto_xyzr", "close_gripper"):
                ctx = SimpleNamespace(
                    inputs=SimpleNamespace(tool_name=tool, tool_args={}, tool_result={"ok": True}), extra={}
                )
                await rail.before_tool_call(ctx)
                await rail.after_tool_call(ctx)

        asyncio.run(steps())
        return {"result_type": "answer", "output": "done"}

    monkeypatch.setattr(worker, "run_robot_task", run)
    engine = RunEngine(registry.get_task("pick_box"), {}, workspace=str(tmp_path), body_key="piper", runtime=runtime)
    engine.start()
    engine.join(3)
    yield engine
    runtime.close()
    runtime.store.close()


def test_runtime_facts_are_rendered_in_order(running_engine):
    events = running_engine.drain()
    steps = [data for kind, data in events if kind == "step_finished"]
    assert [step["tool"] for step in steps] == ["home", "get_grasp_info_simple", "goto_xyzr", "close_gripper"]
    assert all(step["label"] for step in steps)
    finished = [data for kind, data in events if kind == "run_finished"]
    assert finished[0]["ok"]
    assert isinstance(finished[0]["log_tail"], str)
    assert not running_engine.is_running()


def test_two_page_subscriptions_do_not_consume_each_others_events(running_engine):
    other_page = running_engine.subscribe()
    first = running_engine.drain()
    assert other_page.drain() == first
    assert running_engine.drain() == []
    assert other_page.drain() == []


def test_refresh_does_not_retry_a_rejected_submission(tmp_path):
    from unittest.mock import Mock

    runtime = Mock()
    runtime.prepare_binding.side_effect = ValueError("invalid config")
    engine = RunEngine(registry.get_task("pick_box"), {}, workspace=str(tmp_path), body_key="piper", runtime=runtime)
    engine.start()
    refreshed = engine.subscribe()
    refreshed.start()
    runtime.prepare_binding.assert_called_once()
    assert refreshed.drain()[0][0] == "run_finished"


def test_preview_is_encoded_only_in_workbench(running_engine):
    events = running_engine.drain()
    frames = [data for kind, data in events if kind == "frame"]
    assert frames
    assert all(uri.startswith("data:image/jpeg;base64,") for uri in frames)


def test_step_frame_reference_is_rendered(running_engine):
    engine = running_engine
    ref = engine._runtime.artifacts.frame(engine.job_id, np.zeros((4, 4, 3), dtype=np.uint8), slot="1")
    engine._runtime.store.append_event(engine.job_id, "step_frame", {"index": 1, "artifact": ref})
    frame = [data for kind, data in engine.drain() if kind == "step_frame"][0]
    assert frame["index"] == 1
    assert frame["uri"].startswith("data:image/jpeg;base64,")


def test_rerun_with_keeps_body_and_task_but_takes_the_given_config(tmp_path):
    task = registry.get_task("pick_box")
    config = {"env": {"cfg": {"prompt": "把黑盒放到白盒上"}}, "agent": {"mode": "tool"}}
    engine = RunEngine(task, config, workspace=str(tmp_path), body_key="piper")

    edited = {"env": {"cfg": {"prompt": "改过的指令"}}, "agent": {"mode": "tool"}}
    twin = engine.rerun_with(edited, config_source=engine.config_source)

    assert twin is not engine
    assert twin._task is task and twin._workspace == str(tmp_path)
    assert twin._body_key == "piper"
    # 重跑用的是传进来的配置,不是引擎开跑时那份快照。
    assert twin._config.get("env.cfg.prompt") == "改过的指令"
    assert engine._config.get("env.cfg.prompt") == "把黑盒放到白盒上"
    twin._config.set("env.cfg.prompt", "又改了")  # 深拷贝:动新引擎不回写调用方的 dict
    assert edited["env"]["cfg"]["prompt"] == "改过的指令"


def test_engine_exposes_the_body_and_task_it_ran(tmp_path):
    task = registry.get_task("pick_box")
    engine = RunEngine(task, {}, workspace=str(tmp_path), body_key="piper")
    assert engine.body_key == "piper"
    assert engine.task_key == task.key


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


def test_strip_vision_overrides_remote_and_nested_legacy_detector():
    cfg = {
        "detector": {"mode": "remote", "endpoint": {"url": "https://inference.invalid"}},
        "env": {"cfg": {"api_servers": [{"_target_": "x.grounding_dino"}]}},
    }
    out = strip_vision_services(cfg)
    assert out["detector"] == {"mode": "disabled"}
    assert "api_servers" not in out["env"]["cfg"]
    assert cfg["detector"]["mode"] == "remote"


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


def test_expired_event_cursor_restores_final_result_from_snapshot(running_engine, monkeypatch):
    runtime = running_engine._runtime
    final = runtime.get_job(running_engine.job_id)
    monkeypatch.setattr(runtime, "read_events", lambda *a, **kw: {"events": [], "gap": True, "next_seq": 10001})
    events = running_engine.drain()
    assert ("run_finished", final["result"]) in events
    assert any(kind == "log" and "保留窗口" in data["msg"] for kind, data in events)


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
