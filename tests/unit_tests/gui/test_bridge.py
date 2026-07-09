# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""UIBridgeRail:用假 ctx / emitter 驱动钩子,断言事件序列(不依赖 Qt)。"""

from __future__ import annotations

from typing import Any

import pytest

from jiuwensymbiosis.gui.bridge import UIBridgeRail
from jiuwensymbiosis.gui.mock_sessions import build_mock_robot_session


class _Resp:
    """假 LLM 回复,只暴露 after_model_call 读取的 ``content``。"""

    def __init__(self, content: Any) -> None:
        self.content = content


class _Inputs:
    def __init__(self, tool_name: str, tool_args: Any, tool_result: Any = None, response: Any = None) -> None:
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.tool_result = tool_result
        self.response = response


class _Ctx:
    def __init__(self, inputs: _Inputs, exception: Exception | None = None) -> None:
        self.inputs = inputs
        self.extra: dict = {}
        self.exception = exception
        self.forced: dict | None = None

    def request_force_finish(self, result: dict) -> None:
        self.forced = result


class _Emitter:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def step_started(self, d: dict) -> None:
        self.events.append(("start", d))

    def step_finished(self, d: dict) -> None:
        self.events.append(("finish", d))

    def frame(self, rgb: Any) -> None:
        self.events.append(("frame", None))

    def narration(self, t: str) -> None:
        self.events.append(("narration", t))

    def safety_event(self, d: dict) -> None:
        self.events.append(("safety", d))


@pytest.fixture
def session():
    return build_mock_robot_session()


async def test_motion_tool_emits_start_finish_and_frame(session):
    emitter = _Emitter()
    rail = UIBridgeRail(emitter, session)
    ctx = _Ctx(_Inputs("goto_xyzr", {"x": 1, "y": 2, "z": 80}, tool_result={"ok": True}))

    await rail.before_tool_call(ctx)
    await rail.after_tool_call(ctx)

    kinds = [e[0] for e in emitter.events]
    assert kinds == ["start", "narration", "finish", "frame"]
    finish = next(d for k, d in emitter.events if k == "finish")
    assert finish["ok"] is True
    assert finish["index"] == 1


async def test_non_motion_tool_has_no_frame(session):
    emitter = _Emitter()
    rail = UIBridgeRail(emitter, session)
    ctx = _Ctx(_Inputs("get_pose", {}))

    await rail.before_tool_call(ctx)
    await rail.after_tool_call(ctx)

    assert [e[0] for e in emitter.events] == ["start", "narration", "finish"]


async def test_safety_rejection_emits_failed_step_and_safety_event(session):
    emitter = _Emitter()
    rail = UIBridgeRail(emitter, session)
    err = ValueError("SafetyRail: refusing goto_xyzr: x=9999 out of bounds")
    ctx = _Ctx(_Inputs("goto_xyzr", {"x": 9999, "y": 0, "z": 80}), exception=err)

    await rail.on_tool_exception(ctx)

    kinds = [e[0] for e in emitter.events]
    assert "safety" in kinds
    finish = next(d for k, d in emitter.events if k == "finish")
    assert finish["ok"] is False
    assert "SafetyRail" in finish["error"]


async def test_stop_request_forces_finish_without_step(session):
    emitter = _Emitter()
    rail = UIBridgeRail(emitter, session, should_stop=lambda: True)
    ctx = _Ctx(_Inputs("home", {}))

    await rail.before_tool_call(ctx)

    assert emitter.events == []
    assert ctx.forced is not None
    assert ctx.forced.get("result_type") == "stopped"


async def test_assistant_text_captured_and_attached_to_step(session):
    """after_model_call 抓到的本轮 LLM 文本,应挂到随后 start/finish 步骤的详情里。"""
    emitter = _Emitter()
    rail = UIBridgeRail(emitter, session)

    await rail.after_model_call(_Ctx(_Inputs("", None, response=_Resp("我先回到初始位置再识别物体。"))))
    ctx = _Ctx(_Inputs("home", {}, tool_result={"ok": True}))
    await rail.before_tool_call(ctx)
    await rail.after_tool_call(ctx)

    start = next(d for k, d in emitter.events if k == "start")
    finish = next(d for k, d in emitter.events if k == "finish")
    assert start["assistant_text"] == "我先回到初始位置再识别物体。"
    assert finish["assistant_text"] == "我先回到初始位置再识别物体。"


async def test_non_string_assistant_content_is_ignored(session):
    """content 非字符串(如多模态分块)时留空,不得抛异常。"""
    emitter = _Emitter()
    rail = UIBridgeRail(emitter, session)

    await rail.after_model_call(_Ctx(_Inputs("", None, response=_Resp([{"type": "image"}]))))
    ctx = _Ctx(_Inputs("get_pose", {}))
    await rail.before_tool_call(ctx)
    await rail.after_tool_call(ctx)

    start = next(d for k, d in emitter.events if k == "start")
    assert start["assistant_text"] == ""


async def test_robot_control_is_unwrapped_in_step_label(session):
    emitter = _Emitter()
    rail = UIBridgeRail(emitter, session)
    ctx = _Ctx(_Inputs("robot_control", {"action": "close_gripper", "params": {}}, tool_result={"ok": True}))

    await rail.before_tool_call(ctx)
    await rail.after_tool_call(ctx)

    start = next(d for k, d in emitter.events if k == "start")
    assert start["tool"] == "close_gripper"
    assert start["label"] == "闭合夹爪抓取"
