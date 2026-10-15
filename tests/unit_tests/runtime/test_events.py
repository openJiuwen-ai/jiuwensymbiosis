# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the headless agent task-event rail."""

from __future__ import annotations

from typing import Any

import pytest

from jiuwensymbiosis.runtime.events import TaskEventRail, _result_ok_error, summarize_output


class _Inputs:
    def __init__(
        self,
        tool_name: str,
        tool_args: Any,
        tool_result: Any = None,
        response: Any = None,
    ) -> None:
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.tool_result = tool_result
        self.response = response


class _Ctx:
    def __init__(self, inputs: _Inputs, exception: Exception | None = None) -> None:
        self.inputs = inputs
        self.extra: dict[str, Any] = {}
        self.exception = exception
        self.forced: dict[str, Any] | None = None

    def request_force_finish(self, result: dict[str, Any]) -> None:
        self.forced = result


class _Emitter:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def step_started(self, data: dict[str, Any]) -> None:
        self.events.append(("step_started", data))

    def step_finished(self, data: dict[str, Any]) -> None:
        self.events.append(("step_finished", data))

    def frame(self, rgb: Any) -> None:
        self.events.append(("frame", rgb))

    def step_frame(self, index: int, rgb: Any) -> None:
        self.events.append(("step_frame", {"index": index, "rgb": rgb}))

    def safety_event(self, data: dict[str, Any]) -> None:
        self.events.append(("safety_event", data))


class _Env:
    def __init__(self, rgb: Any = None, *, error: Exception | None = None) -> None:
        self.rgb = rgb
        self.error = error

    def get_observation(self):
        if self.error:
            raise self.error
        return type("Observation", (), {"rgb": self.rgb})()


class _Session:
    def __init__(self, *, rgb: Any = None, overlay: Any = None, env_error: Exception | None = None) -> None:
        self.env = _Env(rgb, error=env_error)
        self.api = _Api(overlay)


class _Api:
    def __init__(self, overlay: Any) -> None:
        self.overlay = overlay

    def pop_last_detection_overlay(self) -> Any:
        overlay, self.overlay = self.overlay, None
        return overlay


class _ToolOutput:
    def __init__(self, success: bool, error: str = "", data: Any = None) -> None:
        self.success = success
        self.error = error
        self.data = data


async def _run_step(
    rail: TaskEventRail,
    *,
    name: str,
    args: Any = None,
    result: Any = None,
) -> _Ctx:
    ctx = _Ctx(_Inputs(name, {} if args is None else args, tool_result=result))
    await rail.before_tool_call(ctx)
    await rail.after_tool_call(ctx)
    return ctx


async def test_motion_capability_emits_machine_facts_and_one_frame():
    emitter = _Emitter()
    rgb = object()
    rail = TaskEventRail(emitter, _Session(rgb=rgb))

    ctx = await _run_step(rail, name="goto_xyzr", args={"x": 1, "y": 2}, result={"ok": True})

    assert [kind for kind, _ in emitter.events] == ["step_started", "step_finished", "frame"]
    started = emitter.events[0][1]
    finished = emitter.events[1][1]
    assert started == {"index": 1, "tool": "goto_xyzr", "params": {"x": 1, "y": 2}, "assistant_text": ""}
    assert finished["ok"] is True
    assert finished["index"] == 1
    assert finished["tool"] == "goto_xyzr"
    assert "label" not in started and "narration" not in started
    assert emitter.events[2][1] is rgb
    assert ctx.extra["_task_event_open"] is False


async def test_action_classification_uses_capability_and_home_motion_tag():
    emitter = _Emitter()
    rail = TaskEventRail(emitter, _Session(rgb="snapshot"))

    await _run_step(rail, name="get_pose", result={"ok": True})
    await _run_step(rail, name="close_gripper", result={"ok": True})
    await _run_step(rail, name="home", result={"ok": True})

    assert [kind for kind, _ in emitter.events] == [
        "step_started",
        "step_finished",
        "step_started",
        "step_finished",
        "frame",
        "step_started",
        "step_finished",
        "frame",
    ]


async def test_detection_capability_pins_overlay_after_step_finished():
    emitter = _Emitter()
    overlay = object()
    rail = TaskEventRail(emitter, _Session(overlay=overlay))

    await _run_step(rail, name="get_grasp_info_simple", args={"object_name": "box"}, result={"ok": True})

    assert [kind for kind, _ in emitter.events] == ["step_started", "step_finished", "step_frame"]
    assert emitter.events[-1][1] == {"index": 1, "rgb": overlay}


async def test_detection_without_overlay_emits_no_frame():
    emitter = _Emitter()
    rail = TaskEventRail(emitter, _Session())

    await _run_step(rail, name="get_grasp_info_simple", result={"ok": True})

    assert [kind for kind, _ in emitter.events] == ["step_started", "step_finished"]


async def test_robot_control_json_args_are_unwrapped():
    emitter = _Emitter()
    rail = TaskEventRail(emitter, _Session(rgb="frame"))
    ctx = _Ctx(
        _Inputs(
            "robot_control",
            '{"action":"close_gripper","params":{"force_n":2}}',
            tool_result={"ok": True},
        )
    )

    await rail.before_tool_call(ctx)
    await rail.after_tool_call(ctx)

    assert emitter.events[0][1]["tool"] == "close_gripper"
    assert emitter.events[0][1]["params"] == {"force_n": 2}
    assert emitter.events[-1][0] == "frame"


async def test_assistant_text_is_attached_without_presentation_event():
    emitter = _Emitter()
    rail = TaskEventRail(emitter, _Session())
    response = type("Response", (), {"content": "  inspect first  "})()
    await rail.after_model_call(_Ctx(_Inputs("", None, response=response)))

    await _run_step(rail, name="get_pose", result="pose")

    assert emitter.events[0][1]["assistant_text"] == "inspect first"
    assert emitter.events[1][1]["assistant_text"] == "inspect first"
    assert all(kind != "narration" for kind, _ in emitter.events)


async def test_exception_emits_failed_step_and_safety_fact():
    emitter = _Emitter()
    rail = TaskEventRail(emitter, _Session())
    error = ValueError("SafetyRail: refusing goto_xyzr: out of bounds")
    ctx = _Ctx(_Inputs("goto_xyzr", {"x": 9999}), exception=error)

    await rail.on_tool_exception(ctx)

    assert [kind for kind, _ in emitter.events] == ["step_started", "step_finished", "safety_event"]
    assert emitter.events[1][1]["ok"] is False
    assert "SafetyRail" in emitter.events[1][1]["error"]
    assert emitter.events[2][1] == {"rail": "SafetyRail", "kind": "reject", "detail": str(error)}


async def test_stop_forces_machine_stopped_result_without_step_event():
    emitter = _Emitter()
    rail = TaskEventRail(emitter, _Session(), should_stop=lambda: True)
    ctx = _Ctx(_Inputs("home", {}))

    await rail.before_tool_call(ctx)

    assert emitter.events == []
    assert ctx.forced == {"output": "stopped", "result_type": "stopped"}


async def test_frame_acquisition_failure_is_best_effort():
    emitter = _Emitter()
    rail = TaskEventRail(emitter, _Session(env_error=RuntimeError("camera offline")))

    await _run_step(rail, name="goto_xyzr", result={"ok": True})

    assert [kind for kind, _ in emitter.events] == ["step_started", "step_finished"]


@pytest.mark.parametrize(
    ("method", "tool_name", "exception"),
    [
        ("step_started", "get_pose", None),
        ("step_finished", "get_pose", None),
        ("frame", "goto_xyzr", None),
        ("step_frame", "get_grasp_info_simple", None),
        ("safety_event", "get_pose", ValueError("SafetyRail: reject")),
    ],
)
async def test_emitter_fact_write_failures_propagate(method, tool_name, exception):
    class _FailingEmitter(_Emitter):
        def __getattribute__(self, name: str):
            if name == method:

                def fail(*args):
                    raise OSError(f"{method} persistence failed")

                return fail
            return super().__getattribute__(name)

    emitter = _FailingEmitter()
    rail = TaskEventRail(emitter, _Session(rgb="frame", overlay="overlay"))
    ctx = _Ctx(_Inputs(tool_name, {}, tool_result={"ok": True}), exception=exception)

    if exception is not None:
        with pytest.raises(OSError, match=f"{method} persistence failed"):
            await rail.on_tool_exception(ctx)
    elif method == "step_started":
        with pytest.raises(OSError, match=f"{method} persistence failed"):
            await rail.before_tool_call(ctx)
    else:
        await rail.before_tool_call(ctx)
        with pytest.raises(OSError, match=f"{method} persistence failed"):
            await rail.after_tool_call(ctx)


def test_wrapped_and_inner_action_failures_are_both_recognized():
    assert _result_ok_error({"ok": False, "reason": "outer failure"}) == (False, "outer failure")
    assert _result_ok_error(_ToolOutput(False, "driver failed")) == (False, "driver failed")
    wrapped = _ToolOutput(True, data={"action": "locate_for_grasp", "result": {"ok": False, "reason": "no depth"}})
    assert _result_ok_error(wrapped) == (False, "no depth")
    assert _result_ok_error("unstructured success") == (True, "")


def test_summary_is_truncated_only_after_limit():
    assert summarize_output("short") == "'short'"
    summary = summarize_output("x" * 17000)
    assert len(summary) == 16001
    assert summary.endswith("…")
