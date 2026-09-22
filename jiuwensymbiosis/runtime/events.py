# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Headless task events collected from agent tool-call hooks.

The rail emits machine-readable execution facts through a small emitter
interface. Presentation-specific labels and narration belong to consumers.
"""

from __future__ import annotations

import json
import time
from typing import Any

from jiuwensymbiosis.agent.abstractions import AgentRail
from jiuwensymbiosis.agent.observation import action_observation, begin_action_observation
from jiuwensymbiosis.api.actions import ACTIONS
from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["TaskEventRail", "summarize_output"]

_MAX_SUMMARY = 16000
_INDEX_KEY = "_task_event_index"
_STARTED_KEY = "_task_event_started_at"
_OPEN_KEY = "_task_event_open"


def summarize_output(result: Any) -> str:
    """Return a bounded representation of a tool result for event consumers."""
    text = repr(result)
    return text if len(text) <= _MAX_SUMMARY else text[:_MAX_SUMMARY] + "…"


def _extract(ctx: Any) -> tuple[str, dict[str, Any]]:
    """Read the effective action name and parameters from a rail context."""
    inputs = getattr(ctx, "inputs", None)
    name = getattr(inputs, "tool_name", "") or ""
    args = getattr(inputs, "tool_args", None)
    if isinstance(args, str) and args:
        try:
            parsed = json.loads(args)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            args = parsed
    params = args if isinstance(args, dict) else {}
    if name == "robot_control":
        action = params.get("action", "")
        action_params = params.get("params", {})
        if action:
            return str(action), action_params if isinstance(action_params, dict) else {}
    return name, params


def _result_ok_error(result: Any) -> tuple[bool, str]:
    """Read success and error from direct results and wrapped tool outputs.

    A call can fail at the tool boundary, or complete successfully while the
    inner action result reports ``ok=False``. Both levels must be considered.
    Missing success fields retain the historical default of success.
    """
    if isinstance(result, dict):
        ok = result.get("ok", result.get("success", True))
        err = result.get("error") or result.get("reason") or ""
        data = result.get("data")
    else:
        ok = getattr(result, "success", getattr(result, "ok", True))
        err = getattr(result, "error", "") or ""
        data = getattr(result, "data", None)
    if not ok:
        return False, str(err)
    inner = data.get("result") if isinstance(data, dict) and "result" in data else data
    if isinstance(inner, dict) and inner.get("ok") is False:
        return False, str(inner.get("reason") or inner.get("error") or "")
    return True, str(err)


def _event_class(tool_name: str) -> str | None:
    """Classify actions from the shared ActionSpec vocabulary.

    Motion/grasp actions trigger an ordinary post-action frame. Their shared
    ActionSpec capability identifies the domain; the matching action tag
    distinguishes state-changing calls from reads such as ``get_pose``.
    Detection capability consumes the API's detection overlay instead. Home
    has no capability gate, so its explicit motion tag supplies that fact.
    """
    spec = ACTIONS.get(tool_name)
    if spec is None:
        return None
    capability = spec.capability
    if capability == "vision.detection":
        return "detection"
    if capability is not None and capability.startswith("motion.") and "motion" in spec.tags:
        return "frame"
    if capability is not None and capability.startswith("grasp.") and "grasp" in spec.tags:
        return "frame"
    if capability is None and "motion" in spec.tags:
        return "frame"
    return None


class TaskEventRail(AgentRail):
    """Emit structured step, safety, assistant-text and frame facts.

    ``emitter`` must provide ``step_started(dict)``, ``step_finished(dict)``,
    ``frame(rgb)``, ``step_frame(index, rgb)`` and ``safety_event(dict)``.
    Failures from those methods propagate so a caller that persists events can
    stop execution when it cannot record a fact. Camera/overlay acquisition is
    best-effort and does not turn an otherwise valid action into a failure.
    """

    # Observe after SafetyRail (priority 50); preserve the outcome or rejection.
    priority = 1

    def __init__(self, emitter: Any, session: Any, should_stop: Any = None) -> None:
        self.emitter = emitter
        self.session = session
        self.should_stop = should_stop
        self._counter = 0
        self._turn_text = ""

    async def after_model_call(self, ctx: Any) -> None:
        """Attach the latest textual model response to subsequent step facts."""
        response = getattr(getattr(ctx, "inputs", None), "response", None)
        content = getattr(response, "content", None)
        self._turn_text = content.strip() if isinstance(content, str) else ""

    async def before_tool_call(self, ctx: Any) -> None:
        """Emit a step-start fact, or request the agent to stop before a step."""
        begin_action_observation(ctx)
        if self.should_stop is not None and self.should_stop():
            ctx.request_force_finish({"output": "stopped", "result_type": "stopped"})
            return
        name, params = _extract(ctx)
        self._counter += 1
        ctx.extra[_INDEX_KEY] = self._counter
        ctx.extra[_STARTED_KEY] = time.monotonic()
        ctx.extra[_OPEN_KEY] = True
        self.emitter.step_started(
            {
                "index": self._counter,
                "tool": name,
                "params": params,
                "assistant_text": self._turn_text,
            }
        )

    async def after_tool_call(self, ctx: Any) -> None:
        """Emit a completed step and any associated frame fact."""
        name, params = _extract(ctx)
        index, duration = self._close(ctx)
        result = getattr(getattr(ctx, "inputs", None), "tool_result", None)
        ok, error = _result_ok_error(result)
        info: dict[str, Any] = {
            "index": index,
            "tool": name,
            "ok": ok,
            "duration_s": duration,
            "output": summarize_output(result),
            "params": params,
            "assistant_text": self._turn_text,
        }
        if not ok and error:
            info["error"] = error
        self.emitter.step_finished(info)

        event_class = _event_class(name)
        if event_class == "frame":
            self._emit_frame(ctx)
        elif event_class == "detection":
            self._emit_detection_overlay(index)

    async def on_tool_exception(self, ctx: Any) -> None:
        """Emit a failed step and a separate fact for SafetyRail rejections."""
        name, params = _extract(ctx)
        if not ctx.extra.get(_OPEN_KEY):
            # A higher-priority rail may reject before this rail observes the call.
            self._counter += 1
            ctx.extra[_INDEX_KEY] = self._counter
            ctx.extra[_STARTED_KEY] = time.monotonic()
            ctx.extra[_OPEN_KEY] = True
            self.emitter.step_started(
                {
                    "index": self._counter,
                    "tool": name,
                    "params": params,
                    "assistant_text": self._turn_text,
                }
            )
        index, duration = self._close(ctx)
        error = str(getattr(ctx, "exception", "") or "")
        self.emitter.step_finished(
            {
                "index": index,
                "tool": name,
                "ok": False,
                "duration_s": duration,
                "error": error,
                "params": params,
                "assistant_text": self._turn_text,
            }
        )
        if "SafetyRail" in error:
            self.emitter.safety_event({"rail": "SafetyRail", "kind": "reject", "detail": error})

    def _close(self, ctx: Any) -> tuple[int, float]:
        """Return the current step index and elapsed time, then close it."""
        index = int(ctx.extra.get(_INDEX_KEY, self._counter))
        started_at = ctx.extra.get(_STARTED_KEY)
        duration = round(time.monotonic() - started_at, 3) if isinstance(started_at, int | float) else 0.0
        ctx.extra[_OPEN_KEY] = False
        return index, duration

    def _emit_frame(self, ctx: Any) -> None:
        """Capture one observation frame; emitter errors remain fatal."""
        try:
            rgb = action_observation(ctx, self.session.env).rgb
        except Exception as exc:  # a missing preview frame must not fail the action
            logger.debug("capture task frame failed: %s", exc)
            return
        if rgb is not None:
            self.emitter.frame(rgb)

    def _emit_detection_overlay(self, index: int) -> None:
        """Pin the latest API overlay to its producing step when available."""
        api = getattr(self.session, "api", None)
        pop = getattr(api, "pop_last_detection_overlay", None)
        if not callable(pop):
            return
        try:
            overlay = pop()
        except Exception as exc:  # a missing diagnostic overlay must not fail the action
            logger.debug("pop detection overlay failed: %s", exc)
            return
        if overlay is not None:
            self.emitter.step_frame(index, overlay)
