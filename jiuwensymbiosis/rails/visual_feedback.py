# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""VisualFeedbackRail — inject a camera frame back into the agent context after
each motion / grasp tool call, as a generic openjiuwen rail.

Two-phase injection (fixes message-ordering bug where injecting in
``after_tool_call`` produced ``assistant(tool_calls) → user(image) →
tool(result)``, which OpenAI-style APIs reject — tool result must immediately
follow its tool call):

1. ``after_tool_call``: grab frame, encode, **stage** the message on
   ``ctx.extra["visual_feedback_pending"]``. Does not touch ModelContext.
2. ``before_model_call``: openjiuwen has by now written all ToolMessages, so
   flush pending → ``await ctx.context.add_messages(UserMessage(...))``,
   yielding ``assistant → tool(result) → user(image) → next model call``.
3. ``after_invoke``: drop any pending never flushed (no next model call to
   see it).

Requires a VLM. The rail no-ops gracefully if the env reports no rgb frame or
if PIL is missing. Injection failures never escape to the tool lifecycle
(would corrupt a successful action into a tool failure).
"""

from __future__ import annotations

import base64
import io
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jiuwensymbiosis.agent.abstractions import AgentRail
from jiuwensymbiosis.agent.trace import _TRACE_RAIL_KEY, TraceEventSink

logger = logging.getLogger(__name__)

_DEFAULT_TRIGGER_TAGS = frozenset({"motion", "grasp"})
_PENDING_KEY = "visual_feedback_pending"
_INJECTED_KEY = "visual_feedback_injected"


@dataclass
class _PendingFrame:
    """A frame staged in ``after_tool_call`` awaiting flush in ``before_model_call``.

    Carries the trace step and on-disk path captured at staging time so the
    flush-time trace event lands on the correct entry (not ``entries[-1]``,
    which by flush time may be a later step in a multi-tool iteration).
    """

    b64: str
    tool_name: str
    trace_step: int | None
    frame_path: str | None


def _encode_jpeg_b64(rgb_bgr_or_rgb: Any, quality: int = 80) -> str | None:
    """Best-effort JPEG encode + base64. Returns None on failure."""
    try:
        import numpy as np
        from PIL import Image

        arr = rgb_bgr_or_rgb
        if not isinstance(arr, np.ndarray):
            return None
        if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
            return None
        img = Image.fromarray(arr[..., :3].astype("uint8"))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:  # noqa: BLE001
        logger.warning("VisualFeedbackRail: encode failed: %s", exc)
        return None


class VisualFeedbackRail(AgentRail):
    """Inject post-action camera frames into the agent ModelContext.

    Args:
        session: The ``RobotSession`` (must expose ``.env``).
        watch_tools: Tool *names* that always trigger frame capture, regardless of tags.
        trigger_tags: ``robot_tool`` tags that trigger capture
            (default: motion + grasp).
        directive_text: Short instruction appended next to the image.
        max_frames_per_invoke: Cap to avoid context blow-up on long episodes.
            Counts both staged and flushed frames so the two-phase path can't
            bypass the limit.
        jpeg_quality: 1-95.
    """

    def __init__(
        self,
        session: Any,
        *,
        watch_tools: set[str] | None = None,
        trigger_tags: set[str] | None = None,
        directive_text: str = (
            "Frame captured after the previous action. Verify the action succeeded, then decide the next step."
        ),
        max_frames_per_invoke: int = 8,
        jpeg_quality: int = 80,
        trace_sink: TraceEventSink | None = None,
        frame_sink: Callable[[Any, str], str | None] | None = None,
    ) -> None:
        """Initialize the visual feedback rail.

        Args:
            trace_sink: Optional sink notified when a frame is *flushed* into
                the ModelContext (set by ``build_robot_agent`` when tracing on).
                Fires on success or failure — a failed flush is recorded with
                ``success=False`` so the trace never falsely claims injection.
            frame_sink: Optional callable ``(rgb_ndarray, tool_name) -> path``
                that persists the *same* frame being staged, so on-disk trace
                frames match what the agent will see. The returned path surfaces
                in the flush-time trace event's ``detail["frame_path"]``. Set by
                the builder when tracing + ``trace_save_frames`` are both on.
        """
        super().__init__()
        self.session = session
        self.watch_tools = set(watch_tools) if watch_tools else set()
        self.trigger_tags = frozenset(trigger_tags) if trigger_tags else _DEFAULT_TRIGGER_TAGS
        self.directive_text = directive_text
        self.max_frames_per_invoke = max_frames_per_invoke
        self.jpeg_quality = jpeg_quality
        self.trace_sink = trace_sink
        self.frame_sink = frame_sink

    # ----------------------------------------------------------- rail callbacks
    async def after_tool_call(self, ctx: Any) -> None:
        """Stage a frame after a motion/grasp tool call — do NOT inject yet.

        Injecting here would precede the ToolMessage openjiuwen writes after
        ``execute()`` returns, yielding an illegal ``… → user(image) →
        tool(result)`` order. We only capture + stage; the actual
        ``add_messages`` happens in ``before_model_call`` once ToolMessages are
        settled.
        """
        inputs = getattr(ctx, "inputs", None)
        tool_name = getattr(inputs, "tool_name", "") or ""
        tool_args = getattr(inputs, "tool_args", None)
        if not tool_name or not self._should_trigger(tool_name, tool_args=tool_args):
            return
        pending = ctx.extra.setdefault(_PENDING_KEY, [])
        n_so_far = len(pending) + len(ctx.extra.get(_INJECTED_KEY, []))
        if n_so_far >= self.max_frames_per_invoke:
            return
        rgb = self._grab_frame_rgb()
        if rgb is None:
            return
        frame_path: str | None = None
        if self.frame_sink is not None:
            try:
                frame_path = self.frame_sink(rgb, tool_name)
            except (TypeError, ValueError, OSError) as exc:
                logger.warning("VisualFeedbackRail: frame_sink failed: %s", exc)
        b64 = _encode_jpeg_b64(rgb, quality=self.jpeg_quality)
        if b64 is None:
            return
        pending.append(
            _PendingFrame(
                b64=b64,
                tool_name=tool_name,
                trace_step=self._current_trace_step(ctx),
                frame_path=frame_path,
            )
        )

    async def before_model_call(self, ctx: Any) -> None:
        """Flush staged frames now that ToolMessages are in the context.

        Fires at the top of each iteration, after the previous iteration's
        ToolMessages have been written, yielding the legal order
        ``assistant → tool(result) → user(image) → model call``.
        """
        pending = ctx.extra.pop(_PENDING_KEY, None)
        if not pending:
            return
        for pf in pending:
            ok = await self._inject(ctx, pf.b64, pf.tool_name)
            if ok:
                ctx.extra.setdefault(_INJECTED_KEY, []).append(pf.tool_name)
            self._notify_inject(pf.tool_name, ok, step=pf.trace_step, frame_path=pf.frame_path)

    async def after_invoke(self, ctx: Any) -> None:
        """Drop staged frames never flushed (no next model call to see them)."""
        ctx.extra.pop(_PENDING_KEY, None)

    # ------------------------------------------------------------------ helpers
    def _should_trigger(self, tool_name: str, *, tool_args: Any = None) -> bool:
        """Check whether the given tool name should trigger a frame capture.

        When ``RobotControlTool`` is used, the tool name is ``"robot_control"``
        and the actual action lives in ``tool_args["action"]``.  We unwrap this
        so that motion / grasp actions dispatched through the single entry point
        still trigger visual feedback.
        """
        effective_name = tool_name
        if tool_name == "robot_control":
            args = tool_args if isinstance(tool_args, dict) else {}
            action = args.get("action", "")
            if action:
                effective_name = str(action)
        if effective_name in self.watch_tools:
            return True
        api = getattr(self.session, "api", None)
        if api is None:
            return False
        method = getattr(api, effective_name, None)
        meta = getattr(method, "__robot_tool__", None)
        if meta is None:
            return False
        return bool(set(meta.tags) & self.trigger_tags)

    def _grab_frame_b64(self) -> str | None:
        """Capture the latest RGB frame and return it as a base64 JPEG."""
        rgb = self._grab_frame_rgb()
        if rgb is None:
            return None
        return _encode_jpeg_b64(rgb, quality=self.jpeg_quality)

    def _grab_frame_rgb(self) -> Any | None:
        """Capture the latest RGB frame as a raw ndarray (or None)."""
        try:
            obs = self.session.env.get_observation()
        except Exception as exc:  # noqa: BLE001
            logger.warning("VisualFeedbackRail: get_observation failed: %s", exc)
            return None
        return getattr(obs, "rgb", None)

    def _current_trace_step(self, ctx: Any) -> int | None:
        """Step number of the active trace entry, or None if no TraceRail.

        TraceRail stashes itself on ``ctx.extra[_TRACE_RAIL_KEY]`` at
        ``before_invoke``; its ``trace.current_step`` is the most recently
        created entry — the step this frame belongs to. Returns None when
        tracing is off or the trace isn't initialized yet.
        """
        trace_rail = getattr(ctx, "extra", {}).get(_TRACE_RAIL_KEY) if hasattr(ctx, "extra") else None
        if trace_rail is None:
            return None
        trace = getattr(trace_rail, "trace", None)
        if trace is None:
            return None
        step = getattr(trace, "current_step", None)
        return step if step and step > 0 else None

    def _notify_inject(self, tool_name: str, success: bool, *, step: int | None, frame_path: str | None) -> None:
        sink = self.trace_sink
        if sink is None:
            return
        detail = {"tool_name": tool_name, "frame_path": frame_path}
        # Step-aware sinks (TraceRail) get the precise entry; legacy sinks
        # fall back to record_rail_event via duck-typing.
        step_fn = getattr(sink, "record_rail_event_at_step", None)
        try:
            if step is not None and callable(step_fn):
                step_fn(
                    rail_name="VisualFeedback",
                    kind="inject_frame",
                    detail=detail,
                    success=success,
                    step=step,
                )
            else:
                sink.record_rail_event(
                    rail_name="VisualFeedback",
                    kind="inject_frame",
                    detail=detail,
                    success=success,
                )
        except (AttributeError, TypeError, ValueError):
            pass

    async def _inject(self, ctx: Any, b64: str, tool_name: str) -> bool:
        """Append a synthetic user message with the image into the ModelContext.

        Returns True only if the message was added; False on missing
        ModelContext (fast-path op-ctx), missing ``add_messages``, or injection
        failure. Never raises — a rail exception would corrupt the model call
        or (from ``after_tool_call``) turn a successful action into a tool
        failure via ``ON_TOOL_EXCEPTION``. ``except Exception`` deliberately
        leaves ``asyncio.CancelledError`` (a ``BaseException``) to propagate.
        """
        mc = getattr(ctx, "context", None)
        if mc is None or not hasattr(mc, "add_messages"):
            return False
        from openjiuwen.core.foundation.llm.schema.message import UserMessage

        message = UserMessage(
            content=[
                {"type": "text", "text": f"[after {tool_name}] {self.directive_text}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]
        )
        try:
            await mc.add_messages(message)
        except Exception as exc:  # noqa: BLE001 — see docstring re: CancelledError
            logger.warning("VisualFeedbackRail: add_messages failed: %s", exc)
            return False
        return True
