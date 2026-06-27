# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""VisualFeedbackRail — inject a camera frame back into the agent context after
each motion / grasp tool call, as a generic openjiuwen rail.

What it does, on every ``after_tool_call``:
1. If the just-executed tool's tags include ``"motion"`` or ``"grasp"`` (or
   the tool name is in ``watch_tools``), grab the latest RGB frame from
   ``session.env.get_observation()``.
2. Encode as base64 JPEG.
3. Append a synthetic user message to the agent's ModelContext containing the
   image and a short directive ("verify result, decide next step").

Requires a VLM. The rail no-ops gracefully if the env reports no rgb frame
or if PIL is missing.

Why a rail and not glue inside the tool? Because the same rail then works
for any robot+tool that emits the right tag — no copy-paste per task.
"""

from __future__ import annotations

import base64
import io
import logging
from collections.abc import Callable
from typing import Any

from jiuwensymbiosis.agent.abstractions import AgentRail
from jiuwensymbiosis.agent.trace import TraceEventSink

logger = logging.getLogger(__name__)

_DEFAULT_TRIGGER_TAGS = frozenset({"motion", "grasp"})


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
            trace_sink: Optional sink notified each time a frame is injected
                (set by ``build_robot_agent`` when tracing is on).
            frame_sink: Optional callable ``(rgb_ndarray, tool_name) -> path``
                that persists the *same* frame being injected, so the on-disk
                trace frames match what the agent saw. Set by the builder when
                tracing + ``trace_save_frames`` are both on.
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

    # ----------------------------------------------------------- rail callback
    async def after_tool_call(self, ctx: Any) -> None:
        """Capture and inject a frame after a motion/grasp tool call."""
        inputs = getattr(ctx, "inputs", None)
        tool_name = getattr(inputs, "tool_name", "") or ""
        tool_args = getattr(inputs, "tool_args", None)
        if not tool_name or not self._should_trigger(tool_name, tool_args=tool_args):
            return
        n_so_far = len(ctx.extra.get("visual_feedback_injected", []))
        if n_so_far >= self.max_frames_per_invoke:
            return
        rgb = self._grab_frame_rgb()
        if rgb is None:
            return
        # Persist the same frame the agent will see, if a sink is wired.
        frame_path: str | None = None
        if self.frame_sink is not None:
            try:
                frame_path = self.frame_sink(rgb, tool_name)
            except (TypeError, ValueError, OSError) as exc:
                logger.warning("VisualFeedbackRail: frame_sink failed: %s", exc)
        b64 = _encode_jpeg_b64(rgb, quality=self.jpeg_quality)
        if b64 is None:
            return
        self._inject(ctx, b64, tool_name)
        ctx.extra.setdefault("visual_feedback_injected", []).append(tool_name)
        self._notify_inject(tool_name, frame_path)

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
        # Look up the tool meta on the api to read tags.
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

    def _notify_inject(self, tool_name: str, frame_path: str | None) -> None:
        sink = self.trace_sink
        if sink is None:
            return
        try:
            sink.record_rail_event(
                rail_name="VisualFeedback",
                kind="inject_frame",
                detail={"tool_name": tool_name, "frame_path": frame_path},
                success=True,
            )
        except (AttributeError, TypeError, ValueError):
            pass

    def _inject(self, ctx: Any, b64: str, tool_name: str) -> None:
        """Append a synthetic user message with the image into the model context.

        openjiuwen's ``ModelContext`` shape evolves; we look up
        ``add_message`` / ``append_message`` and fall back to mutating
        ``messages``. If neither is available we record into ``ctx.extra``
        so downstream rails (or post-mortem inspection) can still see it.
        """
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": f"[after {tool_name}] {self.directive_text}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
            ],
        }
        mc = getattr(ctx, "context", None)
        for attr in ("add_message", "append_message", "add_user_message"):
            fn = getattr(mc, attr, None) if mc is not None else None
            if callable(fn):
                try:
                    fn(message)
                    return
                except Exception:  # noqa: BLE001
                    pass
        if mc is not None and hasattr(mc, "messages"):
            try:
                mc.messages.append(message)
                return
            except Exception:  # noqa: BLE001
                pass
        # Last resort — make it visible to downstream rails / debugging.
        ctx.extra.setdefault("visual_feedback_pending", []).append(message)
