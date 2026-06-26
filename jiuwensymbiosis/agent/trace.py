# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Structured execution-trace recording for agent invocations.

This module adds a parallel rail (:class:`TraceRail`) that, when enabled via
``RobotAgentConfig.enable_tracing``, captures each tool call's name / args /
result / timing / observation snapshot, plus rail events (SafetyRail rejections,
RecoveryRail recovery, VisualFeedbackRail frame injections) and ``WARNING``+
log lines — then persists a single JSON trace to the workspace ``traces/``
directory on invoke completion.

Design:

- TraceRail is an :class:`~jiuwensymbiosis.agent.abstractions.AgentRail` using
  ``before_tool_call`` / ``after_tool_call`` / ``on_tool_exception`` /
  ``before_invoke`` / ``after_invoke`` hooks. It collects into an in-memory
  :class:`ExecutionTrace` (shared via ``ctx.extra``) and flushes once in
  ``after_invoke`` — one disk write per run.
- Rail events are pushed by the other rails via the :class:`TraceEventSink`
  Protocol (``trace_sink`` constructor arg on SafetyRail / RecoveryRail /
  VisualFeedbackRail). TraceRail implements it.
- Log lines are captured by :class:`~jiuwensymbiosis.utils.logging.TraceLogHandler`
  bound to this rail; no business code changes needed.

No tool / env / ``@robot_tool`` is modified.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from jiuwensymbiosis.agent.abstractions import AgentRail

if TYPE_CHECKING:
    # TraceLogHandler is defined in utils.logging; imported only for type
    # checking to avoid a runtime cycle. It is received via
    # attach_log_handler(), never instantiated here.
    from jiuwensymbiosis.utils.logging import TraceLogHandler

logger = logging.getLogger(__name__)

_TRACE_CURRENT_KEY = "trace_current_step"  # ctx.extra key holding the active TraceEntry
_TRACE_RAIL_KEY = "trace_rail"  # ctx.extra key holding the TraceRail (for sink dispatch)

_DEFAULT_CAPTURE_LOGGERS = ("jiuwensymbiosis",)
_DEFAULT_CAPTURE_LOG_LEVEL = logging.WARNING
_MAX_OUTPUT_SUMMARY = 2000  # truncate verbose tool outputs in the trace JSON


@runtime_checkable
class TraceEventSink(Protocol):
    """Sink that other rails push structured events to.

    The three built-in rails (Safety / Recovery / VisualFeedback) accept an
    optional ``trace_sink`` and call ``record_rail_event`` at their key points so
    the trace records *real* outcomes (e.g. did RecoveryRail's home() succeed)
    instead of the TraceRail guessing from exceptions.
    """

    def record_rail_event(
        self,
        *,
        rail_name: str,
        kind: str,
        detail: dict,
        success: bool,
    ) -> None: ...


def _unwrap_robot_control(tool_name: str, tool_args: Any) -> tuple[str, Any]:
    """Return (effective_tool_name, effective_args).

    ``RobotControlTool`` collapses every action behind a single ``robot_control``
    entry with ``{action, params}``; the other rails unpack this, so the trace
    should too (a trace entry named ``goto_xyzr`` is far more useful than one
    named ``robot_control``).
    """
    if tool_name == "robot_control" and isinstance(tool_args, dict):
        action = tool_args.get("action", "")
        params = tool_args.get("params", {})
        if action:
            return str(action), params if isinstance(params, dict) else {}
    return tool_name, tool_args


def _json_safe(obj: Any, *, depth: int = 0) -> Any:
    """Recursively coerce ``obj`` to JSON-serialisable primitives.

    numpy arrays → lists; numpy scalars → py scalars; bytes → base64 str;
    everything else unsupported → ``repr()`` string. Depth-bounded to avoid
    pathological nesting.
    """
    if depth > 8:
        return repr(obj)[:_MAX_OUTPUT_SUMMARY]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.ndarray):
        if obj.size > 64:
            return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, bytes):
        try:
            return base64.b64encode(obj).decode("ascii")
        except ValueError:
            return "<bytes>"
    if isinstance(obj, dict):
        return {str(k): _json_safe(v, depth=depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_json_safe(v, depth=depth + 1) for v in obj]
    # Objects with __dict__ (dataclasses, plain objects) — best effort.
    if hasattr(obj, "__dict__"):
        try:
            return {k: _json_safe(v, depth=depth + 1) for k, v in vars(obj).items() if not k.startswith("_")}
        except (TypeError, AttributeError, RecursionError):
            return repr(obj)[:_MAX_OUTPUT_SUMMARY]
    return repr(obj)[:_MAX_OUTPUT_SUMMARY]


def _summarise_output(value: Any) -> str:
    """One-line, length-capped string of a tool result for quick scanning."""
    try:
        if isinstance(value, str):
            s = value
        else:
            s = json.dumps(_json_safe(value), ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        s = repr(value)
    if len(s) > _MAX_OUTPUT_SUMMARY:
        return s[:_MAX_OUTPUT_SUMMARY] + f"…<+{len(s) - _MAX_OUTPUT_SUMMARY} chars>"
    return s


@dataclass
class TraceEntry:
    """One tool-call step in an execution trace."""

    step: int
    tool_name: str = ""
    input_params: dict = field(default_factory=dict)
    success: bool = True
    error: str | None = None
    started_at: float = 0.0
    duration_s: float = 0.0
    observation: dict | None = None  # pose/joints/extra (no raw rgb/depth)
    frame_path: str | None = None
    output_summary: str = ""
    rail_events: list[dict] = field(default_factory=list)
    log_events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "tool_name": self.tool_name,
            "input_params": _json_safe(self.input_params),
            "success": self.success,
            "error": self.error,
            "started_at": self.started_at,
            "duration_s": round(self.duration_s, 4),
            "observation": _json_safe(self.observation),
            "frame_path": self.frame_path,
            "output_summary": self.output_summary,
            "rail_events": _json_safe(self.rail_events),
            "log_events": _json_safe(self.log_events),
        }


@dataclass
class ExecutionTrace:
    """Full trace of one ``agent.invoke()`` run."""

    conversation_id: str = ""
    robot_name: str = ""
    query: str | None = None
    started_at: float = field(default_factory=time.time)
    entries: list[TraceEntry] = field(default_factory=list)
    trace_log: list[dict] = field(default_factory=list)  # log events with no active step
    workspace: str = ""
    # Frame captured at invoke start (before the first tool call). Together with
    # each step's ``frame_path`` (the *after* frame) this gives a before/after
    # pair per step: step N's before-frame is step N-1's after-frame, and step
    # 1's before-frame is this ``initial_frame_path``. Set only when
    # ``save_frames`` is on and the first observation is available.
    initial_frame_path: str | None = None
    # Pending rail/log events that arrived before any step started (rare: rails
    # firing in before_invoke). Flushed into the next entry or trace_log.
    _pending_events: list[dict] = field(default_factory=list, repr=False)
    _step_counter: int = field(default=0, repr=False)

    def new_entry(self, tool_name: str, input_params: dict, started_at: float) -> TraceEntry:
        self._step_counter += 1
        entry = TraceEntry(
            step=self._step_counter,
            tool_name=tool_name,
            input_params=input_params,
            started_at=started_at,
        )
        # Attach any events that arrived before this step started.
        if self._pending_events:
            entry.rail_events.extend(self._pending_events)
            self._pending_events.clear()
        self.entries.append(entry)
        return entry

    @property
    def current_step(self) -> int:
        """The step number of the most recently created entry (0 if none yet).

        Read-only view of the internal counter so callers (e.g. a frame_sink
        aligning a frame filename with the active step) don't reach into the
        private ``_step_counter``. Returns the last *assigned* step number, so
        it's valid between ``new_entry`` and the next ``new_entry`` — exactly
        the window in which an ``after_tool_call`` frame is captured.
        """
        return self._step_counter

    def record_rail_event(
        self,
        *,
        rail_name: str,
        kind: str,
        detail: dict,
        success: bool,
        step: int | None = None,
    ) -> None:
        event = {
            "rail_name": rail_name,
            "kind": kind,
            "detail": _json_safe(detail),
            "success": success,
            "ts": time.time(),
        }
        target = self._find_entry(step) if step is not None else self._current_entry()
        if target is not None:
            target.rail_events.append(event)
        else:
            self._pending_events.append(event)

    def record_log_event(
        self,
        *,
        logger_name: str,
        level: str,
        msg: str,
        ts: float,
        step: int | None = None,
    ) -> None:
        event = {
            "logger": logger_name,
            "level": level,
            "msg": msg[:_MAX_OUTPUT_SUMMARY],
            "ts": ts,
        }
        target = self._find_entry(step) if step is not None else self._current_entry()
        if target is not None:
            target.log_events.append(event)
        else:
            self.trace_log.append(event)

    def _current_entry(self) -> TraceEntry | None:
        return self.entries[-1] if self.entries else None

    def _find_entry(self, step: int | None) -> TraceEntry | None:
        if step is None:
            return self._current_entry()
        for e in self.entries:
            if e.step == step:
                return e
        return None

    def to_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "robot_name": self.robot_name,
            "query": self.query,
            "started_at": self.started_at,
            "entries": [e.to_dict() for e in self.entries],
            "trace_log": _json_safe(self.trace_log),
            "workspace": self.workspace,
            "initial_frame_path": self.initial_frame_path,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str)

    def run_token(self) -> str:
        """Return the run token shared by the JSON file and its frames subdir.

        Format: ``{safe_cid}_{stamp}_{usec:06d}_{pid}``. Both the trace JSON
        (``...token.json``) and the per-run frames dir (``frames/{token}/``)
        derive from this so a historical trace's ``frame_path`` references stay
        valid — no later run overwrites them (it writes into its own subdir).
        """
        dt = time.localtime(self.started_at)
        stamp = time.strftime("%Y%m%d_%H%M%S", dt)
        usec = int((self.started_at - int(self.started_at)) * 1_000_000)
        cid = self.conversation_id or "noinv"
        # Sanitise conversation_id for use as a filename component.
        safe_cid = "".join(c if c.isalnum() or c in "-_" else "_" for c in cid)[:64]
        return f"{safe_cid}_{stamp}_{usec:06d}_{os.getpid()}"

    def save(self, traces_dir: Path, *, frames_dir: Path | None = None) -> Path:
        """Write the trace JSON to ``traces_dir``. Returns the file path.

        ``frames_dir`` is accepted for symmetry but frames are written
        incrementally during the run; this call only flushes the JSON.
        """
        traces_dir = Path(traces_dir)
        traces_dir.mkdir(parents=True, exist_ok=True)
        path = traces_dir / f"{self.run_token()}.json"
        path.write_text(self.to_json(), encoding="utf-8")
        return path


def _encode_jpeg(rgb: Any, quality: int = 80) -> bytes | None:
    """Best-effort JPEG encode of an RGB ndarray. Returns raw bytes (not b64)."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
            return None
        img = Image.fromarray(rgb[..., :3].astype("uint8"))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    except (ValueError, TypeError, OSError) as exc:
        logger.warning("TraceRail: frame encode failed: %s", exc)
        return None


def _observation_snapshot(env: Any) -> dict | None:
    """Capture pose/joints/extra from the env, dropping raw rgb/depth arrays."""
    try:
        obs = env.get_observation()
    except (RuntimeError, OSError, AttributeError, ValueError) as exc:
        logger.warning("TraceRail: get_observation failed: %s", exc)
        return None
    snap: dict[str, Any] = {}
    pose = getattr(obs, "pose", None)
    if pose is not None:
        snap["pose"] = pose if isinstance(pose, dict) else repr(pose)
    joints = getattr(obs, "joints", None)
    if joints is not None:
        snap["joints"] = list(joints) if isinstance(joints, (list, tuple)) else joints
    extra = getattr(obs, "extra", None)
    if extra:
        snap["extra"] = extra
    return snap or None


class TraceRail(AgentRail):
    """Parallel rail that records a structured trace of an agent invoke.

    Enabled via ``RobotAgentConfig.enable_tracing``. Placed at ``priority = 0``
    so its ``before_tool_call`` runs first (records request time) and
    ``after_tool_call`` runs first (records post-action observation / timing).

    Args:
        session: ``RobotSession`` (used for ``env`` and ``api`` lookups).
        workspace: Traces are written under ``<workspace>/traces`` (frames under
            ``<workspace>/traces/frames/{run_token}/`` — each invoke gets its own
            subdir so ``step_NNN.jpg`` never collides across runs).
        max_entries: Cap on recorded steps (oldest dropped beyond this).
        max_frames: Cap on JPEG frames saved per invoke.
        save_frames: When True, save a JPEG after each motion/grasp step.
        console: When True, print a one-line per-step dashboard to stdout.
        jpeg_quality: 1-95 for saved frames.
        capture_loggers: Logger-name prefixes whose ``WARNING``+ records are
            captured into the trace via :class:`TraceLogHandler`.
        capture_log_level: Minimum level for captured log records.
    """

    priority = 0

    def __init__(
        self,
        session: Any,
        *,
        workspace: str,
        max_entries: int = 200,
        max_frames: int = 50,
        save_frames: bool = False,
        console: bool = False,
        jpeg_quality: int = 80,
        capture_loggers: tuple[str, ...] = _DEFAULT_CAPTURE_LOGGERS,
        capture_log_level: int = _DEFAULT_CAPTURE_LOG_LEVEL,
        traces_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self.session = session
        self.workspace = workspace
        self.max_entries = max_entries
        self.max_frames = max_frames
        self.save_frames = save_frames
        self.console = console
        self.jpeg_quality = jpeg_quality
        self.capture_loggers = tuple(capture_loggers)
        self.capture_log_level = capture_log_level

        self._trace: ExecutionTrace | None = None
        self._frames_saved = 0
        self._traces_dir = Path(traces_dir) if traces_dir else Path(workspace) / "traces"
        # Base frames dir; each invoke writes into its own run-named subdir (see
        # ``_frame_run_token`` / ``before_invoke``) so step_NNN.jpg from one run
        # never overwrites another run's frames — the trace JSON's frame_path
        # references must stay valid for replay of historical traces.
        self._frames_base_dir = self._traces_dir / "frames"
        self._frame_run_token: str | None = None
        self._log_handler: TraceLogHandler | None = None  # set by builder
        self._log_handler_loggers: tuple[str, ...] = ()

    # ------------------------------------------------------------------ accessors
    @property
    def trace(self) -> ExecutionTrace | None:
        """The in-progress (or completed) trace. None until before_invoke."""
        return self._trace

    @property
    def frames_dir(self) -> Path:
        # During an invoke this is the run-named subdir; between invokes the base.
        if self._frame_run_token is not None:
            return self._frames_base_dir / self._frame_run_token
        return self._frames_base_dir

    @property
    def traces_dir(self) -> Path:
        return self._traces_dir

    def attach_log_handler(self, handler: TraceLogHandler, loggers: tuple[str, ...]) -> None:
        """Bind a TraceLogHandler to forward captured log lines here.

        ``loggers`` records which loggers the handler was attached to so
        :meth:`detach_log_handler` can remove it from exactly those loggers.
        """
        self._log_handler = handler
        self._log_handler_loggers = tuple(loggers)
        handler.set_sink(self)

    def detach_log_handler(self) -> None:
        """Remove the TraceLogHandler from all loggers it was attached to.

        Idempotent. Called on session teardown so a handler is never left
        dangling on a long-lived logger across builds.
        """
        import logging as _logging

        handler = self._log_handler
        if handler is None:
            return
        for name in self._log_handler_loggers:
            _logging.getLogger(name).removeHandler(handler)
        handler.set_sink(None)
        self._log_handler = None
        self._log_handler_loggers = ()

    def close(self) -> None:
        """Flush any pending trace, then detach the log handler.

        The full teardown for a TraceRail — called by ``RobotSession.disconnect``
        when the agent's session ends. Between invocations, use :meth:`finalize`
        (which keeps the handler attached so the next invoke can rebind it).
        """
        try:
            self.finalize()
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("TraceRail: finalize during close failed: %s", exc)
        self.detach_log_handler()

    # ------------------------------------------------------- TraceEventSink impl
    def record_rail_event(
        self,
        *,
        rail_name: str,
        kind: str,
        detail: dict,
        success: bool,
    ) -> None:
        if self._trace is None:
            return
        self._trace.record_rail_event(
            rail_name=rail_name,
            kind=kind,
            detail=detail,
            success=success,
        )

    def record_log_event(
        self,
        *,
        logger_name: str,
        level: str,
        msg: str,
        ts: float,
        step: int | None = None,
    ) -> None:
        if self._trace is None:
            return
        self._trace.record_log_event(
            logger_name=logger_name,
            level=level,
            msg=msg,
            ts=ts,
            step=step,
        )

    # ----------------------------------------------------------------- lifecycle
    async def before_invoke(self, ctx: Any) -> None:
        inputs = getattr(ctx, "inputs", None)
        conversation_id = getattr(inputs, "conversation_id", "") or ""
        query = getattr(inputs, "query", None)
        robot_name = getattr(self.session, "name", "robot")
        self._trace = ExecutionTrace(
            conversation_id=conversation_id,
            robot_name=robot_name,
            query=query if isinstance(query, str) else None,
            workspace=self.workspace,
        )
        self._frames_saved = 0
        # Pin this invoke's run token so its frames land in a dedicated subdir
        # (``frames/{token}/step_NNN.jpg``), not the shared flat ``frames/`` —
        # matching the JSON filename so historical trace ``frame_path`` refs stay
        # valid across runs (no cross-run overwrite of step_NNN.jpg).
        self._frame_run_token = self._trace.run_token()
        ctx.extra[_TRACE_RAIL_KEY] = self
        # Capture an "initial" frame at invoke start (before any tool call) so
        # step 1 has a before-frame to pair with its after-frame in replay.
        # Step 1's before-frame is this; step N>1's before-frame is step N-1's
        # after-frame, so we only grab this one extra frame per invoke. Uses
        # step 0 so the file ``step_000.jpg`` never collides with a real step
        # (steps are 1-based). Shares the ``max_frames`` budget like any frame.
        if self.save_frames and self._frames_saved < self.max_frames:
            init_frame = self._maybe_save_frame(0)
            if init_frame is not None:
                self._trace.initial_frame_path = str(init_frame)
        # Restore log handler sink so log capture works across multiple invokes.
        if self._log_handler is not None:
            self._log_handler.set_sink(self)

    async def after_invoke(self, ctx: Any) -> None:
        self._finalize()

    # -------------------------------------------------------------- tool hooks
    async def before_tool_call(self, ctx: Any) -> None:
        if self._trace is None:
            return
        inputs = getattr(ctx, "inputs", None)
        tool_name = getattr(inputs, "tool_name", "") or ""
        tool_args = getattr(inputs, "tool_args", None)
        effective_name, effective_args = _unwrap_robot_control(tool_name, tool_args)
        entry = self._trace.new_entry(
            tool_name=effective_name,
            input_params=dict(effective_args) if isinstance(effective_args, dict) else {},
            started_at=time.time(),
        )
        ctx.extra[_TRACE_CURRENT_KEY] = entry
        if self.console:
            params = entry.input_params
            print(f"[trace] #{entry.step} {effective_name}({params}) …", flush=True)

    async def after_tool_call(self, ctx: Any) -> None:
        if self._trace is None:
            return
        entry = ctx.extra.get(_TRACE_CURRENT_KEY)
        if entry is None:
            return
        inputs = getattr(ctx, "inputs", None)
        tool_result = getattr(inputs, "tool_result", None)
        entry.duration_s = time.time() - entry.started_at
        entry.output_summary = _summarise_output(tool_result)
        # If on_tool_exception already marked this step as failed, keep that
        # verdict: openjiuwen's @rail fires ON_TOOL_EXCEPTION (except block)
        # *before* AFTER_TOOL_CALL (finally block), so a SafetyRail rejection
        # reaches after_tool_call with success=False / error already set.
        # Re-inferring success here (default True when no tool_result) would
        # wrongly mask the recorded failure — the core debugging signal.
        if entry.success and entry.error is None:
            # success inference: ToolOutput(success=...) if present, else "no error"
            success = True
            if tool_result is not None:
                success = (
                    bool(getattr(tool_result, "success", getattr(tool_result, "ok", True)))
                    if not isinstance(tool_result, (str, bytes))
                    else True
                )
            entry.success = success
        # Observation snapshot (best-effort).
        entry.observation = _observation_snapshot(getattr(self.session, "env", None))
        # Frame save (best-effort, capped).
        if self.save_frames and self._frames_saved < self.max_frames:
            frame_path = self._maybe_save_frame(entry.step)
            if frame_path is not None:
                entry.frame_path = str(frame_path)
        # Enforce max_entries: drop oldest beyond cap.
        if len(self._trace.entries) > self.max_entries:
            self._trace.entries = self._trace.entries[-self.max_entries :]
        ctx.extra.pop(_TRACE_CURRENT_KEY, None)
        if self.console:
            mark = "✅" if entry.success else "❌"
            print(
                f"[trace]   └ {mark} {entry.duration_s:.2f}s"
                + (f" | {entry.output_summary[:80]}" if entry.output_summary else ""),
                flush=True,
            )

    async def on_tool_exception(self, ctx: Any) -> None:
        if self._trace is None:
            return
        entry = ctx.extra.get(_TRACE_CURRENT_KEY)
        if entry is None:
            return
        exc = getattr(ctx, "exception", None)
        entry.success = False
        entry.error = f"{type(exc).__name__ if exc else 'Exception'}: {exc}" if exc else "tool exception"
        entry.duration_s = time.time() - entry.started_at
        # Note: a SafetyRail rejection raises ValueError in before_tool_call, so
        # this hook sees it; the rail event itself is pushed by SafetyRail via
        # the trace_sink (more precise than parsing the message here).
        if self.console:
            print(f"[trace]   └ ❌ {entry.error}", flush=True)

    # ------------------------------------------------------------------ finalize
    def finalize(self) -> Path | None:
        """Flush the trace JSON to disk. Safe to call multiple times.

        Returns the written path, or None if no trace was started / already
        flushed. Detaches the log handler so a later invoke re-attaches cleanly.
        """
        if self._trace is None:
            return None
        path = self._trace.save(self._traces_dir)
        if self._log_handler is not None:
            self._log_handler.set_sink(None)
        done = path
        self._trace = None
        self._frame_run_token = None
        return done

    def _finalize(self) -> None:
        try:
            self.finalize()
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("TraceRail: finalize failed: %s", exc)

    def _maybe_save_frame(self, step: int) -> Path | None:
        """Grab + save a frame for the given step (used by after_tool_call)."""
        env = getattr(self.session, "env", None)
        if env is None:
            return None
        try:
            obs = env.get_observation()
        except (RuntimeError, OSError, AttributeError, ValueError) as exc:
            logger.warning("TraceRail: frame grab failed: %s", exc)
            return None
        rgb = getattr(obs, "rgb", None)
        if rgb is None:
            return None
        return self._save_frame(rgb, step)

    def save_frame_for_sink(self, rgb: Any) -> str | None:
        """Save a caller-provided frame (used by VisualFeedbackRail's frame_sink).

        This lets the on-disk trace frame be the *same* frame injected into the
        agent context. Caps at ``max_frames``.
        """
        if self._frames_saved >= self.max_frames:
            return None
        # Align the frame filename with the active step number so it matches
        # entry.step; falls back to the frame counter if no trace is active.
        step = self._trace.current_step if self._trace is not None else self._frames_saved
        result = self._save_frame(rgb, step)
        return str(result) if result is not None else None

    def _save_frame(self, rgb: Any, step: int) -> Path | None:
        data = _encode_jpeg(rgb, quality=self.jpeg_quality)
        if data is None:
            return None
        frames_dir = self.frames_dir
        frames_dir.mkdir(parents=True, exist_ok=True)
        path = frames_dir / f"step_{step:03d}.jpg"
        try:
            path.write_bytes(data)
        except OSError as exc:
            logger.warning("TraceRail: frame write failed: %s", exc)
            return None
        self._frames_saved += 1
        return path


__all__ = [
    "TraceEntry",
    "ExecutionTrace",
    "TraceRail",
    "TraceEventSink",
]
