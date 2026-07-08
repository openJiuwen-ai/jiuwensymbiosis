# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""DiagnosisRail — inject a compact failure diagnosis into the next LLM turn.

Online half of the Trace Feedback Loop (``docs/trace-feedback-loop-design.md``
§4.2): on a failed step, stage a message with the current params, a small
causal chain, and system state (recovery result / pose), then flush it before
the next model call. Best-effort — a diagnosis failure never becomes a tool
failure.

Two invariants that are easy to break and not obvious from the code:

- **Two-phase injection**: ``on_tool_exception`` / ``after_tool_call`` only
  *stage* on ``ctx.extra``; ``before_model_call`` flushes via
  ``add_messages``. Injecting in after_tool_call would precede openjiuwen's
  ToolMessage and yield the illegal ``… → user(diag) → tool(result)`` order
  that OpenAI-style APIs reject (same contract as ``VisualFeedbackRail``).
- **priority = 5** (below TraceRail's 100 and the default 50): TraceRail must
  create/update the active entry first. Failure is inferred from
  ``ctx.exception`` / ``tool_result`` before the trace entry, so a priority
  reshuffle can't silently break detection. Type A/B failure channels and
  per-step dedup are covered in the design doc.
"""

from __future__ import annotations

import json
from typing import Any

from jiuwensymbiosis.agent.abstractions import AgentRail
from jiuwensymbiosis.agent.trace import _TRACE_RAIL_KEY
from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger(__name__)

_DIAG_PENDING_KEY = "diagnosis_pending"
_DIAG_INJECTED_KEY = "diagnosis_injected"


class DiagnosisRail(AgentRail):
    """Inject a compact failure diagnosis into the next LLM turn.

    Args:
        session: ``RobotSession`` (used to read the active trace via the
            TraceRail stashed on ``ctx.extra`` and, best-effort, the env pose).
        max_chars: Soft cap on the rendered message. When exceeded the causal
            chain is dropped first, then the system state, preserving the
            current step as long as possible.
        history_steps: Maximum entries in the causal chain.
        history_kinds: Rail-event ``kind`` values that mark a past entry as
            relevant to the current failure (default: safety reject + recovery).
    """

    priority = 5

    def __init__(
        self,
        session: Any,
        *,
        max_chars: int = 1500,
        history_steps: int = 3,
        history_kinds: tuple[str, ...] = ("reject", "recover"),
    ) -> None:
        super().__init__()
        self.session = session
        self.max_chars = int(max_chars)
        self.history_steps = int(history_steps)
        self.history_kinds = frozenset(history_kinds)

    # ----------------------------------------------------------- rail callbacks
    async def on_tool_exception(self, ctx: Any) -> None:
        """Type B propagation path — ``ctx.exception`` is set."""
        self._maybe_stage(ctx)

    async def after_tool_call(self, ctx: Any) -> None:
        """Type A catch-path (``tool_result.success is False``); also the finally
        hook after ``on_tool_exception`` for Type B (per-step dedup skips re-stage)."""
        self._maybe_stage(ctx)

    async def before_model_call(self, ctx: Any) -> None:
        """Flush staged diagnoses now that ToolMessages are in the context."""
        pending = ctx.extra.pop(_DIAG_PENDING_KEY, None) if hasattr(ctx, "extra") else None
        if not pending:
            return
        for text in pending:
            await self._inject(ctx, text)

    async def after_invoke(self, ctx: Any) -> None:
        """Drop staged diagnoses never flushed (no next model call to see them)."""
        if hasattr(ctx, "extra"):
            ctx.extra.pop(_DIAG_PENDING_KEY, None)

    # ------------------------------------------------------------------ staging
    def _maybe_stage(self, ctx: Any) -> None:
        """Detect failure, dedup per step, build text, stage on ``ctx.extra``.

        Dedup is by step number: Type B fires both ``on_tool_exception`` and
        ``after_tool_call`` (finally) on the same step — only the first stages.
        """
        if not hasattr(ctx, "extra"):
            return
        trace_rail = self._trace_rail(ctx)
        if trace_rail is None:
            return
        trace = getattr(trace_rail, "trace", None)
        if trace is None:
            return
        step = getattr(trace, "current_step", 0) or 0
        injected: set[int] = ctx.extra.setdefault(_DIAG_INJECTED_KEY, set())
        if step in injected:
            return
        failed, error = self._is_failed(ctx, trace)
        if not failed:
            return
        entry = trace.entries[-1] if trace.entries else None
        text = self._build_message(ctx, entry, error, trace)
        if not text:
            return
        injected.add(step)
        ctx.extra.setdefault(_DIAG_PENDING_KEY, []).append(text)

    def _is_failed(self, ctx: Any, trace: Any) -> tuple[bool, str]:
        """Decide whether the current step failed and return an error string.

        Priority (design §4.2): ``ctx.exception > tool_result.error >
        entry.error > "unknown error"``.
        """
        exc = getattr(ctx, "exception", None)
        if exc is not None:
            return True, f"{type(exc).__name__}: {exc}"
        inputs = getattr(ctx, "inputs", None)
        tool_result = getattr(inputs, "tool_result", None) if inputs is not None else None
        if tool_result is not None and not isinstance(tool_result, (str, bytes)):
            success = bool(getattr(tool_result, "success", getattr(tool_result, "ok", True)))
            if not success:
                er = getattr(tool_result, "error", None)
                return True, str(er) if er else "tool returned success=False"
        entry = trace.entries[-1] if trace.entries else None
        if entry is not None and not getattr(entry, "success", True):
            er = getattr(entry, "error", None)
            return True, str(er) if er else "trace entry marked failed"
        return False, ""

    def _build_message(self, ctx: Any, entry: Any, error: str, trace: Any) -> str:
        """Render the three-section diagnosis: current + causal + system state."""
        inputs = getattr(ctx, "inputs", None)
        # SKILL mode dispatches as robot_control{action,params}; prefer the
        # entry's tool_name (TraceRail already unwraps it) so the diagnosis
        # shows the real action, falling back to unwrapping tool_args.
        tool_name = ""
        if entry is not None:
            tool_name = getattr(entry, "tool_name", "") or ""
        if not tool_name and inputs is not None:
            raw_name = getattr(inputs, "tool_name", "") or ""
            tool_name = self._unwrap_action(raw_name, getattr(inputs, "tool_args", None)) or raw_name or "unknown"
        params = getattr(entry, "input_params", {}) if entry is not None else {}

        current_lines = [
            f"[diagnosis] step failed: {tool_name}",
            f"  error: {error}",
            f"  params: {self._fmt_params(params)}",
        ]
        rail_events = getattr(entry, "rail_events", []) if entry is not None else []
        log_events = getattr(entry, "log_events", []) if entry is not None else []
        for ev in rail_events:
            current_lines.append(f"  rail: {self._fmt_rail_event(ev)}")
        for ev in log_events:
            current_lines.append(f"  log: {self._fmt_log_event(ev)}")

        causal = self._causal_chain(entry, trace)
        system_state = self._system_state(entry, rail_events)

        parts = ["### 诊断：上一步失败", "\n".join(current_lines)]
        if causal:
            parts.append("### 相关历史（可能反复失败）\n" + "\n".join(causal))
        if system_state:
            parts.append("### 系统状态\n" + "\n".join(system_state))
        parts.append("请据此修正参数或换策略，不要用相同参数重试。")
        return self._truncate("\n\n".join(parts))

    def _causal_chain(self, current: Any, trace: Any) -> list[str]:
        """Recent related entries: same tool_name or a matching rail-event kind."""
        entries = list(trace.entries or [])
        if current is not None and entries and entries[-1] is current:
            candidates = entries[:-1]
        else:
            candidates = entries
        out: list[str] = []
        for e in reversed(candidates):
            if len(out) >= self.history_steps:
                break
            if not self._related(e, current):
                continue
            mark = "ok" if getattr(e, "success", True) else f"FAIL: {getattr(e, 'error', '') or '?'}"
            out.append(
                f"  - #{getattr(e, 'step', '?')} {getattr(e, 'tool_name', '?')}"
                f"({self._fmt_params(getattr(e, 'input_params', {}))}) → {mark}"
            )
        return out

    def _related(self, entry: Any, current: Any) -> bool:
        """Same tool name, or a rail event whose kind is in ``history_kinds``."""
        cur_name = getattr(current, "tool_name", None) if current is not None else None
        if cur_name and getattr(entry, "tool_name", "") == cur_name:
            return True
        for ev in getattr(entry, "rail_events", []) or []:
            kind = ev.get("kind") if isinstance(ev, dict) else getattr(ev, "kind", None)
            if kind in self.history_kinds:
                return True
        return False

    def _system_state(self, entry: Any, rail_events: list) -> list[str]:
        """Recovery result + current pose — tells the LLM the arm state changed."""
        out: list[str] = []
        for ev in rail_events or []:
            detail = ev.get("detail") if isinstance(ev, dict) else getattr(ev, "detail", None)
            rail_name = ev.get("rail_name") if isinstance(ev, dict) else getattr(ev, "rail_name", "")
            if rail_name == "RecoveryRail":
                if isinstance(detail, dict):
                    out.append(f"  recovery: home_ok={detail.get('home_ok')}, released_ok={detail.get('released_ok')}")
        pose = self._current_pose(entry)
        if pose is not None:
            out.append(f"  pose: {pose}")
        return out

    def _current_pose(self, entry: Any) -> Any | None:
        """Pose from the entry's observation snapshot, else best-effort from env."""
        obs = getattr(entry, "observation", None) if entry is not None else None
        if isinstance(obs, dict) and obs.get("pose") is not None:
            return obs["pose"]
        env = getattr(self.session, "env", None)
        if env is None:
            return None
        try:
            raw = env.get_observation()
        except (RuntimeError, OSError, AttributeError, ValueError) as exc:
            logger.debug("DiagnosisRail: pose read failed: %s", exc)
            return None
        pose = getattr(raw, "pose", None)
        return pose if isinstance(pose, dict) else (repr(pose) if pose is not None else None)

    # ------------------------------------------------------------------ inject
    async def _inject(self, ctx: Any, text: str) -> bool:
        """Append a synthetic user message into the ModelContext.

        Returns False (never raises) on missing ModelContext (fast-path op-ctx),
        missing ``add_messages``, or injection failure — a rail exception would
        corrupt the model call or turn a successful action into a tool failure.
        ``except Exception`` leaves ``asyncio.CancelledError`` to propagate.
        """
        mc = getattr(ctx, "context", None)
        if mc is None or not hasattr(mc, "add_messages"):
            return False
        from openjiuwen.core.foundation.llm.schema.message import UserMessage

        message = UserMessage(content=[{"type": "text", "text": text}])
        try:
            await mc.add_messages(message)
        except Exception as exc:  # noqa: BLE001 — see docstring re: CancelledError
            logger.warning("DiagnosisRail: add_messages failed: %s", exc)
            return False
        return True

    # ------------------------------------------------------------------ helpers
    def _trace_rail(self, ctx: Any) -> Any | None:
        extra = getattr(ctx, "extra", None)
        if not isinstance(extra, dict):
            return None
        return extra.get(_TRACE_RAIL_KEY)

    def _unwrap_action(self, tool_name: str, tool_args: Any) -> str:
        """Real action name for a ``robot_control`` dispatch, else ``""``.

        openjiuwen delivers ``tool_args`` as a JSON string at hook time (not a
        dict), so parse first. Mirrors SafetyRail / RecoveryRail / TraceRail.
        """
        if tool_name != "robot_control":
            return ""
        args = tool_args
        if isinstance(args, str) and args:
            try:
                args = json.loads(args)
            except ValueError:
                return ""
        if isinstance(args, dict):
            action = args.get("action", "")
            return str(action) if action else ""
        return ""

    def _fmt_params(self, params: Any) -> str:
        """Compact one-line params for the diagnosis (not the full trace)."""
        if not isinstance(params, dict) or not params:
            return "{}"
        try:
            s = repr({k: params[k] for k in list(params)[:8]})
        except (TypeError, ValueError):
            s = "<unrepr>"
        return s if len(s) <= 200 else s[:197] + "..."

    def _fmt_rail_event(self, ev: Any) -> str:
        if isinstance(ev, dict):
            return f"{ev.get('rail_name', '?')}/{ev.get('kind', '?')} {ev.get('detail', {})}"
        return f"{getattr(ev, 'rail_name', '?')}/{getattr(ev, 'kind', '?')} {getattr(ev, 'detail', {})}"

    def _fmt_log_event(self, ev: Any) -> str:
        if isinstance(ev, dict):
            return f"[{ev.get('level', '?')}] {ev.get('msg', '')}"
        return f"[{getattr(ev, 'level', '?')}] {getattr(ev, 'msg', '')}"

    def _truncate(self, text: str) -> str:
        """Drop sections causal → system_state (never the current step), then
        hard-truncate. Design §7.1: 优先保留当前步和系统状态. Sections are
        matched by header text, not index, so a missing block can't shift
        which section gets dropped."""
        if len(text) <= self.max_chars:
            return text
        for header in ("### 相关历史", "### 系统状态"):
            if len(text) <= self.max_chars:
                break
            text = self._drop_section(text, header)
        if len(text) <= self.max_chars:
            return text
        return text[: max(self.max_chars - 20, 0)] + "\n…[truncated]"

    @staticmethod
    def _drop_section(text: str, header_prefix: str) -> str:
        """Remove the ``\\n\\n``-delimited section starting with ``header_prefix``."""
        sections = text.split("\n\n")
        kept = [s for s in sections if not s.startswith(header_prefix)]
        return "\n\n".join(kept)
